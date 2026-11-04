"""Flow-matching training wrappers.

``WrappedVectorField`` adapts a vector-field network to the ``torchdyn``
NeuralODE interface and ``FlowNetTrainBase`` trains the flow on top of a
learned embedding.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from torchvision import transforms

# from torchcfm.conditional_flow_matching import *
from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper

from fingeo_supplement.models.cfm import OTFlowMatcher

from fingeo_supplement.models.base_model import *
from torchdyn.core import NeuralODE

import random


class WrappedVectorField(torch.nn.Module):
    """Adapt an ``f(x, t)`` network to the NeuralODE ``f(t, x)`` signature."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Wrap a vector-field model.

        Parameters
        ----------
        model : torch.nn.Module
            A module callable as ``model(x, t)``.
        """
        super().__init__()
        self.model = model

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Evaluate the wrapped vector field at ``(t, x)``.

        Parameters
        ----------
        t : torch.Tensor (b,)
            Time value(s), broadcast to the batch.
        x : torch.Tensor (b, d)
            State points.

        Returns
        -------
        torch.Tensor (b, d)
            Vector field value ``model(x, t)``.
        """
        t = t.repeat(x.shape[0])
        return self.model(x, t)


# From https://github.com/kksniak/metric-flow-matching/blob/main/mfm/flow_matchers/flow_net_train.py
class FlowNetTrainBase(ModelBase):
    """Trainable vector-field (flow) network for conditional flow matching.

    Learns a time-dependent vector field ``v(x, t)`` by regressing it to the
    conditional OT flow between consecutive timepoint batches, normalized to
    the global time interval ``[t_global_min, t_global_max]``.
    """

    def __init__(
        self,
        embed_model,
        flow_net: torch.nn.Module,
        t_global_min: float,
        t_global_max: float,
        **kwargs,
    ) -> None:
        """Create the flow network training module.

        Parameters
        ----------
        embed_model
            Embedding model (``EmbedNetTrainBase``) providing the ``flow_matcher``.
        flow_net : torch.nn.Module
            Vector-field network ``f(x, t)``.
        t_global_min : float
            Minimum global timepoint across the dataset.
        t_global_max : float
            Maximum global timepoint across the dataset.
        **kwargs
            Forwarded to :class:`ModelBase` (expects ``config``).
        """
        super().__init__(**kwargs)
        self.flow_matcher = embed_model.flow_matcher
        # self.flow_matcher = OTFlowMatcher(sigma=self.config.sigma)
        self.flow_net = flow_net
        self.t_global_min = t_global_min
        self.t_global_max = t_global_max
        self.embed_model = embed_model

    def normalize_time(self, t: torch.Tensor) -> torch.Tensor:
        """Map time to ``[0, 1]`` within the global time interval.

        Parameters
        ----------
        t : torch.Tensor (b,)
            Time value(s).

        Returns
        -------
        torch.Tensor (b,)
            Normalized time in ``[0, 1]``.
        """
        return (t - self.t_global_min) / (self.t_global_max - self.t_global_min)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the vector field at ``(x, t)``.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            State points.
        t : torch.Tensor (b,)
            Time value(s).

        Returns
        -------
        torch.Tensor (b, d)
            Vector field value.
        """
        return self.flow_net(x, t)

    def _prepare_batch(
        self, batch: tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move and normalize a ``(x0, x1, t0, t1)`` batch.

        Parameters
        ----------
        batch : tuple
            Tuple of ``(x0, x1, t0, t1)``: start/end states and start/end times.

        Returns
        -------
        x0 : torch.Tensor (b, d)
            Normalized start states.
        x1 : torch.Tensor (b, d)
            Normalized end states.
        t0 : torch.Tensor (b,)
            Normalized start times.
        t1 : torch.Tensor (b,)
            Normalized end times.
        """

        device = self.get_device()

        x0, x1, t0, t1 = batch
        # BUG: `.to(device)` returns a new tensor; the result is discarded
        # so x0/x1/t0/t1 are NOT actually moved to `device`. Fix: reassign, e.g. `x0 = x0.to(device)`.
        x0.to(device)
        x1.to(device)
        t0.to(device)
        t1.to(device)
        x0 = self.normalize(x0)
        x1 = self.normalize(x1)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)

        return x0, x1, t0, t1

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        """Conditional flow-matching regression loss.

        For each batch pair, sample an intermediate location and the
        conditional flow from the flow matcher, and regress the
        network's velocity to it (MSE).

        Parameters
        ----------
        batch : tuple
            One ``(x0, x1, t0, t1)`` batch.

        Returns
        -------
        torch.Tensor
            Scalar mean flow-matching loss.
        """

        x0, x1, t0, t1 = self._prepare_batch(batch)
        loss = 0

        for i in range(x0.shape[0]):

            t, xt, dxt, _ = self.flow_matcher.sample_location_and_conditional_flow(
                x0[i], x1[i], t0[i], t1[i]
            )

            vt = self(
                xt.detach(), t.detach()
            )  # TODO: the main inefficiency, we should only do forward once outside the loop

            loss += torch.mean((vt - dxt.detach()) ** 2)

        return loss / x0.shape[0]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One training step; logs ``train_loss_flow``.

        Parameters
        ----------
        batch : tuple
            One ``(x0, x1, t0, t1)`` batch.
        batch_idx : int
            Batch index in the current epoch.

        Returns
        -------
        torch.Tensor
            The flow loss.
        """
        loss = self._compute_loss(batch)

        self.log("train_loss_flow", loss)
        return loss

    def sample_traj(
        self,
        x: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        num_samples: int,
        steps: int = 2000,
    ) -> torch.Tensor:
        """Integrate the learned vector field to sample trajectories.

        Uses a NeuralODE (dopri5 solver, adjoint sensitivity) to integrate
        the learned vector field from ``t0`` to ``t1`` for a random subset
        of the input points.

        Parameters
        ----------
        x : torch.Tensor (n, d)
            Initial states; ``num_samples`` will be drawn from these.
        t0 : torch.Tensor (n,)
            Start time.
        t1 : torch.Tensor (n,)
            End time.
        num_samples : int
            Number of trajectories to sample.
        steps : int
            Number of integration time points to return.

        Returns
        -------
        torch.Tensor (steps, num_samples, d)
            Trajectories, in the original (unnormalized) input space.
        """

        device = self.get_device()

        x = self.normalize(x)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)

        node = NeuralODE(
            WrappedVectorField(self.flow_net), solver="dopri5", sensitivity="adjoint"
        )
        indices = np.random.choice(x.shape[0], num_samples, replace=False)
        indices = torch.from_numpy(indices).to(device)
        with torch.no_grad():
            traj = node.trajectory(
                x[indices].float().to(device),
                t_span=torch.linspace(t0, t1, steps),
            ).cpu()

        traj = [self.unnormalize(x) for x in traj]
        traj = torch.stack(traj, dim=0)

        return traj

    def sample(
        self,
        x: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        num_samples: int,
        steps: int = 2000,
    ) -> torch.Tensor:
        """Sample the terminal states of trajectories (last time step).

        Parameters
        ----------
        x : torch.Tensor (n, d)
            Initial states.
        t0 : torch.Tensor (n,)
            Start time.
        t1 : torch.Tensor (n,)
            End time.
        num_samples : int
            Number of trajectories to sample.
        steps : int
            Number of integration time points.

        Returns
        -------
        torch.Tensor (num_samples, d)
            Terminal states.
        """
        traj = self.sample_traj(x, t0, t1, num_samples, steps)
        return traj[-1]

    def sample_and_weight(
        self,
        x: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        num_samples: int,
        steps: int = 2000,
    ) -> tuple[torch.Tensor, None]:
        """Sample terminal states (unweighted; weights are ``None``).

        Parameters
        ----------
        x : torch.Tensor (n, d)
            Initial states.
        t0 : torch.Tensor (n,)
            Start time.
        t1 : torch.Tensor (n,)
            End time.
        num_samples : int
            Number of trajectories to sample.
        steps : int
            Number of integration time points.

        Returns
        -------
        samples : torch.Tensor (num_samples, d)
            Terminal states.
        None
            Always ``None``; the weight is not computed in this baseline.
        """
        samples = self.sample(x, t0, t1, num_samples, steps)
        return samples, None


############################################################################################################################
# GROWTH?
############################################################################################################################

# class WrappedVectorFieldGrowth(torch.nn.Module):
#     def __init__(self, flow_model, growth_model, c, w=1.0):
#         super().__init__()
#         self.flow_model = flow_model
#         self.growth_model = growth_model
#         self.c = c
#         self.w = w

#     def forward(self, t, x, *args, **kwargs):
#         x = x[..., :-1]
#         t = t.repeat(x.shape[0])
#         v = self.flow_model(x, self.c, t)
#         h = self.growth_model(x, self.c, t)
#         return torch.cat([v, h], dim = -1)

# class GrowthFlowNetTrainBase(FlowNetTrainBase):
#     def __init__(
#         self,
#         growth_net,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.growth_net = growth_net

#     def forward(self, x, c, t):
#         return self.flow_net(x, c, t), self.growth_net(x, c, t)

#     def _compute_loss(self, batch):

#         x0, x1, t0, t1, cond = self._prepare_batch(batch)
#         t0 = self.normalize_time(t0)
#         t1 = self.normalize_time(t1)

#         loss = 0

#         for i in range(x0.shape[0]):

#             t, xt, ut, gt, log_etat = self.flow_matcher.sample_location_and_conditional_flow(x0[i], x1[i], t0[i], t1[i],
#                                                                                             self.get_device())

#             c = cond[i].unsqueeze(0).repeat(x0[i].shape[0], 1)

#             device = self.get_device()
#             mask = (torch.rand(c.shape[0], 1) > self.config.cfg_p_u).int().to(device)
#             vt, ht = self(xt, c * mask, t)

#             weight = torch.exp(log_etat).unsqueeze(1)
#             loss += torch.mean((vt - ut) ** 2 * weight)
#             loss += torch.mean((ht.squeeze(1) - gt) ** 2 * weight)


#         return loss / x0.shape[0]

#     def sample_traj_and_weight(self, x, c, t0, t1, num_samples, steps=2000):

#         device = self.get_device()
#         t0 = self.normalize_time(t0)
#         t1 = self.normalize_time(t1)

#         node = NeuralODE(WrappedVectorFieldGrowth(self.flow_net, self.growth_net,
#                                             c, self.config.cfg_w), solver="dopri5", sensitivity="adjoint")
#         indices = np.random.choice(x.shape[0], num_samples, replace=False)
#         indices = torch.from_numpy(indices).to(device)
#         x_g = np.concatenate([x[indices], np.zeros((num_samples, 1))], axis = 1) #intialize log weights at zero

#         with torch.no_grad():
#             traj = node.trajectory(
#                 torch.from_numpy(x_g).float().to(device),
#                 t_span=torch.linspace(t0, t1, steps),
#             ).cpu()

#         return traj

#     def sample_and_weight(self, x, c, t0, t1, num_samples, steps=2000):
#         traj = self.sample_traj_and_weight(x, c, t0, t1, num_samples, steps)
#         xs = traj[-1,:,:-1]
#         ws = traj[-1,:,-1]
#         ws = F.softmax(ws)
#         return xs, ws

#     def sample_traj(self, x, c, t0, t1, num_samples, steps=2000):

#         traj = self.sample_traj_and_weight(x, c, t0, t1, num_samples, steps)
#         return traj[:,:,:-1]

#     def sample(self, x, c, t0, t1, num_samples, steps=2000):
#         traj = self.sample_traj(x, c, t0, t1, num_samples, steps)
#         return traj[-1]

############################################################################################################################
# METRIC?
############################################################################################################################
