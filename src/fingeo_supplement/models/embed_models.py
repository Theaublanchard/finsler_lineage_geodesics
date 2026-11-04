"""Embedding models.

``EmbedNetTrainBase`` learns the latent embedding ``phi`` / ``psi`` used to
define the Finsler metric; ``FinslerEmbedNetTrainBase`` adds the lineage
split of the embedding (Euclidean part + asymmetric part) and the geodesic
sampling used in the embedding phase.
"""

import torch
import torch.nn as nn
from typing import Union, Callable

import numpy as np

from fingeo_supplement.models.base_model import *
from fingeo_supplement.utils.frozen import *
from fingeo_supplement.models.cfm import *
import torch.nn.functional as F
from torch.func import jvp


def sample(x: torch.Tensor, size: int) -> torch.Tensor:
    """Randomly subsample ``size`` rows of ``x`` with replacement.

    Parameters
    ----------
    x : torch.Tensor (N, ...)
        Input tensor.
    size : int
        Number of rows to draw.

    Returns
    -------
    torch.Tensor (size, ...)
        ``size`` sampled rows.
    """
    indices = torch.randint(0, x.shape[0], size=(size,))
    return x[indices]


class EmbedNetTrainBase(ModelBase):
    """Trainable embedding + geodesic model for metric flow matching.

    Learns an embedding ``phi`` (and geodesic deformation ``eta``) whose
    pullback approximates the Finsler metric ``F``, trained with an
    embedding-matching loss (``|J phi v - F|``) plus a geodesic energy
    loss (``E[F^2]``) along OT-coupled pairs.
    """

    def __init__(
        self,
        embed_net: torch.nn.Module,
        metric_model,
        geo_net: torch.nn.Module,
        t_global_min: float,
        t_global_max: float,
        *args,
        **kwargs,
    ) -> None:
        """Create the embedding training module.

        Parameters
        ----------
        embed_net : torch.nn.Module
            Embedding network ``phi: M -> R^l``.
        metric_model
            Metric model (``MetricNetTrainBase``) implementing the local cost ``F(x, v)``.
        geo_net : torch.nn.Module
            Geodesic deformation network ``eta``.
        t_global_min : float
            Minimum global timepoint.
        t_global_max : float
            Maximum global timepoint.
        *args, **kwargs
            Forwarded to :class:`ModelBase` (expects ``config``).
        """

        super().__init__(*args, **kwargs)
        self.embed_net = embed_net
        self.geo_net = geo_net
        self.metric_model = metric_model
        self.t_global_min = t_global_min
        self.t_global_max = t_global_max

        self.flow_matcher = MetricFlowMatcher(
            sigma=self.config.sigma,
            method=self.config.method,
            reg=self.config.reg,
            reg_m=self.config.reg_m,
            geo_fn=self.geo_fn,
            cost_matrix_fn=self.cost_matrix_fn,
        )

    def embed_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding map (identity if ``no_learning`` / Euclidean OT).

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.

        Returns
        -------
        torch.Tensor (b, latent_dim)
            Embedded points.
        """
        if self.config.no_learning or self.config.use_euclidean_ot:
            return x
        return self.embed_net(x)

    def geo_fn(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> Union[torch.Tensor, float]:
        """Geodesic deformation ``eta(x0, x1, t)``.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.
        t : torch.Tensor (b,)
            Time value(s).

        Returns
        -------
        torch.Tensor or float
            Deformation (0 if ``no_learning``).
        """
        if self.config.no_learning:
            return 0
        return self.geo_net(torch.cat([x0, x1], dim=-1), t)

    def cost_matrix_fn(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Squared Euclidean OT cost between embedded points.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.

        Returns
        -------
        torch.Tensor (b, b)
            Cost matrix ``||phi(x0) - phi(x1)||^2``.
        """
        return torch.cdist(self.embed_fn(x0), self.embed_fn(x1)) ** 2

    def F(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Local cost ``F(x, v)`` delegated to the metric model.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor (b,)
            Local cost per row.
        """
        return self.metric_model(x, v)

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
        x0 = x0.to(device)
        x1 = x1.to(device)
        t0 = t0.to(device)
        t1 = t1.to(device)
        x0 = self.normalize(x0)
        x1 = self.normalize(x1)
        t0 = self.normalize_time(t0)
        t1 = self.normalize_time(t1)

        return x0, x1, t0, t1

    def embed_loss(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Embedding-matching loss ``|J phi(x) v - F(x, v)|``.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor
            Mean absolute mismatch.
        """
        df_x_v = jvp(self.embed_fn, (x,), (v,))[1]
        norm_diff = torch.abs(torch.norm(df_x_v, dim=-1) - self.F(x, v))
        loss = torch.mean(norm_diff)
        return loss

    def geo_loss(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Geodesic energy loss ``0.5 * E[F(x, v)^2]``.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor
            Mean squared local cost, scaled by 0.5.
        """
        return 0.5 * torch.mean(self.F(x, v) ** 2)

    def _compute_loss(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute embedding + geodesic losses over a paired batch.

        For each pair, sample a location and conditional flow from the
        flow matcher; train the embedding to match the metric on both
        random and flow-aligned velocities, and the geodesic to keep
        energy low.

        Parameters
        ----------
        batch : tuple
            One ``(x0, x1, t0, t1)`` batch. See :meth:`_prepare_batch` for details.

        Returns
        -------
        loss_embed : torch.Tensor
            Mean embedding-matching loss.
        loss_geo : torch.Tensor
            Mean geodesic energy loss.
        """
        x0, x1, t0, t1 = self._prepare_batch(batch)
        loss_embed, loss_geo = 0, 0

        for i in range(x0.shape[0]):
            t, xt, dxt, log_etat = self.flow_matcher.sample_location_and_conditional_flow(
                x0[i],
                x1[i],
                t0[i],
                t1[i],
                ot_sample=self.config.ot_in_embed,
                time_per_batch=self.config.time_per_batch,
            )
            xt_free, dxt_free = xt.detach(), dxt.detach()

            # Train embed with random vectors towards real data
            if self.config.random_velocities:
                x0_x1 = torch.cat([x0[i], x1[i]], dim=0)
                v = sample(x0_x1, xt_free.shape[0]) - xt_free
                v /= torch.norm(v, dim=-1, keepdim=True)
                loss_embed += self.embed_loss(xt_free, v)

            # TODO: should we normalize dxt_free too?  Morally yes because we're comparing two homogeneous norms
            loss_embed += self.embed_loss(xt_free, dxt_free)
            loss_geo += self.geo_loss(xt, dxt)

        return loss_embed / x0.shape[0], loss_geo / x0.shape[0]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One training step; logs embed/geo/total losses.

        Parameters
        ----------
        batch : tuple
            One ``(x0, x1, t0, t1)`` batch.
        batch_idx : int
            Batch index in the current epoch.

        Returns
        -------
        torch.Tensor
            Sum of embedding and geodesic losses.
        """
        loss_embed, loss_geo = self._compute_loss(batch)

        self.log("train_loss_embed", loss_embed)
        self.log("train_loss_geo", loss_geo)
        self.log("train_loss_geo_embed", loss_embed + loss_geo)
        return loss_embed + loss_geo

    def _sample_geodesic(
        self, batch: tuple, timepoints, ot_sample: bool = True, weighted: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Evaluate the deterministic geodesic at the given timepoints.

        Temporarily zeros the flow-matcher noise, samples an OT-coupled
        pair, and interpolates along the learned geodesic at each
        requested timepoint.

        Parameters
        ----------
        batch : tuple
            A single ``(x0, x1, t0, t1)`` batch (asserts one pair).
        timepoints
            Times (normalized) at which to evaluate the path.
        ot_sample : bool
            If True, draw the pair from the OT coupling.
        weighted : bool
            If True, also return the log-eta weights.

        Returns
        -------
        paths : torch.Tensor (len(timepoints), N, dim)
            Path points at each requested timepoint.
        weights : torch.Tensor, optional (len(timepoints), N)
            Only returned when ``weighted=True``: log-eta weights.
        """

        old_sigma = self.flow_matcher.sigma
        self.flow_matcher.sigma = 0

        x0, x1, t0, t1 = self._prepare_batch(batch)
        assert x0.shape[0] == 1

        paths = []
        weights = []

        x0_, x1_ = x0[0], x1[0]
        t0_, t1_ = t0[0], t1[0]
        r0_, r1_ = None, None
        if ot_sample:
            x0_, x1_, r0_, r1_ = self.flow_matcher.ot_sampler.sample_plan(
                x0_, x1_
            )  # Freeze a sample of points from coupling

        for j in timepoints:
            t = torch.tensor(j)
            t = t.unsqueeze(0).repeat(x0_.shape[0])
            t.requires_grad_(True)
            _, xt, _, log_etat = self.flow_matcher.sample_location_and_conditional_flow(
                x0_, x1_, t0_, t1_, t=t, r0=r0_, r1=r1_, ot_sample=False
            )
            xt = self.unnormalize(xt)
            paths.append(xt.detach())
            weights.append(log_etat.detach())

        self.flow_matcher.sigma = old_sigma
        if weighted:
            return torch.stack(paths, dim=0), torch.stack(weights, dim=0)
        return torch.stack(paths, dim=0)

    def sample_geodesic_path(
        self, batch: tuple, num_points: int, ot_sample: bool = True, weighted: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Sample the geodesic path at ``num_points`` equally spaced times.

        Parameters
        ----------
        batch : tuple
            A single ``(x0, x1, t0, t1)`` batch.
        num_points : int
            Number of timepoints to evaluate.
        ot_sample : bool
            Draw the pair from the OT coupling.
        weighted : bool
            Also return log-eta weights.

        Returns
        -------
        paths : torch.Tensor (num_points, N, dim)
            Path points at ``num_points`` equally spaced times.
        weights : torch.Tensor, optional (num_points, N)
            Only returned when ``weighted=True``.
        """
        timepoints = np.linspace(0, 1, num_points)
        return self._sample_geodesic(batch, timepoints, ot_sample, weighted)

    def sample_geodesic_time(
        self, batch: tuple, t: float, ot_sample: bool = True, weighted: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Sample the geodesic at a single (normalized) time ``t``.

        Parameters
        ----------
        batch : tuple
            A single ``(x0, x1, t0, t1)`` batch.
        t : float
            Normalized time in ``[0, 1]``.
        ot_sample : bool
            Draw the pair from the OT coupling.
        weighted : bool
            Also return the log-eta weight.

        Returns
        -------
        path : torch.Tensor (N, dim)
            The path point at ``t``.
        weight : torch.Tensor, optional (N,)
            Only returned when ``weighted=True``: the log-eta weight.
        """
        _, _, t0, t1 = self._prepare_batch(batch)
        t0 = t0[0]
        t1 = t1[0]
        timepoints = torch.Tensor([t])
        timepoints = self.normalize_time(timepoints)
        timepoints = (timepoints - t0) / (t1 - t0)

        if weighted:
            paths, weights = self._sample_geodesic(batch, timepoints, ot_sample, weighted)
            return paths[0], weights[0]
        paths = self._sample_geodesic(batch, timepoints, ot_sample, weighted)
        return paths[0]


class FinslerEmbedNetTrainBase(EmbedNetTrainBase):
    """Embedding model with the asymmetric Finsler distance split.

    Splits the latent into a Euclidean part ``phi`` and an asymmetric
    part ``psi`` (projected onto a learnable ``beta``) so the distance
    ``||phi(x) - phi(y)|| + <psi(x) - psi(y), beta>`` (ReLU'd) matches the
    local Finsler metric.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Create the Finsler embedding training module.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :class:`EmbedNetTrainBase` (expects ``config``).
        """

        super().__init__(*args, **kwargs)
        self.beta = nn.Parameter(torch.randn(self.config.latent_dim // 2))

    # TODO: this really only makes sense if skip = False
    def embed_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``x`` (full latent; the Finsler split is applied downstream).

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.

        Returns
        -------
        torch.Tensor (b, latent_dim)
            Embedded points.
        """
        return self.embed_net(x)

    def embed_loss(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Asymmetric embedding-matching loss.

        Matches ``||J phi v|| + <J psi v, beta>`` to ``F(x, v)`` using the
        split latent (first half = ``psi``, rest = ``phi``).

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor
            Mean absolute mismatch.
        """
        df_x_v = jvp(self.embed_fn, (x,), (v,))[1]
        dphi, dpsi = (
            df_x_v[:, self.beta.shape[0] :],
            df_x_v[:, : self.beta.shape[0]] @ -self.beta,
        )
        # BUG: the asymmetric term `dpsi` is used WITHOUT a ReLU, but the paper's
        # local approximation is `||J phi v|| + <J psi v, beta>_+` and `cost_matrix_fn`
        # in this same class DOES apply F.relu. Inconsistent; wrap dpsi in F.relu.
        norm_diff = torch.abs(torch.norm(dphi, dim=-1) + dpsi - self.F(x, v))
        loss = torch.mean(norm_diff)
        return loss

    def cost_matrix_fn(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Asymmetric Finsler OT cost between embedded points.

        ``M = ||phi0 - phi1|| + relu(psi0 - psi1)``, squared.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.

        Returns
        -------
        torch.Tensor (b, b)
            Squared asymmetric cost matrix.
        """
        z0 = self.embed_fn(x0)
        z1 = self.embed_fn(x1)
        phi0, psi0 = z0[:, self.beta.shape[0] :], z0[:, : self.beta.shape[0]] @ self.beta
        phi1, psi1 = z1[:, self.beta.shape[0] :], z1[:, : self.beta.shape[0]] @ self.beta
        M = torch.cdist(phi0, phi1)
        M += F.relu(psi0.unsqueeze(1) - psi1.unsqueeze(0))
        return M**2
