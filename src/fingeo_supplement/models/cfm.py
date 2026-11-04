"""Flow-matching metric models.

Wraps the ``torchcfm`` optimal-transport planners in PyTorch-Lightning
trainable modules: plain ``OTFlowMatcher`` / ``UOTFlowMatcher`` and the
``MetricFlowMatcher`` used as the metric in the Finsler pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Callable
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.conditional_flow_matching import pad_t_like_x, ConditionalFlowMatcher
from torch.func import jvp
from fingeo_supplement.utils.frozen import *
from fingeo_supplement.models.ot_sampler import GeneralOTPlanSampler

############################################################################################################################


class OTFlowMatcher(ConditionalFlowMatcher):
    """Conditional flow matcher with exact-OT pair sampling.

    Samples OT-coupled pairs, interpolates the intermediate location
    ``x_t``, and returns the straight-line conditional flow
    ``u_t = (x1 - x0) / (t_max - t_min)``.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Create the OT flow matcher.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :class:`ConditionalFlowMatcher` (expects ``sigma``).
        """
        super().__init__(*args, **kwargs)
        self.ot_sampler = OTPlanSampler(method="exact")

    def compute_mu_t(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, t_min: float, t_max: float
    ) -> torch.Tensor:
        """Linear interpolation mean ``mu_t`` between ``x0`` and ``x1``.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.
        t : torch.Tensor (b,)
            Time (broadcastable to the batch).
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b, d)
            Interpolated mean.
        """

        t = pad_t_like_x(t, x0)
        return (t_max - t) / (t_max - t_min) * x0 + (t - t_min) / (t_max - t_min) * x1

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor,
        t_min: float,
        t_max: float,
    ) -> torch.Tensor:
        """Sample an intermediate location ``x_t = mu_t + sigma_t * epsilon``.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.
        t : torch.Tensor (b,)
            Time.
        epsilon : torch.Tensor (b, d)
            Noise sample.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b, d)
            Sampled location at ``t``.
        """
        mu_t = self.compute_mu_t(x0, x1, t, t_min, t_max)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_min: float,
        t_max: float,
        t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an OT pair, an intermediate location, and the conditional flow.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start batch.
        x1 : torch.Tensor (b, d)
            End batch.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.
        t : torch.Tensor (b,), optional
            Optional fixed time; if None, sampled uniformly.

        Returns
        -------
        t : torch.Tensor (b,)
            The time at which the location and flow were sampled.
        xt : torch.Tensor (b, d)
            The sampled intermediate location.
        ut : torch.Tensor (b, d)
            The straight-line conditional flow.
        """

        if t is None:
            t = torch.rand(x0.shape[0])
        t = t.type_as(x0)
        t = t * (t_max - t_min) + t_min

        x0, x1 = self.ot_sampler.sample_plan(x0, x1)

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)

        return t, xt, ut

    def compute_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        xt: torch.Tensor,
        t_min: float,
        t_max: float,
    ) -> torch.Tensor:
        """Straight-line conditional flow ``u_t = (x1 - x0) / (t_max - t_min)``.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.
        t : torch.Tensor (b,)
            Time (padded to batch).
        xt : torch.Tensor (b, d)
            Intermediate location (unused).
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b, d)
            Conditional flow.
        """
        del xt
        t = pad_t_like_x(t, x0)
        return (x1 - x0) / (t_max - t_min)


############################################################################################################################


class UOTPlanSampler(OTPlanSampler):
    """Unbalanced-OT plan sampler returning per-point growth rates."""

    # TODO: sanity check this?  The theory uses pi directly, but I think rescaling shouldn't change much?
    def sample_plan(
        self, x0: torch.Tensor, x1: torch.Tensor, replace: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Sample an unbalanced OT plan with per-point mass rates ``r0, r1``.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start batch.
        x1 : torch.Tensor (b, d)
            End batch.
        replace : bool
            Whether to sample with replacement.

        Returns
        -------
        x0_i : torch.Tensor (b, d)
            Sampled start points.
        x1_j : torch.Tensor (b, d)
            Sampled end points.
        r0_i : np.ndarray (b,)
            Unbalanced mass rate for each start point (zeros for balanced).
        r1_j : np.ndarray (b,)
            Unbalanced mass rate for each end point (zeros for balanced).
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        pi_tilde = pi / np.sum(pi)
        r0 = -np.log(np.sum(pi_tilde, axis=1)) - np.log(x0.shape[0])
        r1 = -np.log(np.sum(pi_tilde, axis=0)) - np.log(x0.shape[0])

        return x0[i], x1[j], r0[i], r1[j]


class UOTFlowMatcher(OTFlowMatcher):
    """Unbalanced-OT flow matcher (adds growth rate and log-eta weight)."""

    def __init__(self, reg: float, reg_m: float, *args, **kwargs) -> None:
        """Create the unbalanced flow matcher.

        Parameters
        ----------
        reg : float
            Sinkhorn regularization.
        reg_m : float
            Unbalanced (mass) regularization.
        *args, **kwargs
            Forwarded to :class:`OTFlowMatcher`.
        """
        super().__init__(*args, **kwargs)
        self.ot_sampler = UOTPlanSampler(method="unbalanced", reg=reg, reg_m=reg_m)
        # self.ot_sampler = UOTPlanSampler(method="exact", reg=reg, reg_m=reg_m) #DEBUG

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_min: float,
        t_max: float,
        device: torch.device,
        t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an unbalanced OT pair, location, flow, growth rate, log-eta.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start batch.
        x1 : torch.Tensor (b, d)
            End batch.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.
        device : torch.device
            Device for the growth/weight tensors.
        t : torch.Tensor (b,), optional
            Optional fixed time; if None, sampled uniformly.

        Returns
        -------
        t : torch.Tensor (b,)
            The sampled time.
        xt : torch.Tensor (b, d)
            The sampled intermediate location.
        ut : torch.Tensor (b, d)
            The straight-line conditional flow.
        gt : torch.Tensor (b,)
            The conditional growth rate ``g_t``.
        log_etat : torch.Tensor (b,)
            The interpolated log-mass weight ``log(eta_t)``.
        """

        if t is None:
            t = torch.rand(x0.shape[0])
        t = t.type_as(x0)
        t = t * (t_max - t_min) + t_min

        x0, x1, r0, r1 = self.ot_sampler.sample_plan(x0, x1)
        r0 = torch.from_numpy(r0).to(device)
        r1 = torch.from_numpy(r1).to(device)

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)
        gt = self.compute_conditional_growth(r0, r1, t_min, t_max)
        log_etat = self.compute_log_eta(r0, r1, t, t_min, t_max)

        return t, xt, ut, gt, log_etat

    def compute_conditional_growth(
        self, r0: torch.Tensor, r1: torch.Tensor, t_min: float, t_max: float
    ) -> torch.Tensor:
        """Conditional growth rate ``g_t = (r1 - r0) / (t_max - t_min)``.

        Parameters
        ----------
        r0 : torch.Tensor (b,)
            Unbalanced mass rate at the start endpoint.
        r1 : torch.Tensor (b,)
            Unbalanced mass rate at the end endpoint.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b,)
            Growth rate.
        """
        return (r1 - r0) / (t_max - t_min)

    def compute_log_eta(
        self, r0: torch.Tensor, r1: torch.Tensor, t: torch.Tensor, t_min: float, t_max: float
    ) -> torch.Tensor:
        """Linearly interpolate the log-mass weight ``log(eta_t)``.

        Parameters
        ----------
        r0 : torch.Tensor (b,)
            Unbalanced mass rate at the start endpoint.
        r1 : torch.Tensor (b,)
            Unbalanced mass rate at the end endpoint.
        t : torch.Tensor (b,)
            Time.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b,)
            Interpolated log-mass weight.
        """
        return (t_max - t) / (t_max - t_min) * r0 + (t - t_min) / (t_max - t_min) * r1


############################################################################################################################


class MetricFlowMatcher(OTFlowMatcher):
    """Flow matcher with a metric-geodesic deformation term.

    Adds a geodesic bending coefficient ``gamma(t)`` (peaking at the
    midpoint) weighted by a learned geodesic deformation ``geo_fn`` to
    the interpolated mean, enabling curved (non-straight) transport.
    """

    def __init__(
        self,
        geo_fn: Callable,
        cost_matrix_fn: Callable,
        method: str = "exact",
        reg: float = -1,
        reg_m: float = -1,
        *args,
        **kwargs,
    ) -> None:
        """Create the metric flow matcher.

        Parameters
        ----------
        geo_fn : Callable
            Geodesic deformation ``eta(x0, x1, t)``.
        cost_matrix_fn : Callable
            Function returning the OT cost matrix between two batches.
        method : str
            OT method (``exact`` / ``unbalanced``).
        reg : float
            Sinkhorn regularization.
        reg_m : float
            Mass regularization.
        *args, **kwargs
            Forwarded to :class:`OTFlowMatcher`.
        """
        super().__init__(*args, **kwargs)
        self.ot_sampler = GeneralOTPlanSampler(
            method=method,
            reg=reg,
            reg_m=reg_m,
            cost_matrix_fn=cost_matrix_fn,
            normalize_cost=True,
        )
        self.geo_fn = geo_fn

    def gamma(self, t: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
        """Geodesic bending coefficient (peaks at the interval midpoint).

        ``gamma(t) = 1 - ((t - t_min)/(t_max - t_min))^2 - ((t_max - t)/(t_max - t_min))^2``

        Parameters
        ----------
        t : torch.Tensor (b,)
            Time.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b,)
            Bending coefficient.
        """
        return (
            1.0 - ((t - t_min) / (t_max - t_min)) ** 2 - ((t_max - t) / (t_max - t_min)) ** 2
        )

    def compute_mu_t(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, t_min: float, t_max: float
    ) -> torch.Tensor:
        """Interpolated mean plus the metric geodesic deformation.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.
        t : torch.Tensor (b,)
            Time.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b, d)
            Deformed interpolated mean.
        """
        return (
            (t_max - t) / (t_max - t_min) * x0
            + (t - t_min) / (t_max - t_min) * x1
            + self.gamma(t, t_min, t_max) * self.geo_fn(x0, x1, t)
        )

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_min: float,
        t_max: float,
        t: Optional[torch.Tensor] = None,
        r0: Optional[np.ndarray] = None,
        r1: Optional[np.ndarray] = None,
        ot_sample: bool = True,
        time_per_batch: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an OT pair, deformed location, conditional flow, log-eta.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start batch.
        x1 : torch.Tensor (b, d)
            End batch.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.
        t : torch.Tensor (b,), optional
            Optional fixed time (requires grad); if None, sampled uniformly.
        r0 : np.ndarray (b,), optional
            Optional unbalanced mass rate at the start; if None, zeros.
        r1 : np.ndarray (b,), optional
            Optional unbalanced mass rate at the end; if None, zeros.
        ot_sample : bool
            If True, draw the pair from the OT coupling.
        time_per_batch : int
            Repetition factor for each sampled point.

        Returns
        -------
        t : torch.Tensor (b,)
            The sampled time (with grad).
        xt : torch.Tensor (b, d)
            The deformed sampled location.
        ut : torch.Tensor (b, d)
            The conditional flow (time-derivative of ``xt``).
        log_etat : torch.Tensor (b,)
            The interpolated log-mass weight.
        """

        if r0 is None:
            r0 = np.zeros(x0.shape[0])
        if r1 is None:
            r1 = np.zeros(x1.shape[0])
        if ot_sample:
            x0, x1, r0, r1 = self.ot_sampler.sample_plan(x0, x1)
        r0 = torch.from_numpy(r0).to(x0.device)
        r1 = torch.from_numpy(r1).to(x0.device)

        x0 = x0.repeat((time_per_batch, 1))
        x1 = x1.repeat((time_per_batch, 1))
        r0 = r0.repeat((time_per_batch,))
        r1 = r1.repeat((time_per_batch,))

        if t is None:
            t = torch.rand(x0.shape[0], requires_grad=True)
        t = t.type_as(x0)
        t = t * (t_max - t_min) + t_min

        eps = self.sample_noise_like(x0)
        xt, ut = self.compute_xt_and_conditional_flow(x0, x1, t, eps, t_min, t_max)
        log_etat = self.compute_log_eta(r0, r1, t, t_min, t_max)
        return t, xt, ut, log_etat

    def compute_xt_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        t_min: float,
        t_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute location and its time-derivative (conditional flow) via JVP.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start points.
        x1 : torch.Tensor (b, d)
            End points.
        t : torch.Tensor (b,)
            Time (differentiable).
        eps : torch.Tensor (b, d)
            Noise sample.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        xt : torch.Tensor (b, d)
            The location at ``t``.
        ut : torch.Tensor (b, d)
            The time-derivative of the location (the conditional flow).
        """
        t = pad_t_like_x(t, x0)

        def f(tt):
            t_padded = pad_t_like_x(tt, x0)
            return self.sample_xt(x0, x1, t_padded, eps, t_min, t_max)

        xt, ut = jvp(f, (t,), (torch.ones_like(t),))
        return xt, ut.squeeze(-1)

    def compute_log_eta(
        self, r0: torch.Tensor, r1: torch.Tensor, t: torch.Tensor, t_min: float, t_max: float
    ) -> torch.Tensor:
        """Linearly interpolate the log-mass weight ``log(eta_t)``.

        Parameters
        ----------
        r0 : torch.Tensor (b,)
            Unbalanced mass rate at the start endpoint.
        r1 : torch.Tensor (b,)
            Unbalanced mass rate at the end endpoint.
        t : torch.Tensor (b,)
            Time.
        t_min : float
            Minimum time.
        t_max : float
            Maximum time.

        Returns
        -------
        torch.Tensor (b,)
            Interpolated log-mass weight.
        """
        return (t_max - t) / (t_max - t_min) * r0 + (t - t_min) / (t_max - t_min) * r1
