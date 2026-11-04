"""Optimal-transport plan sampler shared by the metric models.

``GeneralOTPlanSampler`` computes (Sinkhorn / unbalanced) couplings between
batches and exposes a ``sample_plan`` interface used by the CFM / MFM metric
models during the metric training phase.
"""

from __future__ import annotations

import math
import warnings
from typing import Union, Callable

import numpy as np
import torch
import os
import sys
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.conditional_flow_matching import pad_t_like_x, ConditionalFlowMatcher
from fingeo_supplement.utils.frozen import *

from torchcfm.optimal_transport import OTPlanSampler
import ot as pot


class GeneralOTPlanSampler(OTPlanSampler):
    """OT plan sampler with a user-supplied cost-matrix function.

    Supports exact or unbalanced OT; the unbalanced branch returns
    per-point mass rates derived from the plan margins.
    """

    def __init__(
        self,
        cost_matrix_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *args: Union,
        **kwargs: Union,
    ) -> None:
        """Create the general OT plan sampler.

        Parameters
        ----------
        cost_matrix_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            Function ``(x0, x1) -> cost matrix``.
        *args, **kwargs
            Forwarded to :class:`OTPlanSampler` (expects ``method``, etc.).
        """
        super().__init__(*args, **kwargs)
        self.cost_matrix_fn = cost_matrix_fn
        self.method = kwargs['method']

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """Solve the OT plan for the given cost matrix.

        Flattens high-dim inputs, builds the cost via ``cost_matrix_fn``,
        optionally normalizes it, and falls back to a uniform plan on
        numerical failure.

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start batch.
        x1 : torch.Tensor (b, d)
            End batch.

        Returns
        -------
        np.ndarray (N, M)
            The transport plan ``p``.
        """

        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = self.cost_matrix_fn(x0, x1)

        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_plan(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Sample a coupled pair (with unbalanced rates if applicable).

        Parameters
        ----------
        x0 : torch.Tensor (b, d)
            Start batch.
        x1 : torch.Tensor (b, d)
            End batch.

        Returns
        -------
        x0_i : torch.Tensor (b, d)
            Sampled start points (indexed from ``x0``).
        x1_j : torch.Tensor (b, d)
            Sampled end points (indexed from ``x1``).
        r0_i : np.ndarray (b,)
            Unbalanced mass rate for each start point (zeros for balanced).
        r1_j : np.ndarray (b,)
            Unbalanced mass rate for each end point (zeros for balanced).
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=True)

        if self.method == "unbalanced":
            pi_tilde = pi / np.sum(pi)
            r0 = -np.log(np.sum(pi_tilde, axis=1)) - np.log(x0.shape[0])
            r1 = -np.log(np.sum(pi_tilde, axis=0)) - np.log(x0.shape[0])
        else:
            r0 = np.zeros(x0.shape[0])
            r1 = np.zeros(x1.shape[0])

        return x0[i], x1[j], r0[i], r1[j]
