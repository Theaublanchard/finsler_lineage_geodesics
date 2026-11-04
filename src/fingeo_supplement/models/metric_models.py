"""Trainable Finsler metric models.

Base ``MetricNetTrainBase`` and its concrete CFM / MFM / GAGA variants, plus
the ``FinslerMixin`` that adds the lineage-guided asymmetric penalty, and the
``FinslerCFM`` / ``FinslerMFM`` / ``FinslerGAGA`` Finsler metric models.
"""

import torch
import numpy as np
from typing import Optional

from fingeo_supplement.models.base_model import *
from fingeo_supplement.utils.frozen import *
import torch.nn.functional as F
from torch.func import jvp, vjp
from torch.distributions import Categorical
from sklearn.cluster import KMeans


class MetricNetTrainBase(ModelBase):
    """Base class for trainable metric models (local cost ``F(x, v)``).

    Subclasses provide a ``forward(x, v)`` that returns the local cost
    and a ``_compute_loss`` used during the metric training phase.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Create a metric training module.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :class:`ModelBase` (expects ``config``).
        """
        super().__init__(*args, **kwargs)

    def _prepare_batch(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """Move a ``(x, y)`` batch to device and normalize ``x``.

        Parameters
        ----------
        batch : tuple
            Tuple of ``(x, y)``: input points and class labels.

        Returns
        -------
        x : torch.Tensor (b, d)
            Normalized inputs on the model device.
        y : torch.Tensor (b,)
            Class labels on the model device.
        """

        device = self.get_device()

        x, y = batch
        x = x.to(device)
        y = y.to(device)
        x = self.normalize(x)

        return x, y

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One training step; logs ``train_loss_metric``.

        Parameters
        ----------
        batch : tuple
            One ``(x, y)`` batch. See :meth:`_prepare_batch` for details.
        batch_idx : int
            Batch index in the current epoch.

        Returns
        -------
        torch.Tensor
            The metric loss.
        """
        loss = self._compute_loss(batch)

        self.log("train_loss_metric", loss)
        return loss


class MetricNetCFM(MetricNetTrainBase):
    """Plain Euclidean metric (no learned structure) for the CFM baseline.

    The local cost is simply the Euclidean norm ``||v||``, and no
    parameter is learned in the metric phase.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Create the CFM metric.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :class:`MetricNetTrainBase`.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Local cost: Euclidean norm of the velocity.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base point (unused).
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor (b,)
            ``||v||`` per row.
        """
        return torch.norm(v, dim=-1)

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        """Zero loss (no metric parameters to learn).

        Parameters
        ----------
        batch : tuple
            Unused batch.

        Returns
        -------
        torch.Tensor
            A scalar zero tensor with grad enabled.
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def configure_optimizers(self) -> None:
        """No optimizer (no learnable parameters)."""
        return None


class MetricNetMFM(MetricNetTrainBase):
    """Metric Flow Matching (MFM) conformal metric.

    Learns a positive conformal factor ``M(x)`` via a Gaussian kernel
    over ``K`` KMeans cluster centers. The per-cluster bandwidth is set
    from the empirical within-cluster variance; the local cost is
    ``||v|| * sqrt(M(x))``.
    """

    def __init__(
        self,
        K: int = 20,
        kappa: float = 1.0,
        alpha: float = 1.0,
        epsilon: float = 1e-2,
        *args,
        **kwargs,
    ) -> None:
        """Create the MFM metric.

        Parameters
        ----------
        K : int
            Number of KMeans cluster centers.
        kappa : float
            Kernel bandwidth scale.
        alpha : float
            Exponent on the conformal denominator.
        epsilon : float
            Regularization on the conformal denominator.
        *args, **kwargs
            Forwarded to :class:`MetricNetTrainBase`.
        """
        super().__init__(*args, **kwargs)
        self.K = K
        self.clustering_model = KMeans(n_clusters=self.K)
        self.kappa = kappa
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        self.train_dataloader = None  # Will be set externally before training

        self.alpha = alpha
        self.epsilon = epsilon

    def M(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the conformal factor ``M(x)``.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.

        Returns
        -------
        torch.Tensor (b,)
            ``1 / (sum_k W_k exp(-0.5 lamda_k ||x - C_k||^2) + eps)^alpha``
            per row.
        """
        dist2 = torch.cdist(x, self.C) ** 2
        self.phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        h_x = (self.W.to(x.device) * self.phi_x).sum(dim=1)

        M_x = 1 / (h_x + self.epsilon) ** self.alpha

        return M_x.squeeze(-1)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Local cost: ``||v|| * sqrt(M(x))``.

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
        return torch.norm(v, dim=-1) * torch.sqrt(self.M(x))

    def on_before_zero_grad(self, *args, **kwargs) -> None:
        """Clamp kernel weights to stay strictly positive before optimizer step."""
        self.W.data = torch.clamp(self.W.data, min=0.0001)

    def on_train_start(self) -> None:
        """Fit KMeans on the training data and register center/bandwidth buffers.

        Requires ``train_dataloader`` to be set externally before training.
        """
        with torch.no_grad():

            data_to_fit = []
            for batch in self.train_dataloader:
                x, y = self._prepare_batch(batch)
                data_to_fit.append(x.detach().cpu())
            data_to_fit = torch.cat(data_to_fit)

            print("Fitting Clustering model...")
            self.clustering_model.fit(data_to_fit)

            clusters = self.clustering_model.cluster_centers_

            C = torch.tensor(clusters, dtype=torch.float32).to(self.device)
            self.register_buffer("C", C)
            labels = self.clustering_model.labels_
            sigmas = np.zeros((self.K, 1))

            for k in range(self.K):
                points = data_to_fit[labels == k, :]
                variance = ((points - clusters[k]) ** 2).mean(axis=0)
                sigmas[k, :] = np.sqrt(variance.mean())

            lamda = torch.tensor(
                0.5 / (self.kappa * sigmas + 1e-8) ** 2, dtype=torch.float32
            ).to(self.device)
            self.register_buffer("lamda", lamda)

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        """Regularize the conformal factor toward 1.

        Parameters
        ----------
        batch : tuple
            One ``(x, y)`` batch.

        Returns
        -------
        torch.Tensor
            Mean squared deviation of ``M(x)`` from 1.
        """
        x, y = self._prepare_batch(batch)
        loss = ((1 - self.M(x)) ** 2).mean()
        return loss


class MetricNetGAGA(MetricNetTrainBase):
    """Pullback metric via encoder + off-manifold discriminator (GAGA-style).

    The local cost is the Euclidean norm of the Jacobian of the map
    ``[f(x), disc_factor * s(x)]`` applied to ``v``, where ``f`` is an
    encoder and ``s = 1 - positive_prob`` is an off-manifold score.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        discriminator: torch.nn.Module,
        disc_factor: float = 5.0,
        *args,
        **kwargs,
    ) -> None:
        """Create the GAGA pullback metric.

        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder network ``f``.
        discriminator : torch.nn.Module
            Discriminator exposing ``positive_prob``.
        disc_factor : float
            Weight on the off-manifold score term.
        *args, **kwargs
            Forwarded to :class:`MetricNetTrainBase`.
        """
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.discriminator = discriminator
        self.disc_factor = disc_factor

    def _fn(self, x: torch.Tensor) -> torch.Tensor:
        """Compose encoder output with the off-manifold score.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Normalized base points.

        Returns
        -------
        torch.Tensor (b, z + 1)
            ``[f(x), disc_factor * s(x)]`` concatenated along the last dim.
        """
        # x is already ModelBase-normalized (done by embed model's _prepare_batch).
        # Metric pullback of [f(x), β·s(x)] where f=encoder, s=off-manifold probability.
        # Both f and s operate on the same ambient (normalized) x.
        z = self.encoder(x, normalize=False)
        s = 1 - self.discriminator.positive_prob(
            x, normalize=False
        )  # ≈0 on-manifold, ≈1 off-manifold
        return torch.cat([z, self.disc_factor * s.unsqueeze(1)], dim=1)

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Local cost: ``||J _fn(x) v||``.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor (b,)
            Pullback local cost per row.
        """
        f_x, Jf_x_v = jvp(self._fn, (x,), (v,))
        return torch.norm(Jf_x_v, dim=-1)

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        """Zero loss (encoder/discriminator trained elsewhere).

        Parameters
        ----------
        batch : tuple
            Unused batch.

        Returns
        -------
        torch.Tensor
            A scalar zero tensor with grad enabled.
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def configure_optimizers(self) -> None:
        """No optimizer (no learnable parameters in this module)."""
        return None


class FinslerMixin:
    """Mixin adding the lineage-prior Finsler penalty to a metric.

    Implements the paper's asymmetric local norm
    ``F(x, v) = ||v||_g(x) * (1 + λ * F~(x, v/||v||))`` where ``F~`` penalizes
    velocity along lineage-inconsistent class directions using the
    temperature-scaled classifier probabilities and the (padded) lineage
    adjacency matrix ``A``.
    """

    def __init__(
        self,
        tree: np.ndarray,
        temp: float,
        classifier_model: torch.nn.Module,
        lamb: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        """Create the Finsler metric mixin.

        Parameters
        ----------
        tree : np.ndarray (|C|, |C|)
            Lineage adjacency matrix ``A`` (``A[i,j]=1`` if class i
            differentiates into j), as a torch tensor or ndarray.
        temp : float
            Softmax temperature for the classifier probabilities.
        classifier_model : torch.nn.Module
            Cell-type classifier network (logits).
        lamb : float
            Weight ``λ`` on the Finsler penalty term.
        *args, **kwargs
            Forwarded to the concrete metric base class.
        """
        super().__init__(*args, **kwargs)
        self.eps = 1e-8
        self.temp = temp
        self.classifier_model = classifier_model
        self.lamb = lamb
        self.tree = self._pad_tree(tree)

    def classifier_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled classifier probabilities ``softmax(f(x)/temp)``.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.

        Returns
        -------
        torch.Tensor (b, |C|)
            Probability vector(s) over cell types.
        """
        return torch.softmax(self.classifier_model(x) / self.temp, dim=-1)

    def logit_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Raw classifier logits.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.

        Returns
        -------
        torch.Tensor (b, |C|)
            Logits over cell types.
        """
        return self.classifier_model(x)

    def _pad_tree(self, tree: np.ndarray) -> torch.Tensor:
        """Pad ``A`` to ``(|C|+1) x (|C|+1)`` with zeros (for the extra class).

        Parameters
        ----------
        tree : np.ndarray (|C|, |C|)
            Lineage adjacency matrix ``A`` of shape ``(|C|, |C|)``.

        Returns
        -------
        torch.Tensor (|C|+1, |C|+1)
            Padded matrix with a zero row/column for the extra class.
        """
        target = tree.shape[-1] + 1
        padded = torch.zeros((target, target))
        rows = min(target, tree.shape[0])
        cols = min(target, tree.shape[1])
        padded[:rows, :cols] = tree[:rows, :cols]
        return padded

    def fisher_rao(self, x: torch.Tensor) -> torch.Tensor:
        """Diagonal Fisher-Rao metric ``1 / p_c(x)`` per class.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.

        Returns
        -------
        torch.Tensor (b, |C|)
            Per-class diagonal entries.
        """
        p = self.classifier_fn(x)
        diag = 1 / (p + self.eps)
        return diag

    def riemann_norm(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Delegate to the concrete Riemannian base norm.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor (b,)
            Riemannian local norm per row.
        """
        return super().forward(x, v)

    # # Forward mode basic
    # def forward(self, x, v):
    #     riemann_term = super().forward(x, v)
    #     f_x, Jf_x_v = jvp(self.classifier_fn, (x,), (v,))

    #     u = 1 - f_x @ (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     # D = self.fisher_rao(x)
    #     D = torch.ones_like(f_x)

    #     finsler_term = torch.sum(Jf_x_v * D * u, dim=-1)
    #     finsler_term *= riemann_term / torch.norm(v, dim=-1)
    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # # Backward mode basic
    # def forward(self, x, v):
    #     riemann_term = self.riemann_norm(x, v)
    #     f_x = self.classifier_fn(x)

    #     u = 1 - f_x @ (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     _, vjp_fn = vjp(self.classifier_fn, x)
    #     h = vjp_fn(u)[0]

    #     #Crummy approximation to the dual norm
    #     # h_scale = self.riemann_norm(x, h) / torch.norm(h, dim=-1) ** 2
    #     # h = h * h_scale.unsqueeze(-1)

    #     # h_scale = self.riemann_norm(x, h) / torch.norm(h, dim=-1)
    #     # ent = torch.sum(-f_x * torch.log(f_x), dim=-1)

    #     # D = self.fisher_rao(x)

    #     finsler_term = torch.sum(v * h, dim=-1)
    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # Classwise
    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Finsler local cost ``F(x, v)``.

        Combines the concrete Riemannian norm with the lineage-prior
        penalty: ``riemann + λ * (riemann/||v||) * relu(F~(x, v))``
        where ``F~(x, v) = sum_c f_c(x) * relu(<Jf(x) v, u_c>)`` and
        ``u = 1 - (A + I)`` marks illegal class directions.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Base points.
        v : torch.Tensor (b, d)
            Velocity vectors.

        Returns
        -------
        torch.Tensor (b,)
            Finsler local cost per row.
        """
        riemann_term = super().forward(x, v)
        f_x, Jf_x_v = jvp(self.classifier_fn, (x,), (v,))

        u = 1 - (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

        # D = 1 / (f_x + self.eps)
        D = 1

        finsler_term = torch.sum(f_x * F.relu(D * Jf_x_v @ u.T), dim=-1)

        scale = riemann_term / torch.norm(v, dim=-1)

        return riemann_term + self.lamb * scale * F.relu(finsler_term)

    # Classwise logits
    # def forward(self, x, v):
    #     riemann_term = super().forward(x, v)
    #     _, Jl_x_v = jvp(self.logit_fn, (x,), (v,))
    #     f_x = self.classifier_fn(x)

    #     u = 1 - (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     finsler_term = torch.sum(f_x * F.relu(Jl_x_v @ u.T), dim=-1)

    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # Logits
    # def forward(self, x, v):
    #     riemann_term = self.riemann_norm(x, v)
    #     _, Jl_x_v = jvp(self.logit_fn, (x,), (v,))
    #     f_x = self.classifier_fn(x)

    #     u = 1 - f_x @ (self.tree.to(self.get_device()) + torch.eye(f_x.shape[-1], device=x.device))

    #     D = f_x

    #     finsler_term = torch.sum(Jl_x_v * D * u, dim=-1)
    #     return riemann_term + self.lamb * F.relu(finsler_term)

    # Riemann Conformal
    # def forward(self, x, v):
    #     riemann_term = self.riemann_norm(x, v)
    #     f_x = self.classifier_fn(x)

    #     A = self.tree.to(self.get_device())
    #     M = A + A.T + torch.eye(f_x.shape[-1], device=x.device)

    #     conformal = torch.sum(f_x * (f_x @ (1 - M)), dim=-1)

    #     return riemann_term + self.lamb * torch.norm(v, dim=-1) * torch.sqrt(conformal)


class FinslerCFM(FinslerMixin, MetricNetCFM):
    """Finsler metric over the CFM (Euclidean) Riemannian base."""

    pass


class FinslerMFM(FinslerMixin, MetricNetMFM):
    """Finsler metric over the MFM conformal Riemannian base."""

    pass


class FinslerGAGA(FinslerMixin, MetricNetGAGA):
    """Finsler metric over the GAGA pullback Riemannian base."""

    pass
