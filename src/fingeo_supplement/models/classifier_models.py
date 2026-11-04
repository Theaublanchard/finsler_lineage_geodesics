"""Cell-type classifier training wrapper.

``ClassifierNetTrainBase`` wraps an MLP (``classifier_net``) in a Lightning
module with the training step and loss used to learn the cell-type
predictor in the first phase of the pipeline.
"""

from __future__ import annotations

import os
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from torchvision import transforms
import torch.nn.functional as F

from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper

from fingeo_supplement.models.base_model import ModelBase


class ClassifierNetTrainBase(ModelBase):
    """Trainable wrapper for the cell-type classifier.

    Trains a classifier ``f: R^n -> Delta^{|C|}`` with a cross-entropy
    loss, optionally regularized by a KL divergence to a dummy prior
    that pushes one class (the "other" class) to a specified probability.
    The classifier's temperature-scaled softmax is later used by the
    Finsler metric to penalize lineage-inconsistent velocities.
    """

    def __init__(
        self,
        classifier_net: torch.nn.Module,
        *args,
        **kwargs,
    ) -> None:
        """Create the classifier training module.

        Parameters
        ----------
        classifier_net : torch.nn.Module
            The network mapping normalized inputs to logits.
        *args, **kwargs
            Forwarded to :class:`ModelBase` (expects ``config``).
        """
        super().__init__(*args, **kwargs)
        self.classifier_net = classifier_net
        self.dummy_prior: float = self.config.dummy_prior
        self.dummy_kl_weight: float = self.config.dummy_kl_weight
        self.epsilon: float = self.config.classifier.epsilon

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize raw input and pass through the classifier.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Raw (unnormalized) input points.

        Returns
        -------
        torch.Tensor (b, |C|)
            Classifier logits over the ``|C|`` cell-type classes.
        """
        x = self.normalize(x)
        return self.classifier_net(x)

    def forward(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """Classifier forward pass.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Input points.
        normalize : bool
            If True, standardize ``x`` with the model's mean/std before
            the network pass.

        Returns
        -------
        torch.Tensor (b, |C|)
            Classifier logits over the ``|C|`` cell-type classes.
        """
        if normalize:
            x = self.normalize(x)
        return self.classifier_net(x)

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

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        """Cross-entropy loss with optional KL-to-dummy-prior term.

        The dummy prior assigns ``dummy_prior`` mass to the last class
        and the remainder uniformly to the other classes; the KL term
        is scaled by ``dummy_kl_weight``.

        Parameters
        ----------
        batch : tuple
            One ``(x, y)`` batch. See :meth:`_prepare_batch` for details.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss.
        """
        x, y = self._prepare_batch(batch)
        noise = (self.epsilon * torch.randn_like(x)).float()
        logits = self.classifier_net(x + noise)
        ce_loss = F.cross_entropy(logits, y)

        kl_loss = torch.tensor(0.0, device=logits.device)
        if self.dummy_kl_weight > 0:
            num_logits = logits.shape[-1]
            prior = torch.full((num_logits,), 1.0 / num_logits, device=logits.device)
            if num_logits > 1:
                base_prob = (1.0 - self.dummy_prior) / (num_logits - 1)
                prior.fill_(base_prob)
                prior[-1] = self.dummy_prior
            log_probs = F.log_softmax(logits, dim=-1)
            kl_loss = F.kl_div(log_probs, prior, reduction='batchmean', log_target=False)

        loss = ce_loss + self.dummy_kl_weight * kl_loss
        if self.dummy_kl_weight > 0:
            self.log("train_ce", ce_loss.detach())
            self.log("train_kl", kl_loss.detach())
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """One training step; logs ``train_loss_classifier``.

        Parameters
        ----------
        batch : tuple
            One ``(x, y)`` batch.
        batch_idx : int
            Batch index in the current epoch.

        Returns
        -------
        torch.Tensor
            The classifier loss.
        """
        loss = self._compute_loss(batch)

        self.log("train_loss_classifier", loss)
        return loss
