"""Common base class for all trainable Lightning modules in the pipeline.

Provides the shared interface (forward / compute_loss / training_step /
configure_optimizers), a small MLP normalization helper, and a linear-warmup
AdamW optimizer factory used by every phase's model.
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
from typing import Dict

from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper


class ModelBase(pl.LightningModule):
    """Base class shared by all trainable Lightning modules in the pipeline.

    Holds the training configuration, dataset-normalization buffers
    (``mean`` / ``std``), and a common AdamW optimizer with a linear
    warmup learning-rate schedule. All downstream modules (classifier,
    metric, embedding, flow) subclass this.
    """

    def __init__(
        self,
        config,
        *args,
        **kwargs,
    ) -> None:
        """Create a base model.

        Parameters
        ----------
        config : omegaconf.OmegaConf
            Training configuration (lr, warmup_steps, pc_dim, ...).
            Typically a hydra/OmegaConf mapping with attribute access.
        *args
            Extra positional arguments forwarded to LightningModule.
        **kwargs
            Extra keyword arguments forwarded to LightningModule.
        """
        super().__init__(*args, **kwargs)
        self.config = config
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps
        # self.mean = torch.tensor(0)
        # self.std = torch.tensor(1)

        self.register_buffer("mean", torch.zeros(1, self.config.pc_dim))
        self.register_buffer("std", torch.tensor(1.0))

    def get_device(self) -> torch.device:
        """Return the device this module currently resides on."""
        return self.device

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> None:
        """Placeholder forward pass; subclasses must override.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Input points, where ``d`` is the ambient/PC dimension.
        t : torch.Tensor (b,)
            Time value(s) paired with each point.
        """
        pass

    def _compute_loss(self, batch: tuple) -> torch.Tensor:
        """Placeholder loss computation; subclasses must override.

        Parameters
        ----------
        batch : tuple
            One training batch from the phase-specific dataloader.
        """
        pass

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize ``x`` with the registered mean/std buffers.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Points to normalize.

        Returns
        -------
        torch.Tensor (b, d)
            ``(x - mean) / std``.
        """
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize`.

        Parameters
        ----------
        x : torch.Tensor (b, d)
            Normalized points to bring back to the original scale.

        Returns
        -------
        torch.Tensor (b, d)
            ``x * std + mean``.
        """
        return x * self.std + self.mean

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Run one training step and log the loss as ``train_loss``.

        Parameters
        ----------
        batch : tuple
            One training batch.
        batch_idx : int
            Index of the batch in the current epoch.

        Returns
        -------
        torch.Tensor
            The scalar loss tensor.
        """
        loss = self._compute_loss(batch)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Dict:
        """Build the AdamW optimizer with a linear warmup LR schedule.

        Returns
        -------
        dict
            Lightning optimizer dict: AdamW at ``config.lr`` plus a
            LambdaLR scheduler that linearly ramps up over
            ``warmup_steps`` optimizer steps.
        """
        optimizer = AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            warmup_steps = self.warmup_steps
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0  # Or add decay logic here

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',  # update every optimizer step
            'frequency': 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
