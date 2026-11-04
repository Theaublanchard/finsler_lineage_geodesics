"""Copyright (c) Meta Platforms, Inc. and affiliates."""

# https://github.com/kksniak/metric-flow-matching/blob/main/mfm/flow_matchers/ema.py
from __future__ import annotations

import torch


class EMA(torch.nn.Module):
    """Exponential moving-average (EMA) wrapper around a model.

    Keeps a shadow copy of all trainable parameters and, when toggling
    between train and eval modes, swaps the shadow (EMA) weights in for
    evaluation and restores the live weights when training resumes.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        """Wrap ``model`` with an EMA.

        Parameters
        ----------
        model : torch.nn.Module
            The module whose trainable parameters are averaged.
        decay : float
            Base EMA decay rate (warmed up over the first updates).
        """
        super().__init__()
        self.model = model
        self.decay = decay
        if hasattr(self.model, "time_geopath"):
            self.time_geopath = self.model.time_geopath

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))

        self.shadow_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )
        self.backup_params = []

    # #TODO: test this
    # def reinit(self):

    #     self.model.reinit()

    #     self.num_updates.zero_()

    #     self.shadow_params = torch.nn.ParameterList(
    #         [
    #             torch.nn.Parameter(p.clone().detach(), requires_grad=False)
    #             for p in model.parameters()
    #             if p.requires_grad
    #         ]

    #     # Clear backup parameters
    #     self.backup_params = []

    #     # Ensure we're in training mode after reinit
    #     self.train(True)

    def train(self, mode: bool) -> None:
        """Toggle train/eval mode, swapping EMA weights in and out.

        Parameters
        ----------
        mode : bool
            True for training, False for evaluation.
        """
        if self.training and mode == False:
            # Switching from train mode to eval mode.  Backup the model parameters and
            # overwrite with shadow params
            self.backup()
            self.copy_to_model()
        elif not self.training and mode == True:
            # Switching from eval to train mode.  Restore the `backup_params`
            self.restore_to_model()

        super().train(mode)

    def update_ema(self) -> None:
        """Perform one EMA update of the shadow parameters.

        Uses a warmup decay ``min(decay, (1 + n) / (10 + n))`` where
        ``n`` is the number of updates so far.
        """
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs):
        """Forward pass; delegates to the wrapped model."""
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        """Copy the shadow (EMA) parameters into the wrapped model."""
        # copy the shadow (ema) parameters to the model
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shaddow, param in zip(self.shadow_params, params):
            param.data.copy_(shaddow.data)

    def backup(self) -> None:
        """Snapshot the current live parameters of the wrapped model."""
        # Backup the current model parameters
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) -> None:
        """Restore the backed-up (pre-EMA) parameters to the wrapped model."""
        # Restores the backed up parameters to the model.
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)
