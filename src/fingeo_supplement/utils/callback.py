"""PyTorch-Lightning callbacks for the training pipeline.

``DatasetUpdateCallback`` refreshes the OT coupling on a schedule,
``BestModelCallback`` keeps the best checkpoint per phase, and
``MeanChangeEarlyStopping`` stops training when the loss change is small.
"""

from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import numpy as np


class DatasetUpdateCallback(Callback):
    """Refresh an OT-coupled training dataset each epoch.

    Calls ``dataset.update(epoch, embed_net)`` at the start of every
    training epoch (see :class:`ShufflingOTDataset.update`).
    """

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Forward the epoch index and embed net to the dataset's ``update``.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The training loop.
        pl_module : pytorch_lightning.LightningModule
            The Lightning module (provides ``embed_net``).
        """
        dataset = trainer.train_dataloader.dataset
        if hasattr(dataset, "update"):
            dataset.update(trainer.current_epoch, pl_module.embed_net)


class BestModelCallback(Callback):
    """Tracks training loss each epoch and restores the best model weights at the end of training."""

    def __init__(self, monitor: str = "train_loss"):
        """Create the best-model callback.

        Parameters
        ----------
        monitor : str
            Name of the logged metric to track (lower is better).
        """
        self.monitor = monitor
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.best_epoch = 0

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Record the epoch with the best (lowest) monitored loss.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The training loop (provides ``callback_metrics``).
        pl_module : pytorch_lightning.LightningModule
            The Lightning module whose state dict to snapshot.
        """
        loss = trainer.callback_metrics.get(self.monitor)
        if loss is None:
            return
        loss_val = loss.item()
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_epoch = trainer.current_epoch
            self.best_state_dict = {k: v.clone() for k, v in pl_module.state_dict().items()}

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Restore the best state dict captured during training.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The training loop.
        pl_module : pytorch_lightning.LightningModule
            The Lightning module to load the best weights into.
        """
        if self.best_state_dict is not None:
            pl_module.load_state_dict(self.best_state_dict)
            print(
                f"Restored best model from epoch {self.best_epoch} with {self.monitor}={self.best_loss:.6f}"
            )


class MeanChangeEarlyStopping(Callback):
    """Stop training when the windowed mean of the loss stops changing.

    Compares the mean of the last ``window_size`` epochs to the mean of
    the ``window_size`` epochs before it; stops when the absolute
    difference drops below ``delta_threshold`` (and at least
    ``min_epochs`` have been seen).
    """

    def __init__(
        self, window_size: int = 10, delta_threshold: float = 1e-4, min_epochs: int = 20
    ):
        """Create the mean-change early-stopping callback.

        Parameters
        ----------
        window_size : int
            Number of epochs in each mean window.
        delta_threshold : float
            Minimum acceptable change in the mean loss.
        min_epochs : int
            Minimum number of epochs before stopping is allowed.
        """
        self.window_size = window_size
        self.delta_threshold = delta_threshold
        self.min_epochs = min_epochs
        self.train_losses = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Accumulate the epoch loss and trigger stopping on plateau.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The training loop.
        pl_module : pytorch_lightning.LightningModule
            The Lightning module.
        """
        loss = trainer.callback_metrics.get("train_loss")
        if loss is None:
            return

        self.train_losses.append(loss.item())

        if (
            len(self.train_losses) < 2 * self.window_size
            or len(self.train_losses) < self.min_epochs
        ):
            return

        prev_window = self.train_losses[-2 * self.window_size : -self.window_size]
        curr_window = self.train_losses[-self.window_size :]
        delta = abs(np.mean(curr_window) - np.mean(prev_window))

        if delta < self.delta_threshold:
            print(
                f"\nEarly stopping triggered. Δmean(train_loss) = {delta:.2e} < {self.delta_threshold}"
            )
            trainer.should_stop = True
