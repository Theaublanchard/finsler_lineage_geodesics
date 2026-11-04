"""Torch datasets and samplers for the trajectory inference pipeline.

Provides a small ``Config`` dataclass, a ``sample`` helper, and the
``ShufflingDataset`` / ``ShufflingOTDataset`` wrappers that batch paired
samples (and optionally recompute the OT coupling) for the embedding phase.
"""

import numpy as np
import torch
import scipy
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
import os
import pickle
import torch
from torch.utils.data import Dataset, Sampler, DataLoader, TensorDataset
import pytorch_lightning as pl
import random

from fingeo_supplement.utils.ot import *


class Config(dict):
    """A dict subclass exposing keys as attributes.

    Lets callers write ``config.lr`` instead of ``config["lr"]``.
    """

    def __getattr__(self, item: str):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key: str, value):
        self[key] = value

    def __delattr__(self, item: str):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def sample(X, batch_size: int):
    """Randomly draw ``batch_size`` rows of ``X`` (with replacement if needed).

    Parameters
    ----------
    X
        Array of points (first axis = number of points).
    batch_size : int
        Number of rows to draw.

    Returns
    -------
    ``batch_size`` sampled rows of ``X``.
    """
    replace = batch_size > X.shape[0]
    indices = np.random.choice(X.shape[0], batch_size, replace=replace)
    return X[indices]


class ShufflingDataset(Dataset):
    """Dataset of ``(X0, X1, t0, t1)`` pairs with per-item random shuffling.

    Each ``__getitem__`` draws ``batch_size`` rows from the start and end
    point clouds independently (independent shuffling, no OT prior).
    """

    def __init__(self, obj_list, batch_size: int):
        """Create the shuffling dataset.

        Parameters
        ----------
        obj_list
            List of ``(X0, X1, t0, t1)`` tuples.
        batch_size : int
            Rows to draw from each cloud per item.
        """

        self.obj_list = obj_list
        self.batch_size = batch_size

    def __getitem__(self, index: int):
        """Draw a shuffled ``(x0, x1, t0, t1)`` sample for one pair.

        Parameters
        ----------
        index : int
            Index into ``obj_list``.

        Returns
        -------
        x0 : torch.Tensor (batch_size, d)
            Sampled start points.
        x1 : torch.Tensor (batch_size, d)
            Sampled end points.
        t0 : torch.Tensor
            Start time (tensor).
        t1 : torch.Tensor
            End time (tensor).
        """
        obj = self.obj_list[index]
        X0, X1, t0, t1 = obj

        x0 = sample(X0, self.batch_size)
        x1 = sample(X1, self.batch_size)

        return x0, x1, torch.tensor(t0), torch.tensor(t1)

    def __len__(self):
        return len(self.obj_list)


class ShufflingOTDataset(Dataset):
    """Paired dataset with a precomputed OT coupling prior.

    Unlike :class:`ShufflingDataset`, the start and end points are drawn
    from an OT-coupled plan (shared indices) and periodically refreshed
    using the current embedding network via :meth:`update`.
    """

    def __init__(self, obj_list, batch_size: int, update_epoch_rate: int = 50):
        """Create the OT shuffling dataset.

        Parameters
        ----------
        obj_list
            List of ``(X0, X1, t0, t1)`` tuples (raw point clouds).
        batch_size : int
            Rows to draw per item.
        update_epoch_rate : int
            Refresh the OT coupling every this many epochs.
        """

        self.obj_list = obj_list
        self.batch_size = batch_size
        self.update_epoch_rate = update_epoch_rate
        self.ot_list = []

    def __getitem__(self, index: int):
        """Draw a coupled ``(x, y, t0, t1)`` sample for one pair.

        Parameters
        ----------
        index : int
            Index into ``ot_list``.

        Returns
        -------
        x : torch.Tensor (batch_size, d)
            Sampled start points.
        y : torch.Tensor (batch_size, d)
            Coupled end points (same indices as ``x``).
        t0 : torch.Tensor
            Start time (tensor).
        t1 : torch.Tensor
            End time (tensor).
        """
        obj = self.ot_list[index]
        X, Y, t0, t1 = obj

        replace = self.batch_size > X.shape[0]
        indices = np.random.choice(X.shape[0], self.batch_size, replace=replace)

        # Share indices because we have prior OT
        x = X[indices]
        y = Y[indices]

        return x, y, torch.tensor(t0), torch.tensor(t1)

    def __len__(self):
        return len(self.obj_list)

    def update(self, epoch: int, embed_net):
        """Periodically recompute OT samples with the current embedding.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        embed_net
            Embedding network used to compute the OT cost.
        """
        if epoch % self.update_epoch_rate == 0:
            self.recompute_ot_samples(embed_net)

    @torch.no_grad()
    def embed(self, X, Y, embed_net):
        """Embed two point clouds in batches with ``embed_net``.

        Parameters
        ----------
        X : torch.Tensor (N, d)
            Start point cloud.
        Y : torch.Tensor (M, d)
            End point cloud.
        embed_net
            Embedding network.

        Returns
        -------
        embedded_X : torch.Tensor (N, latent_dim)
            Embedding of ``X``.
        embedded_Y : torch.Tensor (M, latent_dim)
            Embedding of ``Y``.
        """
        embed_net.eval()
        embedded_X = []
        embedded_Y = []
        device = next(embed_net.parameters()).device

        xloader = DataLoader(
            TensorDataset(X), batch_size=2048, shuffle=False  # TODO: Hard-coded?
        )
        yloader = DataLoader(
            TensorDataset(Y), batch_size=2048, shuffle=False  # TODO: Hard-coded?
        )

        with torch.no_grad():
            for xb in xloader:
                xb = xb[0].to(device, non_blocking=True)
                ex = embed_net(xb)
                embedded_X.append(ex.cpu())

            for yb in yloader:
                yb = yb[0].to(device, non_blocking=True)
                ey = embed_net(yb)
                embedded_Y.append(ey.cpu())

        # Concatenate all results
        embedded_X = torch.cat(embedded_X, dim=0)
        embedded_Y = torch.cat(embedded_Y, dim=0)

        embed_net.train()

        return embedded_X, embedded_Y

    def recompute_ot_samples(self, embed_net):
        """Recompute the OT-coupled pair list using ``embed_net``.

        For every raw pair, embed both clouds, sample a coupling plan
        (KeOps Sinkhorn), and store the coupled points.

        Parameters
        ----------
        embed_net
            Embedding network defining the OT cost.
        """
        self.ot_list = []
        for obj in self.obj_list:
            X, Y, t0, t1 = obj
            X_emb, Y_emb = self.embed(X, Y, embed_net)
            i, j = sample_from_coupling(
                X_emb, Y_emb, n_samples=X.shape[0] + Y.shape[0], indices=True
            )
            self.ot_list.append((X[i], Y[j], t0, t1))
