"""Evaluation utilities: empirical-Wasserstein distance of predicted trajectories.

Compares sampled trajectories (from the embedding model) to the held-out
ground-truth points at an intermediate timepoint, via entropic OT
(geomloss) or exact OT (POT).
"""

from __future__ import annotations

import numpy as np
import torch

import ot
from geomloss import SamplesLoss
from fingeo_supplement.datasets.process import *


def ot_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    p: int = 1,
    library: str = "geomloss",
) -> float:
    """Compute the empirical optimal-transport distance between two clouds.

    Dispatches to either the geomloss (entropic / Sinkhorn) or the POT
    (exact) implementation based on ``library``.

    Parameters
    ----------
    x : torch.Tensor (N, d)
        Source point cloud.
    y : torch.Tensor (M, d)
        Target point cloud.
    a : torch.Tensor (N,), optional
        Source marginal weights; uniform weights are used when None.
    b : torch.Tensor (M,), optional
        Target marginal weights; uniform weights are used when None.
    p : int
        Transport exponent (1 for W1, 2 for W2).
    library : str
        One of ``"geomloss"`` or ``"pot"``.

    Returns
    -------
    float
        The OT distance between the two distributions.
    """

    if p not in [1, 2]:
        raise NotImplementedError(f"p = {p} not allowed.")

    if library == "geomloss":
        return geomloss_ot_dist(x, y, a, b, p)
    elif library == "pot":
        return pot_ot_dist(x, y, a, b, p)
    else:
        raise NotImplementedError(f"Library {library} not known.")


def geomloss_ot_dist(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    p: int = 2,
) -> float | None:
    """Compute the Sinkhorn (entropic) OT distance using geomloss.

    Parameters
    ----------
    x : torch.Tensor (N, d)
        Source point cloud.
    y : torch.Tensor (M, d)
        Target point cloud.
    a : torch.Tensor (N,), optional
        Source marginal weights; uniform weights are used when None.
    b : torch.Tensor (M,), optional
        Target marginal weights; uniform weights are used when None.
    p : int
        Transport exponent (2 for W2, 1 for W1).

    Returns
    -------
    float or None
        The (sqrt-scaled for p=2) entropic OT distance, or None for
        unsupported ``p``.
    """
    x = x.cuda()
    y = y.cuda()
    loss = SamplesLoss(loss="sinkhorn", p=p, blur=0.05)
    if a is None:
        a = torch.ones(x.shape[0]) / x.shape[0]
    a = a.cuda().float()
    if b is None:
        b = torch.ones(y.shape[0]) / y.shape[0]
    b = b.cuda().float()
    if p == 2:
        return torch.sqrt(2 * loss(a, x, b, y)).item()
    if p == 1:
        return loss(a, x, b, y).item()
    return None


def pot_ot_dist(
    X: torch.Tensor,
    Y: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    p: int = 1,
) -> float:
    """Compute the exact OT distance using POT (ot.emd2).

    Parameters
    ----------
    X : torch.Tensor (N, d)
        Source point cloud.
    Y : torch.Tensor (M, d)
        Target point cloud.
    a : torch.Tensor (N,), optional
        Source marginal weights; uniform weights are used when None.
    b : torch.Tensor (M,), optional
        Target marginal weights; uniform weights are used when None.
    p : int
        Transport exponent (2 for squared euclidean, 1 for euclidean).

    Returns
    -------
    float
        The exact OT cost between the two distributions.
    """
    if a is None:
        a = torch.ones(X.shape[0]) / X.shape[0]
    if b is None:
        b = torch.ones(Y.shape[0]) / Y.shape[0]
    if p == 2:
        M = ot.dist(X, Y, metric='sqeuclidean')
        dist = ot.emd2(a, b, M) ** 0.5
    elif p == 1:
        M = ot.dist(X, Y, metric='euclidean')
        dist = ot.emd2(a, b, M)
    # ot_epsilon = 0.05
    # dist = ot.sinkhorn2(a, b, M, ot_epsilon, method='sinkhorn_log') ** (1/p)
    return dist


def predict(
    embed_model,
    adata,
    t0: float,
    t: float,
    t1: float,
    p: int = 1,
    num_traj: int = 2000,
    batch_size: int = 500,
    library: str = "geomloss",
    icfm: bool = False,
) -> float:
    """Evaluate the embedding model by matching predicted paths to ground truth.

    Samples geodesic paths between the held-out timepoints ``t0`` and
    ``t1`` at an intermediate time ``t`` (or the flow-matched prediction
    when ``icfm``), then measures the OT distance to the true points.

    Parameters
    ----------
    embed_model
        A trained embedding (or flow) model exposing ``sample_geodesic_time``.
    adata
        AnnData with ``obs['timepoint']`` and ``obsm['X_pca']``.
    t0 : float
        Start timepoint.
    t : float
        Intermediate timepoint.
    t1 : float
        End timepoint.
    p : int
        Transport exponent passed to ``ot_dist``.
    num_traj : int
        Number of trajectories to sample.
    batch_size : int
        Batch size used when sampling trajectories.
    library : str
        OT backend passed to ``ot_dist``.
    icfm : bool
        If True, use the flow-matching (CFM) prediction instead of
        geodesic interpolation.

    Returns
    -------
    float
        The OT distance between predicted and true points at time ``t``.
    """

    adata_obs = adata[adata.obs['timepoint'].isin([t0, t1])]

    dataset = extract_paired_dataset(adata_obs)
    train_dataset = ShufflingDataset(
        dataset, batch_size
    )  # TODO: replace with ShufflingOTDataset when that actually works
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    assert num_traj % batch_size == 0, "simpler weighting when batch_size | num_traj"
    samples = []
    weights = []
    total = 0
    while total < num_traj // batch_size:
        batch = next(iter(train_dataloader))
        x, w = embed_model.sample_geodesic_time(batch, t, weighted=True, ot_sample=not icfm)
        samples.append(x)
        weights.append(w)
        total += 1
    samples = torch.cat(samples, dim=0).float()
    weights = torch.cat(weights, dim=0).float()
    weights = torch.softmax(weights, dim=0)

    true = (
        torch.tensor(adata[adata.obs['timepoint'] == t].obsm['X_pca'])
        .to(embed_model.device)
        .float()
    )

    emp_dist = ot_dist(samples, true, weights, None, p=p, library=library)

    return emp_dist
