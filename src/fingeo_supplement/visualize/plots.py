"""Matplotlib / UMAP visualization helpers for trajectory and model output.

Restricts an AnnData to a perturbation, embeds it with UMAP, and plots
sampled points / trajectories / vector fields alongside the data.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import umap


def restrict_adata(adata: ad.AnnData, value) -> ad.AnnData:
    """Keep only cells matching a perturbation target (plus the ctrl-inj).

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to filter.
    value
        The ``gene_target`` value to keep alongside ``'ctrl-inj'``.

    Returns
    -------
    anndata.AnnData
        The filtered dataset.
    """
    return adata[
        (adata.obs['gene_target'] == value) | (adata.obs['gene_target'] == 'ctrl-inj')
    ]


def save_umap(adata_local: ad.AnnData) -> tuple[ad.AnnData, umap.UMAP]:
    """Fit a UMAP model on ``X_pca`` and store the embedding in ``obsm``.

    Parameters
    ----------
    adata_local : anndata.AnnData
        The AnnData object to embed (must have ``obsm['X_pca']``).

    Returns
    -------
    adata_local : anndata.AnnData
        The same object, now with ``obsm['X_umap']`` set.
    umap_model : umap.UMAP
        The fitted UMAP model (for transforming new points).
    """

    X = adata_local.obsm['X_pca']
    umap_model = umap.UMAP().fit(X)  # pca of ctrl adata

    adata_local.obsm["X_umap"] = umap_model.embedding_
    return adata_local, umap_model


def plot_samples(model, adata: ad.AnnData, value, conditions, num_traj: int, t):
    """Plot model-sampled points at a timepoint against the real data.

    Restricts to the perturbation + control, fits UMAP, samples trajectories
    from the model, and overlays the transformed samples on the control
    UMAP plot.

    Parameters
    ----------
    model
        A model exposing ``sample(X0, c, t0, t1, num_traj)``.
    adata : anndata.AnnData
        The full AnnData object.
    value
        The ``gene_target`` to focus on.
    conditions
        Mapping from perturbation to a condition vector.
    num_traj : int
        Number of trajectories to sample.
    t
        The timepoint at which to compare.
    """
    adata_local = restrict_adata(adata, value)
    adata_local, umap_model = save_umap(adata_local)
    adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
    adata_per = adata_local[adata_local.obs['gene_target'] == value]

    timepoints = sorted(adata_per.obs['timepoint'].unique().tolist())
    index = timepoints.index(t)
    if index == 0:
        print("failure")
        return -1

    t0 = timepoints[index - 1]
    t1 = t

    c = conditions[value]
    c = c.unsqueeze(0).repeat(num_traj, 1)

    X0 = adata_per[adata_per.obs['timepoint'] == timepoints[index - 1]].obsm['X_pca']
    X1 = adata_per[adata_per.obs['timepoint'] == timepoints[index]].obsm['X_pca']

    samples = model.sample(X0, c, t0, t1, num_traj).detach().cpu()
    samples_umap = umap_model.transform(samples)

    fig, (ax0, ax1) = plt.subplots(2, figsize=(16, 16))

    sc.pl.umap(
        adata_local[adata_local.obs['timepoint'] == t],
        color="gene_target",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax0,
        show=False,
    )

    sc.pl.umap(
        adata_ctrl[adata_ctrl.obs['timepoint'] == t],
        # color="tissue",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax1,
        show=False,
    )

    ax1.scatter(
        samples_umap[:, 0],
        samples_umap[:, 1],
        marker=".",
        linestyle="-",
        alpha=0.7,
        lw=1.5,
        color='#ff7f0e',
    )

    plt.show()


def plot_trajectories(
    model, adata: ad.AnnData, value, conditions, num_traj: int
) -> list[np.ndarray]:
    """Plot full trajectories sampled by the model alongside the data.

    Samples trajectories from the model over the full time range, transforms
    them with the fitted UMAP model, and plots them over several UMAP
    panels.

    Parameters
    ----------
    model
        A model exposing ``sample_traj(X0, c, t0, t1, num_traj)``.
    adata : anndata.AnnData
        The full AnnData object.
    value
        The ``gene_target`` to focus on.
    conditions
        Mapping from perturbation to a condition vector.
    num_traj : int
        Number of trajectories to sample.

    Returns
    -------
    trajectories : list[np.ndarray]
        The per-trajectory point arrays (UMAP-transformed).
    """
    adata_local = restrict_adata(adata, value)
    adata_local, umap_model = save_umap(adata_local)

    adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
    adata_per = adata_local[adata_local.obs['gene_target'] == value]

    timepoints = sorted(adata_per.obs['timepoint'].unique().tolist())
    t0, t1 = min(timepoints), max(timepoints)

    c = conditions[value]
    c = c.unsqueeze(0).repeat(num_traj, 1)

    X0 = adata_per[adata_per.obs['timepoint'] == min(timepoints)].obsm['X_pca']

    traj = model.sample_traj(X0, c, t0, t1, num_traj)

    trajectories = [traj[:, i].numpy() for i in range(traj.shape[1])]
    trajectories_umap = [umap_model.transform(traj) for traj in trajectories]

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, figsize=(16, 16))

    sc.pl.umap(
        adata_local,
        color="tissue",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax0,
        show=False,
    )

    sc.pl.umap(
        adata_local,
        color="gene_target",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax1,
        show=False,
    )

    sc.pl.umap(adata_per, color="timepoint", ax=ax2, show=False)

    sc.pl.umap(adata_per, color="timepoint", ax=ax3, show=False)

    for traj in trajectories_umap:
        ax3.plot(traj[:, 0], traj[:, 1], marker=".", linestyle="-", alpha=0.7, lw=1.5)

    plt.tight_layout()
    plt.show()

    return trajectories


def plot_vector_field(model, adata: ad.AnnData, value, conditions, num_traj: int):
    """Plot the model's vector field (step directions) over a UMAP panel.

    Samples trajectories, transforms them with UMAP, and draws quiver
    arrows representing the local step directions.

    Parameters
    ----------
    model
        A model exposing ``sample_traj(X0, c, t0, t1, num_traj)``.
    adata : anndata.AnnData
        The full AnnData object.
    value
        The ``gene_target`` to focus on.
    conditions
        Mapping from perturbation to a condition vector.
    num_traj : int
        Number of trajectories to sample.
    """
    adata_local = restrict_adata(adata, value)
    adata_local, umap_model = save_umap(adata_local)

    adata_ctrl = adata_local[adata_local.obs['gene_target'] == 'ctrl-inj']
    adata_per = adata_local[adata_local.obs['gene_target'] == value]

    timepoints = sorted(adata_per.obs['timepoint'].unique().tolist())
    t0, t1 = min(timepoints), max(timepoints)

    c = conditions[value]
    c = c.unsqueeze(0).repeat(num_traj, 1)

    X0 = adata_per[adata_per.obs['timepoint'] == min(timepoints)].obsm['X_pca']

    traj = model.sample_traj(X0, c, t0, t1, num_traj)

    trajectories = [traj[:, i].numpy() for i in range(traj.shape[1])]
    trajectories_umap = [umap_model.transform(traj) for traj in trajectories]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(16, 16))

    sc.pl.umap(
        adata_per,
        color="tissue",
        # Setting a smaller point size to get prevent overlap
        size=2,
        ax=ax1,
        show=False,
    )

    sc.pl.umap(adata_per, color="timepoint", ax=ax2, show=False)

    for traj in trajectories_umap:
        # print(traj[0], traj[1])
        diffs = np.diff(traj, axis=0)
        ax2.quiver(
            traj[0, 0],
            traj[0, 1],
            diffs[0, 0],
            diffs[0, 1],
            angles='xy',
            scale_units='xy',
            color='r',
            linewidth=0.2,
            alpha=0.5,
        )

    plt.show()
