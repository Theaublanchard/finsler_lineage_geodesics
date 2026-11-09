"""Export the paper's qualitative figures to ``outputs/<dataset>/``.

This is the script-based counterpart of the plotting cells in
``zebrafish.ipynb`` / ``mouse_lineage.ipynb`` / ``holdout_toy.ipynb``. It trains
one model (single seed) and saves the standard set of figures the paper shows:

* ``river_plot.png``        — classifier cell-type probability along N geodesics
                              (the ``fig:zebrafish-two-panel`` /
                              ``fig:mouse-blood-two-panel`` river plots)
* ``lineage_tree.png``      — the directed lineage-tree prior (from ``uns['tree']``)
* ``scatter_data.png``      — data projected to 2D, colored by timepoint / type
* ``geodesic_paths.png``    — learned geodesic trajectories overlaid on the data
* ``classifier_pred.png``   — ground-truth vs. predicted (2D datasets only)
* ``classifier_confidence.png`` / ``classifier_entropy.png`` (2D datasets only)

It does NOT run the multi-seed W1 sweep (see ``run_sweeps.sh`` / ``sweep.py``)
and does NOT require WandB.

Usage
-----
    # Zebrafish CNS, Finsler variant (the paper's river plot):
    uv run python scripts/export_plots.py --dataset zebrafish_cns --finsler

    # CFM-only variant, for the side-by-side comparison panel:
    uv run python scripts/export_plots.py --dataset zebrafish_cns --no-finsler

    # Synthetic (2D), both classifiers heatmaps included:
    uv run python scripts/export_plots.py --dataset synthetic --finsler --dim 2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")  # headless rendering before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from fingeo_supplement.simulate.simulate import process_fake_data, sample_path  # noqa: E402
from fingeo_supplement.scripts.run_model import (  # noqa: E402
    build_paired_dataloader,
    build_singleton_dataloader,
    remove_all_forward_hooks,
    run_full_model,
)
from fingeo_supplement.datasets.process import (
    extract_paired_dataset,
    process_data,
)  # noqa: E402
from fingeo_supplement.datasets.dataset import ShufflingDataset  # noqa: E402
from fingeo_supplement.utils.hydra import load_config  # noqa: E402
from fingeo_supplement.paths import FINGEO_OUT_ROOT  # noqa: E402

AVAILABLE_DATASETS = [
    "zebrafish_cns",
    "zebrafish_arch",
    "mouse_blood",
    "mouse_brain",
    "cite",
    "synthetic",
]


def _method_slug(metric: str, finsler: bool) -> str:
    return f"{metric}_finsler" if finsler else metric


def _apply_common_config(config, metric: str, finsler: bool) -> None:
    """Apply the paper/sweep settings shared by every dataset run."""
    OmegaConf.set_struct(config, False)
    config.metric = metric
    config.finsler.use = bool(finsler)
    config.finsler.lamb = 0.5
    config.use_wandb = False
    config.no_learning = False

    # Classifier settings used by the sweep / notebooks.
    config.classifier.num_layers = 2
    config.classifier.hidden_dim = 128
    config.classifier.epsilon = 0.05


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# 2D projection helpers (pure numpy; no extra dependencies)
# --------------------------------------------------------------------------- #
def _make_2d_projector(X_data) -> callable:
    """Return a function mapping ambient points to 2D.

    Identity-ish (truncation) when already <= 2D; otherwise PCA (top-2
    right singular vectors) fitted on ``X_data`` so data and geodesic paths
    share the same embedding.
    """
    X = np.asarray(X_data, dtype=float)
    if X.shape[1] <= 2:
        return lambda Z: np.asarray(Z, dtype=float)[:, :2]
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = Vt[:2]  # (2, d)

    def project(Z) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        return (Z - mean) @ comps.T

    return project


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _cell_type_labels(adata, n_classes: int) -> list[str]:
    itos = adata.uns.get("itos", {})
    return [itos[i] if i in itos else f"class_{i}" for i in range(n_classes)]


def _build_plot_paired_dataloader(config, adata, n_traj: int):
    """Paired dataloader whose inner batch yields exactly ``n_traj`` traj."""
    dataset = extract_paired_dataset(adata)
    train_dataset = ShufflingDataset(dataset, batch_size=n_traj)
    return DataLoader(train_dataset, batch_size=1, shuffle=True)


def _get_geodesic_batch(paired_dataloader):
    """Draw one usable geodesic batch from the plot dataloader.

    The batch is a ``(x0, x1, t0, t1)`` tuple; ``x0`` is whatever the outer
    DataLoader collates to (2D ``(N, d)`` or 3D ``(1, N, d)``). Both are
    acceptable — the model's ``sample_geodesic_path`` consumes it directly,
    mirroring the notebooks.
    """
    for _ in range(10):
        try:
            batch = next(iter(paired_dataloader))
        except StopIteration:
            continue
        if (
            batch is not None
            and len(batch) >= 2
            and batch[0] is not None
            and len(batch[0]) > 0
        ):
            return batch
    raise RuntimeError("Could not draw a usable geodesic batch.")


def plot_river(
    classifier_model, embed_model, plot_dl, adata, num_points: int, out_path: Path
) -> bool:
    """Stackplot of classifier cell-type probability along N geodesics."""
    batch = _get_geodesic_batch(plot_dl)
    device = embed_model.get_device()

    paths = embed_model.sample_geodesic_path(batch, num_points=num_points)
    # paths: (num_points, N, dim)
    paths_flat = paths.reshape(-1, paths.shape[-1])
    n_traj = paths.shape[1]

    with torch.no_grad():
        logits = classifier_model.classify(
            torch.tensor(np.asarray(paths_flat, dtype=float), device=device).float()
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    probs = probs.reshape(paths.shape[0], paths.shape[1], -1)  # (T, N, C)
    n_classes = probs.shape[-1]
    labels = _cell_type_labels(adata, n_classes)

    n_panels = min(n_traj, 10)
    fig, axes = plt.subplots(n_panels, figsize=(12, 2.2 * n_panels), sharex=True, sharey=True)
    flat = np.atleast_1d(axes).flatten()
    time_axis = np.linspace(0, 1, paths.shape[0])
    colors = plt.cm.tab20.colors[:n_classes]

    for i in range(n_panels):
        traj = probs[:, i]  # (T, C)
        flat[i].stackplot(time_axis, traj.T, labels=labels, alpha=0.8, colors=colors)
        flat[i].set_xlim(0, 1)
        flat[i].set_ylim(0, 1)

    handles, _ = flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        title="Predicted Cell Type",
    )
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_lineage_tree(adata, out_path: Path) -> bool:
    """Render the directed lineage tree prior (``uns['tree']``) with networkx."""
    import networkx as nx

    tree = np.asarray(adata.uns["tree"], dtype=float)
    itos = adata.uns.get("itos", {i: f"c{i}" for i in range(tree.shape[0])})
    G = nx.DiGraph()
    G.add_nodes_from(list(itos.values()))
    for i in range(tree.shape[0]):
        for j in range(tree.shape[1]):
            if tree[i, j] > 0 and i != j:
                G.add_edge(itos[i], itos[j])

    fig, ax = plt.subplots(figsize=(10, 8))
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42, k=2.0)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            node_size=1400,
            connectionstyle="arc3,rad=0.1",
        )
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=1400, node_color="#cfe8ff", edgecolors="black"
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    else:
        ax.text(0.5, 0.5, "no directed edges in tree prior", ha="center", va="center")
    ax.set_title("Lineage tree prior (directed)")
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_scatter_data(adata, project, out_path: Path) -> bool:
    """Data projected to 2D, colored by timepoint (and, inset, cell type)."""
    X2 = project(np.asarray(adata.obsm["X_pca"], dtype=float))
    tp = adata.obs["timepoint"].values
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=tp, cmap="viridis", s=4, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Timepoint")
    ax.set_title("Data (2D) colored by timepoint")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_geodesic_paths(
    embed_model, plot_dl, X_data, project, num_points: int, out_path: Path
) -> bool:
    """Learned geodesic trajectories overlaid on the 2D data."""
    batch = _get_geodesic_batch(plot_dl)
    paths = embed_model.sample_geodesic_path(batch, num_points=num_points)
    # paths: (num_points, N, dim)
    X2 = project(np.asarray(X_data, dtype=float))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(X2[:, 0], X2[:, 1], c="lightgrey", s=3, alpha=0.5)
    for i in range(paths.shape[1]):
        p2 = project(np.asarray(paths[:, i], dtype=float))
        ax.plot(
            p2[:, 0],
            p2[:, 1],
            marker=".",
            linestyle="-",
            alpha=0.8,
            lw=1.5,
            color="tab:orange",
        )
    ax.set_title("Learned geodesic trajectories (2D)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _classifier_grid(classifier_model, X_data, project, dim: int, n: int = 400):
    """Evaluate the classifier on a 2D grid; return (xx, yy, probs). 2D only."""
    X2 = project(np.asarray(X_data, dtype=float))
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if dim > 2:
        # zero-pad to ambient dim for the classifier
        grid = np.pad(grid, ((0, 0), (0, dim - 2)), mode="constant")
    device = classifier_model.get_device()
    with torch.no_grad():
        logits = classifier_model.classify(torch.tensor(grid, device=device).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return xx, yy, probs.reshape(xx.shape[0], xx.shape[1], -1)


def plot_classifier_panels(
    classifier_model, adata, X_data, project, dim: int, out_dir: Path
) -> int:
    """GT-vs-pred scatter + confidence/entropy heatmaps (2D datasets)."""
    xx, yy, probs = _classifier_grid(classifier_model, X_data, project, dim)
    predictions = np.argmax(probs, axis=2)
    confidence = np.max(probs, axis=2)
    eps = 1e-10
    entropy = -np.sum(probs * np.log(probs + eps), axis=2)

    X2 = project(np.asarray(X_data, dtype=float))
    y = adata.obs["cell_type_one_hot"].values

    # Per-point predictions on the actual data (for the GT-vs-pred scatter).
    X_amb = np.asarray(X_data, dtype=float)
    if dim > 2:
        X_amb = np.pad(X_amb, ((0, 0), (0, dim - 2)), mode="constant")
    with torch.no_grad():
        device = classifier_model.get_device()
        pt_logits = classifier_model.classify(torch.tensor(X_amb, device=device).float())
    point_pred = torch.argmax(pt_logits, dim=1).cpu().numpy()

    # Panel 1: ground-truth vs predicted.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(X2[:, 0], X2[:, 1], c=y, s=10, alpha=0.6)
    axes[0].set_title("Ground truth (cell type)")
    axes[1].scatter(X2[:, 0], X2[:, 1], c=point_pred, s=10, alpha=0.6)
    axes[1].set_title("Classifier prediction")
    fig.savefig(out_dir / "classifier_pred.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Panel 2: confidence heatmap with data overlaid.
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("viridis", probs.shape[-1])
    ax.imshow(
        cmap(predictions),
        extent=(xx[0, 0], xx[-1, -1], yy[0, 0], yy[-1, 0]),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
    )
    ax.scatter(X2[:, 0], X2[:, 1], c="black", s=8, alpha=0.5)
    ax.set_title("Prediction (colored) + data")
    fig.savefig(out_dir / "classifier_pred_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Confidence.
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        confidence,
        extent=(xx[0, 0], xx[-1, -1], yy[0, 0], yy[-1, 0]),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap="magma",
    )
    ax.scatter(X2[:, 0], X2[:, 1], c="white", s=8, alpha=0.4)
    fig.colorbar(im, ax=ax, label="Confidence")
    ax.set_title("Classifier confidence")
    fig.savefig(out_dir / "classifier_confidence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Entropy.
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        entropy,
        extent=(xx[0, 0], xx[-1, -1], yy[0, 0], yy[-1, 0]),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap="inferno",
    )
    ax.scatter(X2[:, 0], X2[:, 1], c="white", s=8, alpha=0.4)
    fig.colorbar(im, ax=ax, label="Entropy")
    ax.set_title("Classifier entropy")
    fig.savefig(out_dir / "classifier_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 3


def _save_fig(fn, out_dir: Path, name: str) -> None:
    try:
        ok = fn(out_dir / name)
        if ok:
            print(f"  saved {name}")
    except Exception as exc:  # noqa: BLE001 — a single plot must not kill the run
        print(f"  FAILED {name}: {exc!r}")


def _run_real_dataset(
    dataset: str,
    metric: str,
    finsler: bool,
    seed: int,
    out_root: Path,
    n_traj: int,
    num_points: int,
    metric_epochs: int,
    flow_epochs: int,
    embed_epochs: int,
) -> Path:
    _set_seed(seed)
    config = load_config(overrides=[f"dataset={dataset}"])
    _apply_common_config(config, metric, finsler)
    config.metric_max_epochs = metric_epochs
    config.flow_max_epochs = flow_epochs
    config.embed_max_epochs = embed_epochs

    adata = process_data(
        pc_dim=config.pc_dim,
        t0_index=config.t0_index,
        t1_index=config.t1_index,
        data=config.data,
        use_paga=config.use_paga,
        paga_threshold=config.paga_threshold,
        tissue=config.tissue,
    )
    config.num_classes = adata.obs["cell_type"].nunique()

    timepoints = sorted(adata.obs["timepoint"].unique().tolist())
    tree = adata.uns["tree"]
    t0, t1 = timepoints[config.t0_index], timepoints[config.t1_index]
    adata = adata[(adata.obs["timepoint"] >= t0) & (adata.obs["timepoint"] <= t1)]
    adata_train = adata[adata.obs["timepoint"].isin([t0, t1])]

    singleton_dl = build_singleton_dataloader(config, adata_train)
    paired_dl = build_paired_dataloader(config, adata_train)

    classifier_model, metric_model, embed_model, flow_model = run_full_model(
        config=config,
        project=config.project,
        singleton_dataloader=singleton_dl,
        paired_dataloader=paired_dl,
        timepoints=timepoints,
        tree=tree,
        wandb_logger=None,
    )
    for m in (classifier_model, metric_model, embed_model, flow_model):
        remove_all_forward_hooks(m)

    _export_figures(
        config,
        classifier_model,
        embed_model,
        adata,
        out_dir=out_root,
        n_traj=n_traj,
        num_points=num_points,
        ambient_dim=config.pc_dim,
        classifier_heatmaps=False,
    )
    return out_root


def _run_synthetic(
    metric: str,
    finsler: bool,
    seed: int,
    out_root: Path,
    n_traj: int,
    num_points: int,
    size: int,
    dim: int,
    metric_epochs: int,
    flow_epochs: int,
    embed_epochs: int,
) -> Path:
    _set_seed(seed)
    adata = process_fake_data(size, dim, sample_path)
    timepoints = sorted(adata.obs["timepoint"].unique().tolist())
    tree = adata.uns["tree"]

    config = load_config(overrides=["dataset=zebrafish_cns"])
    _apply_common_config(config, metric, finsler)
    OmegaConf.set_struct(config, False)
    config.pc_dim = dim
    config.num_classes = adata.obs["cell_type"].nunique()
    config.t0_index = 0
    config.t1_index = 2
    config.score_batch_size = 1024
    config.finsler.lamb = 5.0 if dim == 2 else 1.0
    config.classifier.epsilon = 0.03 if dim == 2 else 0.01
    config.metric_max_epochs = metric_epochs
    config.flow_max_epochs = flow_epochs
    config.embed_max_epochs = embed_epochs

    t0, t1 = timepoints[config.t0_index], timepoints[config.t1_index]
    adata = adata[(adata.obs["timepoint"] >= t0) & (adata.obs["timepoint"] <= t1)]
    adata_train = adata[adata.obs["timepoint"].isin([t0, t1])]

    singleton_dl = build_singleton_dataloader(config, adata_train)
    paired_dl = build_paired_dataloader(config, adata_train)

    classifier_model, metric_model, embed_model, flow_model = run_full_model(
        config=config,
        project="synthetic",
        singleton_dataloader=singleton_dl,
        paired_dataloader=paired_dl,
        timepoints=timepoints,
        tree=tree,
        wandb_logger=None,
    )
    for m in (classifier_model, metric_model, embed_model, flow_model):
        remove_all_forward_hooks(m)

    _export_figures(
        config,
        classifier_model,
        embed_model,
        adata,
        out_dir=out_root,
        n_traj=n_traj,
        num_points=num_points,
        ambient_dim=dim,
        classifier_heatmaps=True,
    )
    return out_root


def _export_figures(
    config,
    classifier_model,
    embed_model,
    adata,
    out_dir: Path,
    n_traj: int,
    num_points: int,
    ambient_dim: int,
    classifier_heatmaps: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_data = np.asarray(adata.obsm["X_pca"], dtype=float)
    project = _make_2d_projector(X_data)
    plot_dl = _build_plot_paired_dataloader(config, adata, n_traj)

    print(f"Exporting figures to {out_dir} ...")
    _save_fig(
        lambda p: plot_river(classifier_model, embed_model, plot_dl, adata, num_points, p),
        out_dir,
        "river_plot.png",
    )
    _save_fig(lambda p: plot_lineage_tree(adata, p), out_dir, "lineage_tree.png")
    _save_fig(lambda p: plot_scatter_data(adata, project, p), out_dir, "scatter_data.png")
    _save_fig(
        lambda p: plot_geodesic_paths(embed_model, plot_dl, X_data, project, num_points, p),
        out_dir,
        "geodesic_paths.png",
    )

    if classifier_heatmaps and ambient_dim <= 50:
        try:
            n = plot_classifier_panels(
                classifier_model, adata, X_data, project, ambient_dim, out_dir
            )
            print(f"  saved classifier panels ({n} extra figures)")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED classifier panels: {exc!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one model and export the paper's figures."
    )
    parser.add_argument(
        "--dataset",
        default="zebrafish_cns",
        choices=AVAILABLE_DATASETS,
        help="Dataset to use (default: zebrafish_cns).",
    )
    parser.add_argument(
        "--metric",
        default="cfm",
        choices=["cfm", "mfm"],
        help="Base metric (default: cfm).",
    )
    parser.add_argument(
        "--finsler",
        dest="finsler",
        action="store_true",
        default=True,
        help="Use the Finsler variant (default: True).",
    )
    parser.add_argument(
        "--no-finsler",
        dest="finsler",
        action="store_false",
        help="Use the plain (non-Finsler) variant.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42).")
    parser.add_argument(
        "--n-traj", type=int, default=5, help="Number of river-plot trajectories (default: 5)."
    )
    parser.add_argument(
        "--num-geodesic-points",
        type=int,
        default=50,
        help="Geodesic path resolution (default: 50).",
    )
    parser.add_argument(
        "--metric-epochs", type=int, default=2, help="Metric phase epochs (default: 2)."
    )
    parser.add_argument(
        "--flow-epochs", type=int, default=2, help="Flow phase epochs (default: 2)."
    )
    parser.add_argument(
        "--embed-epochs",
        type=int,
        default=3000,
        help="Embed phase epochs (default: 3000; lower for fast sanity runs).",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output dir (default: outputs/<slug>)."
    )

    # Synthetic-only options.
    parser.add_argument(
        "--size", type=int, default=2500, help="Synthetic point count (default: 2500)."
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Synthetic feature dim (default: 2)."
    )
    args = parser.parse_args()

    slug = _method_slug(args.metric, args.finsler)
    if args.out is not None:
        out_dir = Path(args.out)
    elif args.dataset == "synthetic":
        out_dir = FINGEO_OUT_ROOT / "synthetic" / f"d{args.dim}_{slug}"
    else:
        out_dir = FINGEO_OUT_ROOT / args.dataset / slug

    if args.dataset == "synthetic":
        _run_synthetic(
            args.metric,
            args.finsler,
            args.seed,
            out_dir,
            args.n_traj,
            args.num_geodesic_points,
            args.size,
            args.dim,
            args.metric_epochs,
            args.flow_epochs,
            args.embed_epochs,
        )
    else:
        _run_real_dataset(
            args.dataset,
            args.metric,
            args.finsler,
            args.seed,
            out_dir,
            args.n_traj,
            args.num_geodesic_points,
            args.metric_epochs,
            args.flow_epochs,
            args.embed_epochs,
        )

    print(f"\nDone. Figures in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
