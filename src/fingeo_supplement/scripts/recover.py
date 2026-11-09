"""One-command recovery driver: process -> train -> save -> verify.

For a given tissue (``cns`` or ``pa``), this script:

1. Loads the matching Hydra config (``zebrafish_cns`` / ``zebrafish_arch``).
2. Enables the Finsler branch (``finsler.use = True``) with the paper's
   ``finsler.lamb = 0.5``.
3. Runs ``process_data`` (normalization, HVG, PCA, lineage tree) — this is
   where the raw ZScape Perturb atlas in ``data/`` is consumed.
4. Trains the classifier -> metric -> embed phases (``run_full_model``).
5. Saves the trained Finsler metric (``outputs/<tissue>/finsler_metric.pt``).
6. Verifies the Finsler norm (positive homogeneity / non-degeneracy).

It does NOT export the directed graphs: the full cell-level cost is
``n_cells²`` (e.g. ~4.3 billion entries for CNS) and would produce multi-GB
files. Use the dedicated CLI for on-demand graph export::

    uv run python utils/finsler_extract.py outputs/<tissue>/finsler_metric.pt \\
        --data data/processed/<slug>.h5ad --out outputs/<tissue>

Usage (run from ``fingeo_supplement/``)::

    uv run python scripts/recover.py --tissue cns
    uv run python scripts/recover.py --tissue pa
    uv run python scripts/recover.py --tissue both
    uv run python scripts/recover.py --tissue cns --skip-training   # load an existing artifact + verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fingeo_supplement.paths import FINGEO_OUT_ROOT


def _delete_per_tissue_cache(tissue: str) -> None:
    """Remove the per-tissue processed cache for ``tissue`` (if present).

    The cache lives at ``<FINGEO_DATA_DIR>/processed/<slug>.h5ad`` (see
    :func:`fingeo_supplement.paths.processed_cache_path`). Deleting it forces
    the next :func:`datasets.process.process_data` call to rebuild from the raw
    shared cache.
    """
    from fingeo_supplement.paths import processed_cache_path

    p = processed_cache_path(tissue)
    if p.exists():
        p.unlink()
        print(f"[{tissue}] deleted per-tissue cache {p}")


from fingeo_supplement.utils.hydra import load_config
from fingeo_supplement.scripts.run_model import (
    build_singleton_dataloader,
    build_paired_dataloader,
    run_full_model,
    remove_all_forward_hooks,
    save_finsler_artifacts,
)
from fingeo_supplement.datasets.process import process_data
from fingeo_supplement.utils.finsler_extract import (
    load_finsler_metric,
    export_directed_graphs,
    verify_finsler_norm,
)

# Maps a tissue tag to the Hydra dataset config + process_data kwargs.
TISSUES = {
    "cns": {
        "dataset": "zebrafish_cns",
        "data": "zebrafish",
        "tissue": "Central Nervous System",
        "use_paga": False,
        "paga_threshold": 0.0,
        "t0_index": 1,
        "t1_index": 4,
    },
    "pa": {
        "dataset": "zebrafish_arch",
        "data": "zebrafish",
        "tissue": "Pharyngeal Arch",
        "use_paga": True,
        "paga_threshold": 0.2,
        "t0_index": 1,
        "t1_index": 4,
    },
}


def _resolve_tissue(tissue: str) -> str:
    if tissue in TISSUES:
        return tissue
    raise SystemExit(f"Unknown tissue '{tissue}'. Choose from: {list(TISSUES)} or 'both'.")


def run_tissue(
    tissue: str,
    out_root: str | None = None,
    do_training: bool = True,
    use_wandb: bool = False,
    device: str = "auto",
    force: bool = False,
) -> None:
    """Process, (optionally) train, save, extract, and export one tissue.

    Parameters
    ----------
    device
        ``"auto"`` uses a GPU when ``torch.cuda.is_available()`` (default),
        otherwise CPU. Pass ``"gpu"`` to require CUDA (raises if none),
        or ``"cpu"`` to force CPU.
    force
        If True, delete the per-tissue processed cache
        (``data/processed/<slug>.h5ad``) before calling
        :func:`datasets.process.process_data`, forcing a rebuild from the
        raw shared cache. Use after changing ``pc_dim``, the lineage tree,
        or any preprocessing parameter.
    """
    if out_root is None:
        out_root = FINGEO_OUT_ROOT
    spec = TISSUES[_resolve_tissue(tissue)]
    if force:
        _delete_per_tissue_cache(spec["tissue"])
    cuda_available = torch.cuda.is_available()
    if device == "gpu" and not cuda_available:
        raise SystemExit("--gpu requested but no CUDA device is available.")
    force_cpu = (device == "cpu") or (device == "auto" and not cuda_available)
    print(
        f"\n{'=' * 70}\nTissue: {tissue}  ->  config '{spec['dataset']}'  "
        f"| device: {'CPU' if force_cpu else f'GPU ({torch.cuda.get_device_name(0)})'}\n{'=' * 70}"
    )

    # ---- config ----------------------------------------------------------
    config = load_config(overrides=[f"dataset={spec['dataset']}"])
    config.metric = "cfm"
    config.finsler.use = True
    config.finsler.lamb = 0.5
    config.use_wandb = use_wandb
    # Use the GPU by default (faster); fall back to CPU when none is present.
    config.force_cpu = force_cpu

    # ---- data ------------------------------------------------------------
    adata = process_data(
        pc_dim=config.pc_dim,
        t0_index=spec["t0_index"],
        t1_index=spec["t1_index"],
        data=spec["data"],
        use_paga=spec["use_paga"],
        paga_threshold=spec["paga_threshold"],
        tissue=spec["tissue"],
    )
    config.num_classes = adata.obs["cell_type"].nunique()
    timepoints = sorted(adata.obs["timepoint"].unique().tolist())
    t0 = timepoints[spec["t0_index"]]
    t1 = timepoints[spec["t1_index"]]
    adata_range = adata[(adata.obs["timepoint"] >= t0) & (adata.obs["timepoint"] <= t1)]
    adata_train = adata_range[adata_range.obs["timepoint"].isin([t0, t1])]
    print(
        f"[{tissue}] adata {adata.shape}, {config.num_classes} cell types, "
        f"timepoints {timepoints}"
    )

    # ---- train (optional) ------------------------------------------------
    if do_training:
        singleton_dl = build_singleton_dataloader(config, adata_train)
        paired_dl = build_paired_dataloader(config, adata_train)
        classifier_model, metric_model, embed_model, flow_model = run_full_model(
            config=config,
            project=config.project,
            singleton_dataloader=singleton_dl,
            paired_dataloader=paired_dl,
            timepoints=timepoints,
            tree=adata.uns["tree"],
            wandb_logger=None,
        )
        remove_all_forward_hooks(classifier_model)
        remove_all_forward_hooks(metric_model)
        remove_all_forward_hooks(embed_model)
        remove_all_forward_hooks(flow_model)

        # ---- save the trained metric ------------------------------------
        artifact = save_finsler_artifacts(
            config, adata, classifier_model, embed_model, out_root=out_root
        )
    else:
        artifact = Path(out_root) / tissue / "finsler_metric.pt"
        if not artifact.exists():
            raise SystemExit(
                f"--skip-training given but no artifact at {artifact}. "
                "Train first or drop in a finsler_metric.pt."
            )
        print(f"[{tissue}] skipping training, using {artifact}")

    # ---- extract + export -----------------------------------------------
    emb = load_finsler_metric(artifact)
    X = adata_range.obsm["X_pca"]
    cell_type = adata_range.obs["cell_type"].tolist()
    print(f"[{tissue}] X_pca {X.shape}")

    # NOTE: the full cell-level directed graph is ~4.3 billion entries for CNS
    # (65 554² cells) and produces multi-GB files. It is NOT exported here.
    # Use the dedicated CLI for on-demand graph export instead:
    #   uv run python utils/finsler_extract.py outputs/<tissue>/finsler_metric.pt \
    #       --data data/processed/<slug>.h5ad --out outputs/<tissue>
    # result = export_directed_graphs(emb, X, cell_type, out_root, tissue=tissue)
    # for k, v in result.items():
    #     if v:
    #         print(f"[{tissue}] exported {k} graph:")
    #         for kind, p in v.items():
    #             print(f"    {kind}: {p}")

    # ---- verify the Finsler norm ----------------------------------------
    if emb.classifier_net is not None and emb.tree is not None:
        check = verify_finsler_norm(emb, X)
        print(f"[{tissue}] Finsler-norm check: {check}")
    else:
        print(f"[{tissue}] (Finsler-norm check skipped: no classifier/tree in bundle)")
    print(f"[{tissue}] DONE")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tissue", default="both", choices=["cns", "pa", "both"])
    ap.add_argument(
        "--out", default=str(FINGEO_OUT_ROOT), help="output root (default: <repo>/outputs)"
    )
    ap.add_argument(
        "--skip-training",
        action="store_true",
        help="do not train; load an existing finsler_metric.pt and extract",
    )
    ap.add_argument("--use-wandb", action="store_true", help="log to wandb (off by default)")
    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "delete the per-tissue processed cache "
            "(data/processed/<tissue>.h5ad) before processing, "
            "forcing a rebuild from the raw shared cache"
        ),
    )
    device_group = ap.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu",
        dest="device",
        action="store_const",
        const="gpu",
        help="require a CUDA GPU (default: auto, uses GPU if available)",
    )
    device_group.add_argument(
        "--cpu",
        dest="device",
        action="store_const",
        const="cpu",
        help="force CPU (use when no GPU is available / for debugging)",
    )
    args = ap.parse_args()
    if getattr(args, "device", None) is None:
        args.device = "auto"

    tissues = ["cns", "pa"] if args.tissue == "both" else [args.tissue]
    for t in tissues:
        run_tissue(
            t,
            out_root=args.out,
            do_training=not args.skip_training,
            use_wandb=args.use_wandb,
            device=args.device,
            force=args.force,
        )


if __name__ == "__main__":
    main()
