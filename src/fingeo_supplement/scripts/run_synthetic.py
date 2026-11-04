"""Reproduce the paper's synthetic W1 table (``tab::syn``) from the CLI.

This is the script-based counterpart of ``holdout_toy.ipynb``. It generates the
branching "path" synthetic dataset in memory (no raw files, no WandB), trains the
classifier → metric → embed pipeline, and reports the held-out-timepoint
Wasserstein-1 distance for a grid of ``dim`` x ``finsler`` settings, each over
``n_seeds`` independent runs.

Table cell = ``mean(W1 over the held-out intermediate timepoints)``, then
``mean +/- std`` over the seeds — matching the paper.

Usage
-----
    # Both dims, both methods, 10 seeds (default):
    uv run python scripts/run_synthetic.py

    # A single setting:
    uv run python scripts/run_synthetic.py --dim 2 --finsler --n_seeds 10

    # Write a JSON summary:
    uv run python scripts/run_synthetic.py --out results/synthetic.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Keep the synthetic pipeline CPU-friendly and free of external logging.
# Import the project modules (top-level ``models.`` / ``utils.`` imports mean we
# must run with the project root on sys.path — running via ``uv run`` from the
# project root satisfies this).
from fingeo_supplement.simulate.simulate import process_fake_data, sample_path  # noqa: E402
from fingeo_supplement.scripts.run_model import (  # noqa: E402
    build_paired_dataloader,
    build_singleton_dataloader,
    remove_all_forward_hooks,
    run_full_model,
)
from fingeo_supplement.eval.eval import predict  # noqa: E402
from fingeo_supplement.utils.hydra import load_config  # noqa: E402
from fingeo_supplement.utils.report import write_run_report  # noqa: E402

# Synthetic sample functions available in ``simulate.simulate`` (the notebook
# uses ``sample_path`` for the paper's tab::syn).
AVAILABLE_SAMPLE_FNS = [
    "sample_s",
    "sample_s_imbalanced",
    "sample_omega",
    "sample_disk",
    "sample_balance",
    "sample_growth",
    "sample_points",
    "sample_path",
    "sample_diamond",
    "sample_moons",
]
# Import lazily by name so the default (sample_path) is what we advertise.
import fingeo_supplement.simulate.simulate as _sim  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_synthetic_config(config, adata, dim: int, finsler: bool) -> None:
    """Apply the paper's synthetic settings onto a fresh composed config.

    Mirrors the SETTINGS cells of ``holdout_toy.ipynb``.
    """
    OmegaConf.set_struct(config, False)
    config.pc_dim = dim
    config.num_classes = adata.obs["cell_type"].nunique()

    # Training on endpoints t=0 and t=2, interpolating the middle timepoint.
    config.t0_index = 0
    config.t1_index = 2
    config.score_batch_size = 1024

    config.metric = "cfm"
    config.finsler.use = bool(finsler)
    config.finsler.lamb = 5.0 if dim == 2 else 1.0
    config.dummy_kl_weight = 0.05

    config.classifier.num_layers = 2
    config.classifier.hidden_dim = 128
    config.classifier.epsilon = 0.03 if dim == 2 else 0.01

    config.classifier_max_epochs = 1000
    config.metric_max_epochs = 2
    config.flow_max_epochs = 2

    # Standalone script: no WandB, no forced-CPU (matches the notebook).
    config.use_wandb = False
    config.no_learning = False


def _run_one(dim: int, finsler: bool, seed: int, size: int, sample_fn) -> float:
    """Train once at (dim, finsler, seed) and return the mean held-out W1."""
    _set_seed(seed)

    adata = process_fake_data(size, dim, sample_fn)
    timepoints = sorted(adata.obs["timepoint"].unique().tolist())
    tree = adata.uns["tree"]

    config = load_config(overrides=["dataset=zebrafish_cns"])
    _build_synthetic_config(config, adata, dim, finsler)

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
    remove_all_forward_hooks(classifier_model)
    remove_all_forward_hooks(metric_model)
    remove_all_forward_hooks(embed_model)
    remove_all_forward_hooks(flow_model)

    w1_scores = []
    for index in range(config.t0_index + 1, config.t1_index):
        t = timepoints[index]
        w1 = predict(
            embed_model,
            adata,
            t0,
            t,
            t1,
            num_traj=6000,
            library="pot",
            icfm=config.force_icfm,
        )
        w1_scores.append(w1)

    w1_avg = float(torch.mean(torch.tensor(w1_scores)).item())
    # Self-report to disk (this path bypasses train(), so hook it here explicitly).
    try:
        write_run_report(
            config=config,
            w1_avg=w1_avg,
            w1_val_avg=None,
            w1_per_timepoint=[float(x) for x in w1_scores],
            run_name=f"synthetic_dim{dim}_{_method_label(finsler)}_seed{seed}",
            run_id=None,
            # This experiment borrows a real dataset config as a Hydra base for
            # its defaults, so label the report as synthetic (and record the dim).
            extra={"dataset": "synthetic", "dim": dim},
        )
    except Exception as _e:  # never let a report failure kill a finished run
        print(f"[report] WARNING: failed to write run report: {_e!r}")

    return w1_avg


def _method_label(finsler: bool) -> str:
    return "CFM+Finsler" if finsler else "CFM"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's synthetic W1 table (tab::syn)."
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 50],
        help="Feature dimensions to evaluate (default: 2 50).",
    )
    parser.add_argument(
        "--finsler",
        dest="finsler",
        action="store_true",
        default=None,
        help="Run only the CFM+Finsler variant.",
    )
    parser.add_argument(
        "--no-finsler",
        dest="finsler",
        action="store_false",
        help="Run only the plain CFM variant.",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=10,
        help="Number of independent seeds (default: 10).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list (overrides --n_seeds).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2500,
        help="Number of synthetic points (default: 2500).",
    )
    parser.add_argument(
        "--sample-fn",
        default="sample_path",
        choices=AVAILABLE_SAMPLE_FNS,
        help="simulate.simulate generator (default: sample_path).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of the results.",
    )
    args = parser.parse_args()

    sample_fn = getattr(_sim, args.sample_fn)

    finsler_variants = (
        [True] if args.finsler is True else [False] if args.finsler is False else [True, False]
    )
    seeds = args.seeds if args.seeds is not None else list(range(args.n_seeds))

    dims = list(dict.fromkeys(args.dims))  # de-dup, keep order
    rows = []  # list of dicts: dim, method, seed, w1

    for dim in dims:
        for finsler in finsler_variants:
            label = _method_label(finsler)
            for seed in seeds:
                print(f"[run] dim={dim}  method={label}  seed={seed} ...", flush=True)
                try:
                    w1 = _run_one(dim, finsler, seed, args.size, sample_fn)
                except Exception as exc:  # noqa: BLE001 — keep the grid going
                    print(f"  ERROR: {exc!r}", file=sys.stderr)
                    continue
                rows.append({"dim": dim, "method": label, "seed": seed, "w1": w1})

    # Aggregate: mean +/- std over seeds, per (dim, method).
    from collections import defaultdict

    agg = defaultdict(list)
    for r in rows:
        agg[(r["dim"], r["method"])].append(r["w1"])

    print("\n" + "=" * 60)
    header = "Method".ljust(16) + "".join(f"Synthetic(n={d})".rjust(22) for d in dims)
    print(header)
    print("-" * len(header))
    methods = [m for m in ("CFM", "CFM+Finsler")]
    for method in methods:
        line = method.ljust(16)
        for dim in dims:
            vals = agg.get((dim, method), [])
            if vals:
                mean = float(np.mean(vals))
                std = float(np.std(vals))
                line += f"  {mean:.2f} +/- {std:.2f}".rjust(22)
            else:
                line += "  n/a".rjust(22)
        print(line)
    print("=" * 60)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "settings": {
                "dims": dims,
                "finsler": finsler_variants,
                "seeds": seeds,
                "size": args.size,
                "sample_fn": args.sample_fn,
            },
            "runs": rows,
            "aggregated": {
                f"dim={k[0]} {k[1]}": {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "n": len(v),
                }
                for k, v in sorted(agg.items())
            },
        }
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nWrote JSON summary to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
