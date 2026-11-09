"""Recover W1 results from the self-produced run reports.

Reads the per-run ``report.json`` files written by ``utils.report.write_run_report``
(under ``outputs/reports/<slug>/``), groups them by ``(dataset, method,
classifier_epsilon)``, and prints a legible table with the mean +/- std of
``w1_avg`` over the seeds in each group.

This is the "one command" counterpart to the manual WandB-scraping that used to
be required: no dashboard, no raw-run parsing — just the on-disk reports.

Usage
-----
    # Print the aggregated table (mean +/- std of w1_avg per group):
    uv run python scripts/recover_results.py

    # Filter to one dataset / one method:
    uv run python scripts/recover_results.py --dataset zebrafish_arch
    uv run python scripts/recover_results.py --method "CFM+Finsler"

    # Also dump a flat CSV with one row per seed (the raw, un-pooled values):
    uv run python scripts/recover_results.py --out results/all_seeds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from fingeo_supplement.paths import reports_root

REPORT_ROOT = reports_root()
GROUP_KEYS = ("dataset", "method", "classifier_epsilon")
CSV_COLUMNS = [
    "dataset",
    "method",
    "classifier_epsilon",
    "seed",
    "w1_avg",
    "w1_val_avg",
    "run_name",
]


def load_reports(root: Path) -> list[dict]:
    """Load every ``report.json`` under ``root``, deduping by ``run_name``."""
    reports: dict[str, dict] = {}
    for path in sorted(root.glob("*/report.json")):
        try:
            with open(path) as fh:
                rep = json.load(fh)
        except Exception as exc:
            print(f"[recover] WARNING: could not read {path}: {exc!r}")
            continue
        # Prefer the reported run_name as the dedup key; fall back to the path.
        key = rep.get("run_name") or str(path.parent)
        if rep.get("w1_avg") is None:
            continue
        reports[key] = rep
    return list(reports.values())


def group(reports: list[dict]) -> dict[tuple, list[dict]]:
    """Group reports by (dataset, method, classifier_epsilon)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rep in reports:
        key = (rep.get("dataset"), rep.get("method"), rep.get("classifier_epsilon"))
        groups[key].append(rep)
    return groups


def _fmt_eps(value) -> str:
    if value is None:
        return "na"
    return str(value)


def print_table(groups: dict[tuple, list[dict]]) -> None:
    """Option A: one line per group, mean +/- std of w1_avg over its seeds."""
    if not groups:
        print("No run reports found (nothing under outputs/reports/).")
        return

    header = (
        f"{'dataset':<18s} {'method':<14s} {'eps':>6s} "
        f"{'n':>4s}   {'w1_avg (mean +/- std)':>24s}"
    )
    print(header)
    print("-" * len(header))

    def sort_key(k):
        eps = k[2]
        eps_key = (1, str(eps)) if eps is None else (0, str(eps))
        return (str(k[0]), str(k[1]), eps_key)

    for key in sorted(groups.keys(), key=sort_key):
        reps = groups[key]
        values = [r["w1_avg"] for r in reps if r.get("w1_avg") is not None]
        if not values:
            continue
        n = len(values)
        if n >= 2:
            cell = f"{statistics.fmean(values):.4f} +/- {statistics.pstdev(values):.4f}"
        else:
            cell = f"{values[0]:.4f}"
        print(
            f"{str(key[0] or 'na'):<18s} {str(key[1] or 'na'):<14s} "
            f"{_fmt_eps(key[2]):>6s} {n:>4d}   {cell:>24s}"
        )


def write_csv(reports: list[dict], out_path: Path) -> None:
    """Write a flat CSV with one row per seed (raw, un-pooled values)."""
    rows = []
    for rep in reports:
        rows.append(
            {
                "dataset": rep.get("dataset"),
                "method": rep.get("method"),
                "classifier_epsilon": rep.get("classifier_epsilon"),
                "seed": rep.get("seed"),
                "w1_avg": rep.get("w1_avg"),
                "w1_val_avg": rep.get("w1_val_avg"),
                "run_name": rep.get("run_name"),
            }
        )
    rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            str(r["method"]),
            str(r["classifier_epsilon"]),
            str(r["seed"]),
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[recover] wrote {len(rows)} seed rows -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover W1 results from outputs/reports/ self-produced reports."
    )
    parser.add_argument(
        "--reports-dir",
        default=str(REPORT_ROOT),
        help=f"Root directory containing per-run report folders (default: {REPORT_ROOT}).",
    )
    parser.add_argument("--dataset", default=None, help="Only show reports for this dataset.")
    parser.add_argument(
        "--method", default=None, help="Only show reports with this method label."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to also write a flat CSV (one row per seed) to.",
    )
    args = parser.parse_args()

    root = Path(args.reports_dir)
    if not root.exists():
        print(f"No reports directory at {root}. Run a sweep or the synthetic script first.")
        return 0

    reports = load_reports(root)
    if not reports:
        print(f"No report.json files found under {root}.")
        return 0

    # Optional filters.
    if args.dataset:
        reports = [r for r in reports if r.get("dataset") == args.dataset]
    if args.method:
        reports = [r for r in reports if r.get("method") == args.method]

    if not reports:
        print("No reports matched the given filters.")
        return 0

    print_table(group(reports))

    if args.out:
        write_csv(reports, Path(args.out))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
