"""Self-reporting helper for W1-producing runs.

Every run that produces a held-out ``W1`` score calls :func:`write_run_report`
right after computing it. That writes a single ``report.json`` under
``outputs/reports/<slug>/`` capturing the identifying fields (dataset, method,
seed, classifier epsilon, ...) plus the W1 numbers and the full resolved config.

``scripts/recover_results.py`` later aggregates these per-run reports into a
legible table. Nothing here depends on WandB — the on-disk reports are the
source of truth.

All config lookups are defensive (``getattr``) so that a missing key can never
break a finished run: a report is a convenience, not a requirement.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Import the resolved-config serializer lazily so this module stays importable
# even in environments that don't have omegaconf for some reason.
try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - omegaconf is a hard project dep
    OmegaConf = None

from fingeo_supplement.paths import reports_root


def method_label(metric: str | None, finsler_use: bool | None) -> str:
    """Human-readable method name from the config's ``metric`` + ``finsler.use``.

    ``"cfm" -> "CFM"``, ``"mfm" -> "MFM"``, and appends ``+Finsler`` when the
    Finsler lineage penalty is enabled.
    """
    base = (metric or "unknown").strip().upper()
    if finsler_use:
        base += "+Finsler"
    return base


def _safe(value: str | None) -> str:
    """Filesystem-safe slug fragment: non-alphanumerics -> ``_``, trimmed."""
    if value is None:
        return "na"
    return re.sub(r"[^0-9A-Za-z]+", "_", str(value)).strip("_") or "na"


def run_slug(dataset, method, seed, eps) -> str:
    """Build a unique, filesystem-safe directory name for one run."""
    parts = [_safe(dataset), _safe(method), _safe(seed), _safe(eps)]
    return "_".join(parts)


def _git_commit() -> str | None:
    """Best-effort short git commit hash; ``None`` if unavailable. Never raises."""
    env = os.environ.get("GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _resolved_config(config) -> object:
    """Serialize the config to a JSON-friendly container, best effort."""
    if OmegaConf is not None:
        try:
            return OmegaConf.to_container(config, resolve=True)
        except Exception:
            try:
                return OmegaConf.to_container(config, resolve=False)
            except Exception:
                pass
    return config


def write_run_report(
    *,
    config,
    w1_avg,
    w1_val_avg,
    w1_per_timepoint=None,
    run_name: str | None = None,
    run_id: str | None = None,
    extra: dict | None = None,
    out_root: str | Path | None = None,
) -> Path:
    """Write a single per-run ``report.json`` and return its path.

    Parameters
    ----------
    config
        The resolved Hydra config (or any mapping) for this run.
    w1_avg
        Mean held-out (intermediate-timepoint) W1 for the run.
    w1_val_avg
        W1 at the held-out validation timepoint (may be ``None``).
    w1_per_timepoint
        Optional list of the per-timepoint W1 values that average to ``w1_avg``.
    run_name / run_id
        Identifiers for the run (e.g. from ``wandb.init``).
    extra
        Optional dict merged into the report (e.g. ``{"dim": 2}``).
    out_root
        Directory under which ``<slug>/report.json`` is written.

    Returns
    -------
    Path
        The path to the written ``report.json``.
    """
    finsler = getattr(config, "finsler", None)
    classifier = getattr(config, "classifier", None)
    finsler_use = bool(getattr(finsler, "use", False)) if finsler is not None else False
    metric = getattr(config, "metric", None)
    dataset = getattr(config, "dataset", None)
    seed = getattr(config, "seed", None)
    eps = getattr(classifier, "epsilon", None) if classifier is not None else None

    report = {
        "run_name": run_name,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "tissue": getattr(config, "tissue", None),
        "method": method_label(metric, finsler_use),
        "metric": metric,
        "finsler_use": finsler_use,
        "seed": seed,
        "classifier_epsilon": eps,
        "finsler_lamb": getattr(finsler, "lamb", None) if finsler is not None else None,
        "t0_index": getattr(config, "t0_index", None),
        "t1_index": getattr(config, "t1_index", None),
        "w1_per_timepoint": [float(x) for x in (w1_per_timepoint or [])],
        "w1_avg": float(w1_avg),
        "w1_val_avg": (float(w1_val_avg) if w1_val_avg is not None else None),
        "git_commit": _git_commit(),
        "config": _resolved_config(config),
    }
    if extra:
        report.update(extra)

    slug = run_name or run_slug(dataset, report["method"], seed, eps)
    slug = _safe(slug)
    out_dir = (Path(out_root) if out_root is not None else reports_root()) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"

    # ``default=str`` guards against any stray non-serializable value in the
    # config container so the report always lands even if something odd sneaks in.
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"[report] wrote {out_path}  (w1_avg={report['w1_avg']:.4f})")
    return out_path
