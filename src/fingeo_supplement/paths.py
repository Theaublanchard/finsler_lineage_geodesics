"""Canonical repo-anchored paths for data, outputs, and results.

Every directory the codebase reads or writes is anchored to the repo root
(i.e. the parent of ``src/``), not to the current working directory. This
keeps the code correct regardless of where it is invoked from (notebooks,
CLI scripts, ``uv run`` from a sub-folder, etc.).

Each of the three top-level directories can be overridden via an
environment variable, which is useful when the user wants to redirect
downloads / artifacts to a different location:

* ``FINGEO_DATA_DIR``    -- default ``<repo_root>/data``
* ``FINGEO_OUT_ROOT``    -- default ``<repo_root>/outputs``
* ``FINGEO_RESULTS_DIR`` -- default ``<repo_root>/results``

The module is intentionally dependency-free and import-safe: it only touches
``os.environ`` and ``pathlib.Path``, and performs no I/O at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------
# This file lives at <repo_root>/src/fingeo_supplement/paths.py, so the repo
# root is two levels up from this file's parent.
FINGEO_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _env_or_default(env_var: str, default: Path) -> Path:
    """Return the value of ``env_var`` if set and non-empty, else ``default``."""
    value = os.environ.get(env_var, "").strip()
    if value:
        return Path(value).expanduser()
    return default


# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
FINGEO_DATA_DIR: Path = _env_or_default("FINGEO_DATA_DIR", FINGEO_ROOT / "data")
FINGEO_OUT_ROOT: Path = _env_or_default("FINGEO_OUT_ROOT", FINGEO_ROOT / "outputs")
FINGEO_RESULTS_DIR: Path = _env_or_default("FINGEO_RESULTS_DIR", FINGEO_ROOT / "results")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def data_file(name: str) -> Path:
    """Absolute path to a file inside :data:`FINGEO_DATA_DIR`."""
    return FINGEO_DATA_DIR / name


def reports_root() -> Path:
    """Directory that holds per-run ``report.json`` files."""
    return FINGEO_OUT_ROOT / "reports"


def _slug(tissue: str) -> str:
    """Filesystem-safe slug for a tissue name.

    Lower-cases and collapses non-alphanumeric runs to ``_``
    (e.g. ``"Central Nervous System"`` -> ``"central_nervous_system"``).
    """
    slug = tissue.lower()
    slug = "".join(ch if ch.isalnum() else "_" for ch in slug).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def processed_cache_path(tissue: str) -> Path:
    """Path of the per-tissue *processed* cache.

    ``<FINGEO_DATA_DIR>/processed/<slug>.h5ad`` where ``<slug>`` is the tissue
    name lower-cased with non-alphanumeric runs collapsed to ``_`` (e.g.
    ``"Central Nervous System"`` -> ``"central_nervous_system"``).

    Unlike the raw shared cache (``<FINGEO_DATA_DIR>/zebra_preprocessed.h5ad``),
    this file contains the final object consumed by training/eval: the tissue
    subset, ``obs['cell_type']`` / ``obs['cell_type_one_hot']``,
    ``obsm['X_pca']``, and the lineage tree (``uns['tree'/'stoi'/'itos']``).
    It can therefore be read directly with ``anndata.read_h5ad`` without
    re-running ``process_data``.
    """
    return FINGEO_DATA_DIR / "processed" / f"{_slug(tissue)}.h5ad"


__all__ = [
    "FINGEO_ROOT",
    "FINGEO_DATA_DIR",
    "FINGEO_OUT_ROOT",
    "FINGEO_RESULTS_DIR",
    "data_file",
    "reports_root",
    "processed_cache_path",
]
