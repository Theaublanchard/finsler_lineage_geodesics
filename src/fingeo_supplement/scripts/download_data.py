"""Data fetch + readiness check for the ZScape Perturb zebrafish atlas.

Downloads the three raw files ``datasets/process.py`` expects into ``data/``
(the URLs are the ZScape "Genetic Perturbation Atlas" files, distributed as
gzip and transparently decompressed), then verifies that the files are present
and that the cell/gene metadata carry the columns the pipeline depends on.

It **attempts** the download automatically; if the network is unavailable or a
URL fails it reports the failure and prints the manual download steps (see
``DATA_DOWNLOAD.md``). It does not touch files that are already present unless
``--force`` is given. Uses only the standard library (``urllib``).

Run from ``fingeo_supplement/``::

    uv run python scripts/download_data.py                 # download if needed, then verify
    uv run python scripts/download_data.py --no-download   # verify only (no network)
    uv run python scripts/download_data.py --force         # re-download even if present
    uv run python scripts/download_data.py --data-dir data
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_FILES = [
    "zscape_perturb_full_raw_counts.mtx",
    "zscape_perturb_full_cell_metadata.csv",
    "zscape_perturb_full_gene_metadata.csv",
]

CITATION = (
    "Lauren M Saunders, Sanjay R Srivatsan, Madeleine Duran, Michael W Dorrity, "
    "Brent Ewing, Tor H Linbo, Jay Shendure, David W Raible, Cecilia B Moens, "
    "David Kimelman, et al. Embryo-scale reverse genetics at single-cell "
    "resolution. Nature, 623(7988):782-791, 2023. (ZScape Perturb zebrafish atlas.)"
)

# Columns the pipeline requires.
# NOTE: the ZScape Perturb cell metadata timepoint column is named `timepoint`
# (not `day`); `process_data` reads `adata.obs['timepoint']`.
CELL_REQUIRED = ["gene_target", "tissue", "cell_type_broad", "timepoint"]
GENE_REQUIRED = ["gene_short_name"]
CELL_VALUE_CHECKS = {
    "gene_target": "ctrl-inj",
    "tissue": ["Central Nervous System", "Pharyngeal Arch"],
}

# Direct download URLs (ZScape "Genetic Perturbation Atlas" -> individual parts),
# distributed as gzip. Source: https://cole-trapnell-lab.github.io/zscape/downloads/
DOWNLOAD_URLS = {
    "zscape_perturb_full_raw_counts.mtx": "http://trapnell-lab-s3-zscape.s3-website-us-west-2.amazonaws.com/zscape_perturb_full_raw_counts.mtx.gz",
    "zscape_perturb_full_cell_metadata.csv": "http://trapnell-lab-s3-zscape.s3-website-us-west-2.amazonaws.com/zscape_perturb_full_cell_metadata.csv.gz",
    "zscape_perturb_full_gene_metadata.csv": "http://trapnell-lab-s3-zscape.s3-website-us-west-2.amazonaws.com/zscape_perturb_full_gene_metadata.csv.gz",
}


def _download_file(url: str, target: Path, timeout: float) -> int:
    """Download ``url`` to ``target``, transparently decompressing gzip.

    Streams to a ``<target>.part`` temp file first, then decompresses (or
    moves, if the payload is not gzip) to ``target``. On any failure the
    partial artifacts are removed so a later run retries cleanly. Returns the
    final decompressed size in bytes.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (fingeo-supplement data fetch)"}
    )
    step = 25 * 1024 * 1024
    downloaded = 0
    last_report = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            bucket = downloaded // step
            if bucket > last_report:
                last_report = bucket
                print(
                    f"[download_data]     {target.name}: {downloaded / 1048576:.0f} MiB",
                    flush=True,
                )
    if downloaded == 0:
        tmp.unlink(missing_ok=True)
        raise IOError(f"0 bytes downloaded from {url}")
    try:
        with tmp.open("rb") as fh:
            magic = fh.read(2)
        if magic == b"\x1f\x8b":  # gzip -> decompress to target
            with gzip.open(tmp, "rb") as fin, target.open("wb") as fout:
                shutil.copyfileobj(fin, fout, length=1024 * 1024)
        else:  # not gzip -> just rename into place
            shutil.move(str(tmp), str(target))
    except Exception:
        for p in (tmp, target):
            p.unlink(missing_ok=True)
        raise
    if tmp.exists():
        tmp.unlink()
    return target.stat().st_size


def download_missing(data_dir: Path, timeout: float) -> list[str]:
    """Attempt to auto-download any missing DATA_FILES. Returns those still missing."""
    missing = [
        name
        for name in DATA_FILES
        if not (data_dir / name).is_file() or (data_dir / name).stat().st_size == 0
    ]
    if not missing:
        return []
    print(f"[download_data] {len(missing)} file(s) missing - attempting automatic download...")
    for name in missing:
        url = DOWNLOAD_URLS.get(name)
        if not url:
            print(f"[download_data]   skip {name}: no known URL")
            continue
        print(f"[download_data]   -> {name}")
        print(f"[download_data]      {url}")
        try:
            size = _download_file(url, data_dir / name, timeout=timeout)
            print(f"[download_data]   OK   {name} ({size / 1048576:.1f} MiB)")
        except Exception as exc:
            print(f"[download_data]   FAIL {name}: {exc!r}")
    return [
        name
        for name in DATA_FILES
        if not (data_dir / name).is_file() or (data_dir / name).stat().st_size == 0
    ]


def _csv_columns(path: Path) -> list[str]:
    """Read just the header row of a CSV and return its column names."""
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
    return [c.strip() for c in header] if header else []


def _csv_values(path: Path, column: str, limit: int = 200000) -> list[str]:
    """Collect up to ``limit`` unique values of ``column`` from a CSV.

    Bounds work so we do not materialize the entire (large) metadata file into
    memory; for the value checks we only need to know whether a specific value
    is *present*, and the atlas is not that large in rows for these columns.
    """
    values: set[str] = set()
    try:
        with path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                val = row.get(column)
                if val is not None:
                    values.add(val.strip())
    except (csv.Error, UnicodeDecodeError) as exc:  # pragma: no cover - depends on file
        print(f"[download_data]   ! could not parse {path.name}: {exc}")
    return sorted(values)


def manual_steps() -> str:
    return "\n".join(
        [
            "Automatic download was unavailable or failed - fetch the files manually.",
            f"  Source: {CITATION}",
            "  Dataset: ZScape 'Genetic Perturbation Atlas' -> 'For individual parts of the dataset'",
            "  Page:    https://cole-trapnell-lab.github.io/zscape/downloads/",
            "  Direct URLs (gzip):",
            *(f"       {name}: {DOWNLOAD_URLS[name]}" for name in DATA_FILES),
            "  Manual steps:",
            "       mkdir -p data",
            "       wget -P data <the three .gz urls above>",
            "       gunzip data/*.gz   # final names must have NO .gz (see DATA_DOWNLOAD.md)",
            "  Or retry:  uv run python scripts/download_data.py   (re-attempts the download)",
            "  Full instructions: see DATA_DOWNLOAD.md",
        ]
    )


def check(data_dir: Path) -> int:
    print(f"[download_data] checking data dir: {data_dir.resolve()}")
    ok = True

    # 1) File presence.
    for name in DATA_FILES:
        p = data_dir / name
        exists = p.is_file()
        ok &= exists
        size = p.stat().st_size if exists else 0
        print(
            f"  [{'OK ' if exists and size > 0 else 'MISS'}] {name}"
            + (f"  ({size} bytes)" if exists else "")
        )

    if not all((data_dir / n).is_file() for n in DATA_FILES):
        print("\n[download_data] One or more files are missing. Details:\n")
        print(manual_steps())
        return 1

    cell_path = data_dir / "zscape_perturb_full_cell_metadata.csv"
    gene_path = data_dir / "zscape_perturb_full_gene_metadata.csv"

    # 2) Cell metadata columns.
    cell_cols = _csv_columns(cell_path)
    missing_cell = [c for c in CELL_REQUIRED if c not in cell_cols]
    print(f"\n[download_data] cell_metadata columns: {len(cell_cols)} found")
    if missing_cell:
        ok = False
        print(f"  [FAIL] missing required columns: {missing_cell}")
    else:
        print("  [ OK ] all required columns present")

    # 3) Cell metadata value checks.
    for column, expected in CELL_VALUE_CHECKS.items():
        if column not in cell_cols:
            continue
        values = _csv_values(cell_path, column)
        if isinstance(expected, list):
            missing_vals = [v for v in expected if v not in values]
        else:
            missing_vals = [] if expected in values else [expected]
        if missing_vals:
            ok = False
            print(f"  [FAIL] column '{column}' missing expected value(s): {missing_vals}")
        else:
            print(f"  [ OK ] column '{column}' contains expected value(s): {expected}")

    # 4) Gene metadata columns.
    gene_cols = _csv_columns(gene_path)
    missing_gene = [c for c in GENE_REQUIRED if c not in gene_cols]
    print(f"\n[download_data] gene_metadata columns: {len(gene_cols)} found")
    if missing_gene:
        ok = False
        print(f"  [FAIL] missing required columns: {missing_gene}")
    else:
        print("  [ OK ] all required columns present")

    print(
        "\n[download_data] "
        + ("PASS — data is ready for scripts/recover.py" if ok else "FAIL — see issues above")
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent.parent  # fingeo_supplement/
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir", default=None, help="data directory (default: <project>/data)"
    )
    ap.add_argument(
        "--no-download",
        action="store_true",
        help="do not attempt automatic download; only verify what is already in data/",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-download the files even if they already exist",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-connection network timeout in seconds (default: 300)",
    )
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else here / "data"
    if not data_dir.is_absolute():
        data_dir = (Path.cwd() / data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        for name in DATA_FILES:
            (data_dir / name).unlink(missing_ok=True)
            (data_dir / (name + ".part")).unlink(missing_ok=True)

    if not args.no_download:
        still_missing = download_missing(data_dir, timeout=args.timeout)
        if still_missing:
            print("\n[download_data] Automatic download did not fully succeed. Missing:")
            for name in still_missing:
                print(f"  - {name}")
        else:
            print("\n[download_data] Automatic download complete; verifying...")

    return check(data_dir)


if __name__ == "__main__":
    sys.exit(main())
