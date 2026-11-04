# Learning Lineage-guided Geodesics with Finsler Geometry

Runnable implementation of [**“Learning Lineage-guided Geodesics with Finsler Geometry”**](http://arxiv.org/abs/2603.16708) (Zweig, Zhang, Knowles, Azizi).  The method combines a **continuous geometric prior** (a learned Riemannian/Finsler metric) with a **discrete, directed prior** (a cell-lineage tree) to guide trajectory inference / optimal-transport interpolation between timepoints.

This repo also contains the **recovery tooling** used to pull the *trained* Finsler metric (`finsler_metric.pt`) and, on demand, the **directed cell-level / cell-type-level graphs**  back out of a trained run.

> **Status.** The pipeline is implemented and ready to run end-to-end. The code is directly extracted from the official authors implementation available on openreview. It is just made actually usable and readable with minimum changes.

---

## Repository layout

```
fingeo_supplement/
├── README.md                 # this file
├── AUDIT.md                  # paper ↔ code coherence audit (start here for "does it match the paper?")
├── DATA_DOWNLOAD.md          # where the raw atlas comes from + exact files + how to fetch/verify
├── pyproject.toml / uv.lock  # uv project (Python 3.11; torchcfm pinned from GitHub @ 1.0.7)
├── data/                     # RAW atlas files (you provide) + shared processed cache zebra_preprocessed.h5ad
├── outputs/<tissue>/         # RECOVERED ARTIFACT: finsler_metric.pt (graphs are exported on demand — see below)
├── configs/
│   ├── config.yaml           # global config (latent_dim, hidden_dim, classifier, finsler, metric, ...)
│   ├── dataset/
│   │   ├── zebrafish_cns.yaml     # CNS tissue, curated tree (use_paga: false)
│   │   ├── zebrafish_arch.yaml    # Pharyngeal Arch, PAGA tree (paga_threshold: 0.2)
│   │   └── mouse_blood.yaml / mouse_brain.yaml   # mouse (data not shipped — see the Mouse TODO)
│   └── sweeps/                          # cfm, cfm_finsler, mfm, mfm_finsler, icfm(+_finsler), mfm(+_bayes)(+_finsler)
├── models/
│   ├── base_model.py         # ModelBase: normalize/unnormalize (mean/std buffers), device, ...
│   ├── modules.py            # SimpleEmbedNet, SimpleDenseNet, SinNet, ... (building blocks)
│   ├── classifier_models.py  # cell-type classifier (f: R^n -> Δ^{|C|})
│   ├── metric_models.py      # Finsler metric models: FinslerMixin.forward (the F(x,v) norm)
│   ├── embed_models.py       # Finsler embedding (φ, ψ, β): cost_matrix_fn / embed_loss
│   ├── fm_models.py / cfm.py / ot_sampler.py / ema.py   # flow matching + OT machinery
├── datasets/
│   ├── process.py            # process_data: load → normalize → log1p → HVG2000 → PCA100 → attach tree
│   └── dataset.py            # paired / singleton dataloaders over X_pca
├── scripts/
│   ├── run_model.py          # build_* + run_full_model (classifier→metric→embed) + save_finsler_artifacts
│   ├── recover.py            # ONE-COMMAND driver: process→train→save→verify (writes finsler_metric.pt)
│   ├── download_data.py      # fetch raw atlas (auto-download w/ manual fallback) + readiness check
│   ├── sweep.py              # hyperparameter sweep entry point (self-writes a report per seed)
│   ├── run_sweeps.sh         # driver: runs the cfm/cfm_finsler/mfm/mfm_finsler sweeps for one dataset
│   ├── run_synthetic.py      # synthetic W1 table (no data download; self-writes a report per seed)
│   ├── export_plots.py       # paper figures (river plots, lineage tree, geodesic paths, ...)
│   └── recover_results.py    # AGGREGATE: read outputs/reports/*/report.json → mean±std table / CSV
├── utils/
│   ├── finsler_extract.py    # RECOVERY: FinslerEmbedder, load_finsler_metric, export_directed_graphs (also a CLI), verify_finsler_norm
│   ├── lineage.py            # lineage trees: ZEBRAFISH_NEURAL_ADJACENCY, run_paga_tree, incorporate_tree
│   ├── report.py             # write_run_report: per-run report.json (dataset, seed, w1, config)
│   └── hydra.py / frozen.py / ot.py / callback.py / preprocess.py
├── eval/                     # W1 (Wasserstein) evaluation helpers
├── simulate/                 # synthetic data generators
└── visualize/                # river plots / lineage-tree plots
```

---

## Installation

The project uses `uv` (Python 3.11) to manage the environment and dependencies. The `torchcfm` dependency is installed **from GitHub, pinned to tag `1.0.7`** (see `[tool.uv.sources]` in `pyproject.toml`).

```bash
cd fingeo_supplement
uv sync                       # create .venv + install all deps
# check if the environment is working correctly:
uv run python -c "import torchcfm, scanpy, pytorch_lightning, torchdyn, geomloss, ot"
```

## Data

The zebrafish experiments use the **ZScape Perturb atlas** (Saunders et al., *Nature* 2023). See [`DATA_DOWNLOAD.md`](./DATA_DOWNLOAD.md) for the citation, exact filenames, required columns, and the concrete download (Genetic Perturbation Atlas → “individual parts”, served as gzip). In short:

Let the script fetch (and verify) everything — it auto-downloads any missing files from the URLs above, then checks files + columns + `ctrl-inj` + both tissues:

```bash
uv run python scripts/download_data.py                # auto-download if missing, then verify
uv run python scripts/download_data.py --no-download  # verify only (no network)
```

or manually download the three files into `data/` and gunzip them:

```bash
mkdir -p data
wget -P data \
  http://trapnell-lab-s3-zscape.s3-website-us-west-2.amazonaws.com/zscape_perturb_full_raw_counts.mtx.gz \
  http://trapnell-lab-s3-zscape.s3-website-us-west-2.amazonaws.com/zscape_perturb_full_cell_metadata.csv.gz \
  http://trapnell-lab-s3-zscape.s3-website-us-west-2.amazonaws.com/zscape_perturb_full_gene_metadata.csv.gz
cd data && gunzip *.gz && cd ..
```

`preprocess.py` then does **filter → normalize_total(1e4) → log1p → HVG top-2000** and writes a **shared raw cache** `data/zebra_preprocessed.h5ad` (all tissues, 2000 HVGs — reused across both tissues so `--tissue both` does not re-download). It then, per tissue:

1. subsets to the requested `tissue`,
2. renames `cell_type_broad` → `cell_type`,
3. runs `PCA 100`,
4. attaches the lineage tree (CNS: curated `ZEBRAFISH_NEURAL_ADJACENCY`; PA: PAGA threshold 0.2) and prunes cells outside the tree,
5. **writes a per-tissue processed cache** `data/processed/<slug>.h5ad` (e.g. `data/processed/central_nervous_system.h5ad`, `data/processed/pharyngeal_arch.h5ad`).

The per-tissue cache is the **canonical object** consumed by training, recovery, and the notebooks: it already contains `obsm['X_pca']` (100-d), `obs['cell_type']`, `obs['cell_type_one_hot']`, and `uns['tree'/'stoi'/'itos']`. Subsequent `process_data` calls for the same tissue load this file directly and skip the raw cache, PCA, PAGA, and tree attach. Invalidate with `uv run python scripts/recover.py --tissue <cns|pa> --force` after changing `pc_dim`, the lineage tree, or any preprocessing parameter.

---

## Training & Recovery

> This section is about **recovering the trained artifact** (the metric — and, on demand, the directed graphs) for downstream analysis. To reproduce the paper's **W1 result tables** and **figures**, see [Reproducing the paper&#39;s W1 tables &amp; figures](#reproducing-the-papers-w1-tables--figures) below.

Recovery is **two steps**:

1. **Train + save + verify the metric** — `scripts/recover.py`. One command per tissue (`--tissue cns|pa|both`); it sets `metric=cfm`, `finsler.use=True`, `finsler.lamb=0.5`, trains classifier → metric → embed, saves `outputs/<tissue>/finsler_metric.pt`, and verifies the Finsler norm.

   ```bash
   uv run python scripts/recover.py --tissue cns      # Central Nervous System (curated tree)
   uv run python scripts/recover.py --tissue pa       # Pharyngeal Arch (PAGA tree)
   uv run python scripts/recover.py --tissue both     # both, one process (safe: shared cache)
   uv run python scripts/recover.py --tissue cns --skip-training   # load an existing artifact + verify
   ```
2. **Export the directed graphs, on demand** — `utils/finsler_extract.py` (a CLI, or the `export_directed_graphs` function). This reads the `.pt` + the per-tissue processed cache and writes the cell-level and cell-type-level cost matrices (`.npz` / NetworkX / edge-CSV).

   ```bash
   # CNS (slug = central_nervous_system):
   uv run python utils/finsler_extract.py outputs/cns/finsler_metric.pt \
       --data data/processed/central_nervous_system.h5ad --out outputs/cns
   # PA (slug = pharyngeal_arch):
   uv run python utils/finsler_extract.py outputs/pa/finsler_metric.pt \
       --data data/processed/pharyngeal_arch.h5ad --out outputs/pa
   ```

> **The full cell-level graph is large.** It is the all-pairs cost over every cell (`n_cells × n_cells`, e.g. ~65 500² ≈ **4.3 billion entries** for CNS), so the `.npz` / `.nx.pkl` / `.csv` files are **multi-gigabyte**. That is why `recover.py` does not write them by default and instead defers to the `finsler_extract.py` CLI. For most analyses the `finsler_metric.pt` + `load_finsler_metric` (below) is all you need — compute the cost on a *subsampled* set of cells rather than exporting the full matrix. The type-level files (`type_*`, only `|C| × |C|`) are small and cheap.

The three training phases, in plain terms:

1. **Classifier** (`classifier_models.py`) — learns `f(x)`: the cell-type probability used to define “which lineage direction does moving from `x` in direction `v` agree/disagree with”.
2. **Metric** (`metric_models.py`) — the Finsler local cost `F(x, v)` = Riemannian base + the lineage penalty above (the *target*).
3. **Embed** (`embed_models.py`) — learns `(φ, ψ, β)` so that the cheap cost `‖φᵢ−φⱼ‖ + ReLU(ψᵢ−ψⱼ)` approximates `F`, then OT along that cost interpolates unseen timepoints.

Alternate entry points: `scripts/run_model.py::train` (full pipeline used by the notebooks) and the sweep path `scripts/sweep.py` + `configs/sweeps/cfm_finsler.yaml` (λ, classifier smoothing, OT choice).

---

## The two Finsler objects (read this first)

There are **two distinct Finsler things** in this codebase — keep them straight:

1. **Finsler *embedding* `(φ, ψ, β)` — the trained, recoverable metric used for the graphs.**

   - `models/embed_models.py :: FinslerEmbedNetTrainBase`. A single network `embed_net` (`SimpleEmbedNet`, `rescale=0.5`, `skip=False`) maps normalized input to a latent vector `z` of dimension `latent_dim = 100`, split at `d = len(β) = latent_dim // 2 = 50`:

     - `φ(x) = z[:, d:]` (symmetric / Riemannian part, a vector),
     - `ψ(x) = z[:, :d] @ β` (asymmetric part, a scalar), where `β` is a learned length-50 vector.
   - The **directed (asymmetric) cost** between two cells is

     ```LaTeX
     d_F(x, y) = ‖φ(x) − φ(y)‖₂ + ReLU( ψ(x) − ψ(y) )        # NOT squared
     ```

     (the training code squares it only to feed the `p=2` OT solver; the **recovered graph uses the unsquared** cost above). This is what the directed graphs are built from.
2. **Finsler *norm* `F(x, v)` — the training *target* the embedding regresses toward (for analysis).**

   - `models/metric_models.py :: FinslerMixin.forward`. Combines a Riemannian base `‖v‖_g` with a lineage penalty driven by the classifier `f(x) = softmax(classifier(x)/temp)` and the padded lineage adjacency `A`:

     ```
     F(x, v) = ‖v‖_g  +  λ · (‖v‖_g / ‖v‖) · ReLU( Σ_c f_c(x) · ReLU( (Jf(x)·v) · u_c ) )
     u = 1 − (A + I)
     ```

     The `λ`-weighted term penalizes velocities that move probability mass into lineage-*illegal* directions. Under the `cfm` base (used by `recover.py`), `‖v‖_g = ‖v‖` so the scale is 1.

> The **directed graphs** come from object **1**. The **norm `F(x,v)`** (object 2) is exported for analysis/verification. Both are recovered together from one artifact.

## Where the recovered objects live

The trained metric lives at `outputs/<tissue>/finsler_metric.pt` (`<tissue>` ∈ `{cns, pa}`) — created by `recover.py`. The graph files are **not** produced by `recover.py`; they are written on demand by step 2 above (the `finsler_extract.py` CLI) into the same `outputs/<tissue>/` dir:

| File                                                             | Contents                                                                                                                                                  | How to load                                                  |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `finsler_metric.pt`                                            | `embed_net` weights, `β`, `mean`/`std`, classifier, padded `tree`, `stoi`/`itos`, `num_classes`, `finsler{lamb,temp,use}`, full config | `utils.finsler_extract.load_finsler_metric(path)`          |
| `cell_matrix.npz`, `cell_digraph.nx.pkl`, `cell_edges.csv` | **cell-level** directed cost `M[i,j] = ‖φᵢ−φⱼ‖ + ReLU(ψᵢ−ψⱼ)` (asymmetric) — *on-demand*                                           | `np.load` / `networkx.read_pickle` / `pandas.read_csv` |
| `type_matrix.npz`, `type_digraph.nx.pkl`, `type_edges.csv` | **cell-type-level** aggregate directed cost (mean over cell pairs per type) — compare vs tree `A` — *on-demand*                               | same                                                         |

Load & compute example:

```python
from utils.finsler_extract import load_finsler_metric
from datasets.process import process_data

# process_data returns the per-tissue processed object (X_pca, cell_type, tree).
# If data/processed/central_nervous_system.h5ad exists it is loaded directly;
# otherwise it is built from the raw cache and written there for next time.
adata = process_data(
    pc_dim=100,                    # MUST match the artifact's trained pc_dim
    t0_index=1, t1_index=4,
    data="zebrafish", use_paga=False,
    tissue="Central Nervous System",
)

# OR : One-off (once the per-tissue cache exists) — no process_data needed:
import anndata
adata = anndata.read_h5ad("data/processed/central_nervous_system.h5ad")

emb   = load_finsler_metric("outputs/cns/finsler_metric.pt")
X     = adata.obsm["X_pca"]
ct    = adata.obs["cell_type"].tolist()

# Work on a subsample — the full n_cells x n_cells cost is ~4.3B entries for CNS (see above).
Xi    = X[:500]
ci    = ct[:500]

M     = emb.directed_cost(Xi)                # (500, 500) asymmetric cost on the subsample
out   = emb.build_directed_graph(Xi, ci)     # type-level matrix (small: |C| x |C|) is cheap
F     = emb.finsler_norm(X[:32], (X[1:33] - X[:32]))   # F(x, v) on sample tangent pairs
```

> **`pc_dim` must match the trained artifact.** The `embed_net` input dimension is fixed at training time (CNS config: `pc_dim=100`). Loading a `finsler_metric.pt` trained with a different `pc_dim` will raise a shape mismatch in `load_finsler_metric`.

---

## Reproducing the paper's W1 tables & figures

The paper's result tables report the **Wasserstein-1 distance averaged over the held-out intermediate timepoints**, then `mean ± std` over **10 independent seeds**.
Each cell is produced by:

1. `scripts/run_model.py::train()` — trains classifier → metric → embed and runs the held-out W1 loop (`eval/eval.py::predict` with exact OT via `library="pot"`),
2. `scripts/sweep.py` — runs `train()` once per `seed` in a sweep config and logs `w1_avg` to **WandB**. The mean/std over the 10 seeds is the table entry.

The qualitative **figures** (river plots, geodesic paths, lineage trees, classifier heatmaps) are exported by `scripts/export_plots.py` (a single seed — no WandB).

### Prerequisites

- `uv sync` done, and the raw atlas in `data/` (see [Data](#data) and `DATA_DOWNLOAD.md`).
- **WandB configured** (`wandb login`) for the W1 sweeps — results are logged to the WandB dashboard under the dataset's project name.
- A **GPU** is strongly recommended: training is much faster on GPU, and the `geomloss` OT path in `eval/eval.py` calls `.cuda()`. (The W1 evaluation itself uses the CPU-safe `pot` exact-OT path, so the numbers are unaffected by device.)

### Row → config mapping

| Table row   | `metric` | `finsler.use` | Sweep config                        |
| ----------- | ---------- | --------------- | ----------------------------------- |
| CFM         | `cfm`    | `false`       | `configs/sweeps/cfm.yaml`         |
| CFM+Finsler | `cfm`    | `true`        | `configs/sweeps/cfm_finsler.yaml` |
| MFM         | `mfm`    | `false`       | `configs/sweeps/mfm.yaml`         |
| MFM+Finsler | `mfm`    | `true`        | `configs/sweeps/mfm_finsler.yaml` |

> **LineageOT, Moslin, and SF²M** are *external* baselines — they are referenced in the paper, but **not implemented in this repo**. Their rows come from  the authors running those upstream implementations.

### Zebrafish tables (`tab:zebrafish` — CNS + PA)

```bash
# All four methods (cfm / cfm_finsler / mfm / mfm_finsler), 20 trials each
# (2 classifier-smoothing values × 10 seeds):
uv run bash scripts/run_sweeps.sh --dataset zebrafish_cns --count 20
uv run bash scripts/run_sweeps.sh --dataset zebrafish_arch --count 20

# Or a single method:
uv run python scripts/sweep.py --sweep_config configs/sweeps/cfm_finsler.yaml \
    --dataset zebrafish_cns --count 20
```

- **CNS**: curated lineage tree (`ZEBRAFISH_NEURAL_ADJACENCY`), `t0_index=1` (24 hpf) → `t1_index=4` (72 hpf); held-out = timepoints 2 and 3.
- **PA**: PAGA tree (threshold 0.2); same start/end timepoints.

### Mouse-Blood table (`tab:mouse`)

```bash
uv run bash scripts/run_sweeps.sh --dataset mouse_blood --count 20
```

> **TODO.** The mouse data (source atlas, download path, and documentation) are not yet provided in this supplement — *the author will handle the download and documentation of this section.* The code path itself is complete and expects the atlas under `data/atlas/`.

- `t0_index=1` (E7.0) → `t1_index=7` (E8.5); held-out = the 5 intermediate timepoints (matching the paper's "mean W1 over the 5 held-out time points").

### Synthetic table (`tab::syn`)

```bash
# Both dims (2 and 50), both methods (CFM / CFM+Finsler), 10 seeds — no data
# download and no WandB required (results print to stdout):
uv run python scripts/run_synthetic.py

# A single setting:
uv run python scripts/run_synthetic.py --dims 2 --finsler --n_seeds 10

# Write a JSON summary:
uv run python scripts/run_synthetic.py --out results/synthetic.json
```

- Data is generated in memory by `simulate.simulate.sample_path` — nothing to download.
- `d=2`: `finsler.lamb=5.0`, `classifier.epsilon=0.03`.
- `d=50`: `finsler.lamb=1.0`, `classifier.epsilon=0.01`.

### Reading the results

Every W1-producing run **self-writes a report** on completion — no dashboard or scraping needed. Both `sweep.py` and `run_synthetic.py` call `utils/report.py::write_run_report`, which drops a single file per run:

```
outputs/reports/<dataset>_<method>_seed<N>_eps<E>/report.json
```

`report.json` holds the identifying fields (dataset, method, seed, `classifier_epsilon`, `finsler_lamb`, …), the W1 numbers (`w1_avg`, `w1_val_avg`, `w1_per_timepoint`), the git commit, and the full resolved config.

One command then aggregates all reports into a legible table (mean ± std of `w1_avg` over the seeds in each `(dataset, method, epsilon)` group):

```bash
# Aggregated table (mean ± std per group):
uv run python scripts/recover_results.py

# Filter to one dataset / method:
uv run python scripts/recover_results.py --dataset zebrafish_arch
uv run python scripts/recover_results.py --method "CFM+Finsler"

# Also dump a flat CSV, one row per seed (the raw, un-pooled values):
uv run python scripts/recover_results.py --out results/all_seeds.csv
```

- **W1 sweeps**: `w1_avg` is also still logged to **WandB** for the dashboard; the table entry is the `mean ± std` of `w1_avg` over the 10 seeds. (`sweep.py` also logs `w1_val_avg` for the validation timepoint.)
- **`run_synthetic.py`**: prints the `mean ± std` table directly and can write JSON (`--out results/synthetic.json`).

### Generating the paper figures

The qualitative plots are exported to `outputs/<dataset>/<metric>_finsler/` (e.g. `outputs/zebrafish_cns/cfm_finsler/`) by `scripts/export_plots.py` (single seed, no WandB):

```bash
# River plot, lineage tree, data scatter, geodesic paths:
uv run python scripts/export_plots.py --dataset zebrafish_cns --finsler
uv run python scripts/export_plots.py --dataset zebrafish_arch --finsler
uv run python scripts/export_plots.py --dataset mouse_blood --finsler

# Plus classifier confidence/entropy heatmaps (2D synthetic):
uv run python scripts/export_plots.py --dataset synthetic --finsler --dim 2

# Non-Finsler (CFM-only) variant, for the side-by-side river-plot comparison:
uv run python scripts/export_plots.py --dataset zebrafish_cns --no-finsler

# Fast sanity check (cap the slow embed phase; NOT for paper-quality figures):
uv run python scripts/export_plots.py --dataset synthetic --finsler --dim 2 --embed-epochs 1
```

> **Training budget.** The three trained phases each have their own epoch knob: `classifier_max_epochs`, `metric_max_epochs` (hard-capped to 2 for the CFM path), and `embed_max_epochs` (default 3000 — the main trained phase). Use `--embed-epochs` on `export_plots.py` to lower it for a quick smoke test; keep the default for the actual paper figures.

Figure output:

| File                                                                                                            | Contents                                                                                                                         |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `river_plot.png`                                                                                              | Classifier cell-type probability along N geodesics (the `fig:zebrafish-two-panel` / `fig:mouse-blood-two-panel` river plots) |
| `lineage_tree.png`                                                                                            | Directed lineage-tree prior (`adata.uns['tree']`)                                                                              |
| `scatter_data.png`                                                                                            | Data in 2D, colored by timepoint                                                                                                 |
| `geodesic_paths.png`                                                                                          | Learned geodesic trajectories overlaid on the data                                                                               |
| `classifier_pred.png`, `classifier_pred_map.png`, `classifier_confidence.png`, `classifier_entropy.png` | Classifier diagnostics (2D datasets only)                                                                                        |

> For **interactive exploration** (live plots in the notebook), use the notebooks `holdout_toy.ipynb` (synthetic), `zebrafish.ipynb` (zebrafish), and `mouse_lineage.ipynb` (mouse) — each runs one seed and prints `mean(w1_scores)`.

---

## Coherence with the paper

The code faithfully implements the paper’s Finsler metric, penalty, and embedding parameterization for the `cfm` configuration used by `recover.py`. A few implementation details diverge from the paper’s idealized form (most notably a **dummy “other” class** not described in the paper, plus an equivalent-but-differently-stated diagonal convention and a ReLU-placement nuance).

### Known issues

- **Sweep dataset default.** `sweep.py` / `run_sweeps.sh` default `--dataset` to `zebrafish`, which has no matching `configs/dataset/` file (only `zebrafish_cns` / `zebrafish_arch` exist). Always pass `--dataset` explicitly.
- **Mouse data.** Not shipped; download + source documentation pending (see the Mouse-Blood section above).


### Divergences with method stated in the paper

Severity: **cosmetic** (notation only) · **semantic** (changes the value of the recovered object) · **bug** (incorrect code path).

#### D1 — Dummy “other” class — **semantic**

The classifier is built with `output_dim = num_classes + 1` (`scripts/run_model.py :: build_classifier`) and regularized toward a *dummy* class via `dummy_prior` / `dummy_kl_weight` (`models/classifier_models.py :: _compute_loss`). The tree is padded to `(|C|+1)²` (`FinslerMixin._pad_tree`), so `FinslerMixin.forward` sums over **`|C|+1`** classes, the last being the dummy “other” class.

**Paper:** never mentions a dummy class; `f: Rⁿ → Δ^{|C|}` is exactly `|C|` wide and not `|C|+1`.

**Effect:** the implemented norm is `|C|+1`-wide. This is a legitimate implementation choice (probabilistic “not any of the known types” mass), but the paper’s stated `F` does not include it. Recovery (this repo) inherits it — the recovered `finsler_metric.pt` carries the padded tree and the `|C|+1`-wide classifier.

#### D2 — Diagonal convention of `A` — **cosmetic**

The paper states `A_{ii} = 1` (“cells can remain in their state”), so `(1−Aᵀ)_{ii} = 0` — staying in your state is *legal*. The code builds `A` with a **zero** diagonal (`utils/lineage.py :: incorporate_tree` never sets `tree[i,i] = 1`) and then uses `u = 1 − (A + I)`, i.e. `u_{ii} = 1 − 0 − 1 = °0`.

**Net:** the set of penalized (illegal) directions is **identical** — both exclude the diagonal. The difference is only in the *stated* convention (paper: explicit `A_ii=1`; code: `A_ii=0` + added identity). Recorded for clarity; no behavioral difference.

#### D3 — ReLU placement — **cosmetic → borderline semantic**

Paper: per-term ReLU, `Σ_c f_c · ⟨v, w_c⟩₊` (each class contribution clipped individually, then summed). Code: `relu( Σ_c f_c · ⟨v, w_c⟩ )` — the **whole sum** is clipped once (and in `forward` there is an outer `relu(finsler_term)` on top of the per-class `relu` inside the sum).

**Effect:** when the inner per-class terms have mixed sign, the two forms can differ by the amount of cancellation. In practice `f_c ≥ 0` and the inner `relu` already drops negatives, so the outer `relu` of a non-negative sum is a no-op; the divergence is effectively inert. Recorded anyway for exactness.

#### D4 — Config vs. paper hyperparameters — **cosmetic**

- **Batch sizes.** Paper (Param. Details): 512 for classification & metric, 2048 for embedding & geodesic. Code: `classifier`/`metric` `loader_batch_size: 6`, `score_batch_size: 2048`, `flow_batch_size: 512`. The small `loader_batch_size` is used for the OT-coupling batches, not the 512 the paper quotes for classifier/metric.
- **`finsler.temp`.** The code exposes a classifier softmax temperature `temp` (default `1.0`, used in `classifier_fn`); the paper does not name a `temp` hyperparameter (it is folded into the learned classifier). At the default `temp = 1.0` the two agree.

#### D6 — `build_metric` “gaga” branch is a no-op — **bug (unreachable)**

`scripts/run_model.py :: build_metric` has `elif config.metric == "gaga": pass` — so `metric_model` is never assigned and the subsequent `return metric_model` would raise `NameError`. `recover.py` pins `metric = "cfm"`, so this branch is never taken. Recorded for completeness.

---
