"""Data-loading and preprocessing pipeline for the real datasets.

``process_data`` runs PAGA / HVG selection and sets up the AnnData object
(the lineage ``tree``, ``X_pca``, etc.); the ``extract_*_dataset`` helpers
build the singleton and paired sample arrays consumed by the dataloaders.
"""

import os
import numpy as np
import scanpy as sc
import torch
import pandas as pd
from fingeo_supplement.datasets.dataset import *
from fingeo_supplement.utils.preprocess import *
from fingeo_supplement.utils.lineage import *
from fingeo_supplement.paths import FINGEO_DATA_DIR, processed_cache_path
import networkx as nx
import scipy.io as sio
from pathlib import Path

"""Dataset loading, preprocessing, and lineage-tree construction.

Builds an AnnData object (PCA-reduced, tissue-subset, lineage tree
attached) from the zebrafish, CITE, or mouse atlases, plus helpers to
extract paired and singleton training datasets.
"""


def process_data(
    pc_dim: int,
    t0_index: int,
    t1_index: int,
    data: str = "zebrafish",
    use_paga: bool = False,
    paga_threshold: float = 0.2,
    tissue: str = "Central Nervous System",
):
    """Load, preprocess, and attach a lineage tree for a dataset.

    Loads the raw atlas (zebrafish, CITE, or mouse), applies standard
    Scanpy preprocessing (filter, normalize, log1p, HVG, PCA), subsets
    to the requested tissue, builds the lineage adjacency matrix (from
    hard-coded adjacencies or PAGA), and attaches ``cell_type_one_hot``.

    Parameters
    ----------
    pc_dim : int
        Number of PCA components to keep.
    t0_index : int
        Index of the start timepoint (within the observed timepoints).
    t1_index : int
        Index of the end timepoint.
    data : str
        Dataset name: ``"zebrafish"``, ``"cite"``, or ``"mouse"``.
    use_paga : bool
        If True, build the lineage tree via PAGA (else use hard-coded).
    paga_threshold : float
        PAGA connectivity threshold.
    tissue : str
        Tissue to subset to (e.g. ``"Central Nervous System"``,
        ``"Pharyngeal Arch"``, ``"blood"``, ``"brain"``).

    Returns
    -------
    adata : anndata.AnnData
        Preprocessed AnnData with the tree in ``uns['tree']``.
    """
    path = FINGEO_DATA_DIR
    if data == "zebrafish":
        # ---- per-tissue processed cache (fast path) ----------------------
        # If we already built the final object for this tissue (subset +
        # cell_type rename + PCA + tree), load and return it directly.
        # This is the object the README / recovery code actually consumes.
        pcache = processed_cache_path(tissue)
        if os.path.exists(pcache):
            print(f"Loading per-tissue processed cache: {pcache}")
            return sc.read(pcache)

        suffix = "zebra_preprocessed.h5ad"
        filename = os.path.join(path, suffix)
        if os.path.exists(filename):
            adata = load_data(filename)
        else:
            print("Loading raw zebrafish atlas data...")
            mtx_suffix = "zscape_perturb_full_raw_counts.mtx"
            cell_suffix = "zscape_perturb_full_cell_metadata.csv"
            gene_suffix = "zscape_perturb_full_gene_metadata.csv"

            mtx_filename = os.path.join(path, mtx_suffix)
            cell_filename = os.path.join(path, cell_suffix)
            gene_filename = os.path.join(path, gene_suffix)

            adata = sc.read_mtx(mtx_filename)
            cell_metadata = pd.read_csv(cell_filename)
            gene_metadata = pd.read_csv(gene_filename)

            adata = adata.transpose()
            adata.obs = cell_metadata
            adata.var = gene_metadata

            adata.var["gene_short_name"] = adata.var["gene_short_name"].astype(str)
            adata.var_names = adata.var["gene_short_name"].values
            adata.var_names_make_unique()

            adata = adata[adata.obs['gene_target'] == 'ctrl-inj']

            sc.pp.filter_cells(adata, min_genes=100)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)

            adata = adata[:, adata.var['highly_variable']].copy()

            adata.write(filename)
            print(f"Saved preprocessed data to {filename}")

        subset = adata.obs['tissue'] == tissue
        adata = adata[subset]
        adata.obs['cell_type'] = adata.obs['cell_type_broad']
        sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)

        if tissue == "Central Nervous System" and not use_paga:
            adj = ZEBRAFISH_NEURAL_ADJACENCY
        elif tissue == "Pharyngeal Arch" and not use_paga:
            adj = ARCH_NEURAL_ADJACENCY
        else:
            print("using paga")
            cns_root_nodes = [
                'neural progenitor (telencephalon/diencephalon)',
                'neural progenitor (MHB)',
                'neural progenitor (hindbrain)',
                'neural progenitor (hindbrain R7/8)',
                'posterior spinal cord progenitors',
            ]
            arch_root_nodes = [
                'cranial muscle (progenitor)',
                'head mesenchyme (maybe ventral, hand2+)',
            ]
            root_node = ""
            if tissue == "Central Nervous System":
                root_node = cns_root_nodes
            elif tissue == "Pharyngeal Arch":
                root_node = arch_root_nodes
            adj = run_paga_tree(
                adata, 'cell_type', threshold=paga_threshold, root_node=root_node
            )
        adata = incorporate_tree(adata, adj, 'cell_type')

        # ---- persist the per-tissue processed object --------------------
        # Write the final (tree-pruned) adata to a per-tissue cache so that
        # subsequent loads (notebooks, recovery, eval) can read it directly
        # with anndata.read_h5ad without re-running PCA / PAGA / tree attach.
        #
        # Two h5ad-serialization quirks must be worked around before writing:
        #   1. incorporate_tree stores uns['itos'] with INTEGER keys
        #      ({0: name, 1: name, ...}); anndata serializes dict values as
        #      HDF5 groups whose keys must be strings, so an int-keyed dict
        #      raises "AttributeError: 'int' object has no attribute 'rsplit'".
        #   2. Some cell-type names contain '/' (e.g. PAGA's
        #      "head mesenchyme/PA cartilage"); h5py forbids '/' in group
        #      keys ("Incompatible object (Dataset) already exists").
        # We sanitize the keys in place for the write (recursively, in case
        # PAGA attached nested dicts), then restore the originals so the
        # in-memory object returned to training is unchanged. The sole
        # consumer of uns['itos'] (scripts/run_model.py) does ``int(k)`` on
        # the keys, so either form works.
        def _safe_key(k):
            s = k if isinstance(k, str) else str(k)
            return s.replace("/", "∕")  # U+2215 division slash (h5py-safe)

        def _sanitize(value):
            if isinstance(value, dict):
                return {_safe_key(k): _sanitize(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(_sanitize(v) for v in value)
            return value

        pcache = processed_cache_path(tissue)
        os.makedirs(os.path.dirname(pcache), exist_ok=True)
        saved = {}
        for key in ("stoi", "itos"):
            if isinstance(adata.uns.get(key), (dict, list, tuple)):
                saved[key] = adata.uns[key]
                adata.uns[key] = _sanitize(saved[key])
        try:
            adata.write(pcache)
        finally:
            for key, original in saved.items():
                adata.uns[key] = original
        print(f"Saved per-tissue processed cache to {pcache}")

    elif data == "cite":
        suffix = "cite.h5ad"
        filename = os.path.join(path, suffix)
        adata = sc.read(filename)
        adata.obs['gene_target'] = ['ctrl-inj'] * adata.shape[0]
        adata.obs['timepoint'] = adata.obs['day']

        adata = incorporate_tree(adata, CITE_ADJACENCY, 'cell_type')

    elif data == "mouse":
        BLOOD = [
            "Blood progenitors 1",
            "Blood progenitors 2",
            "Haematoendothelial progenitors",
            "Erythroid1",
            "Erythroid2",
            "Erythroid3",
        ]

        BRAIN = [
            "Spinal cord",
            "Caudal neurectoderm",
            "Caudal epiblast",
            "NMP",
        ]
        cache_file = os.path.join(path, "atlas", "mouse_preprocessed.h5ad")
        if os.path.exists(cache_file):
            print("Loading cached preprocessed mouse data...")
            adata = sc.read(cache_file)
        else:
            print("Loading raw mouse atlas data...")

            counts = sio.mmread(os.path.join(path, "atlas", "raw_counts.mtx")).T.tocsr()
            meta = pd.read_csv(os.path.join(path, "atlas", "meta.csv"))
            genes = pd.read_csv(
                os.path.join(path, "atlas", "genes.tsv"),
                sep='\t',
                header=None,
                names=['ensembl_id', 'gene_symbol'],
            )

            adata = sc.AnnData(X=counts)
            adata.obs = meta.reset_index(drop=True)
            adata.obs.index = adata.obs['cell'].values
            adata.var_names = genes['gene_symbol'].values
            adata.var_names_make_unique()

            is_doublet = adata.obs['doublet'].astype(str).str.upper() == 'TRUE'
            is_stripped = adata.obs['stripped'].astype(str).str.upper() == 'TRUE'
            has_celltype = adata.obs['celltype'].notna()
            valid_stage = adata.obs['stage'] != 'mixed_gastrulation'
            mask = ~is_doublet & ~is_stripped & has_celltype & valid_stage
            adata = adata[mask].copy()

            sc.pp.filter_cells(adata, min_genes=100)
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            adata = adata[:, adata.var['highly_variable']].copy()

            adata.write(cache_file)
            print(f"Saved preprocessed data to {cache_file}")

        adata.obs['cell_type'] = adata.obs['celltype'].astype(str)

        stage_to_num = {
            'E6.5': 6.5,
            'E6.75': 6.75,
            'E7.0': 7.0,
            'E7.25': 7.25,
            'E7.5': 7.5,
            'E7.75': 7.75,
            'E8.0': 8.0,
            'E8.25': 8.25,
            'E8.5': 8.5,
        }
        adata.obs['timepoint'] = adata.obs['stage'].map(stage_to_num).astype(float)

        # SUBSET TO TISSUE
        if tissue == "brain":
            cell_list = BRAIN
            root_node = ["NMP", "Caudal epiblast"]
        elif tissue == "blood":
            cell_list = BLOOD
            root_node = "Haematoendothelial progenitors"
        else:
            cell_list = None
            root_node = None
        adata = adata[adata.obs['cell_type'].isin(cell_list)].copy()
        adata.obs['cell_type'] = (
            adata.obs['cell_type'].astype('category').cat.remove_unused_categories()
        )

        # PCA
        sc.tl.pca(adata, n_comps=pc_dim, mask_var=None)
        adata.uns['std'] = np.ones((1, pc_dim))

        # Tree: always use PAGA for mouse
        if use_paga:
            print("using paga")

        adj = run_paga_tree(adata, 'cell_type', threshold=paga_threshold, root_node=root_node)
        adata = incorporate_tree(adata, adj, 'cell_type')

    return adata


def extract_paired_dataset(adata, use_rep='X_pca'):

    dataset = []
    seen_timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    for i in range(len(seen_timepoints) - 1):
        t0 = seen_timepoints[i]
        t1 = seen_timepoints[i + 1]
        adata0 = adata[adata.obs['timepoint'] == t0]
        adata1 = adata[adata.obs['timepoint'] == t1]
        X0 = torch.from_numpy(adata0.obsm[use_rep]).float()
        X1 = torch.from_numpy(adata1.obsm[use_rep]).float()
        obj = (X0, X1, t0, t1)
        dataset.append(obj)

    return dataset


def extract_singleton_dataset(adata, use_rep='X_pca'):

    dataset = torch.from_numpy(adata.obsm[use_rep]).float()
    y = torch.from_numpy(adata.obs['cell_type_one_hot'].to_numpy()).long()
    return dataset, y
