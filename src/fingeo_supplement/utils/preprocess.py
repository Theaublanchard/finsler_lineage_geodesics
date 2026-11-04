"""Preprocessing helpers for the zebrafish (zScope) lineage dataset.

Loads the raw MatrixMarket counts + metadata, runs a standard Scanpy
pipeline (filter, normalize, HVG selection per perturbation target), and
exposes loaders for saved datasets.
"""

import numpy as np
import torch
import scanpy as sc
import anndata as ad
import os
import pandas as pd


def import_zebrafish_data() -> ad.AnnData:
    """Load the raw zebrafish zScope perturbation dataset.

    Reads the MatrixMarket count matrix together with the cell / gene
    metadata CSVs, transposes so rows are cells, and attaches the metadata
    as ``obs`` / ``var``.

    Returns
    -------
    adata : anndata.AnnData
        The raw (unnormalized) dataset.
    """

    path = "data"
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

    return adata


def preprocess(
    adata: ad.AnnData, min_genes: int = 100, min_cells: int = 3, n_top_genes: int = 2000
) -> ad.AnnData:
    """Run the full preprocessing pipeline on the zebrafish dataset.

    Filters cells / genes, normalizes to total counts, log-transforms,
    and selects highly variable genes per perturbation target (unioned
    with the ctrl-inj comparison).

    Parameters
    ----------
    adata : anndata.AnnData
        The raw AnnData object (see ``import_zebrafish_data``).
    min_genes : int
        Minimum number of expressed genes for a cell to be kept.
    min_cells : int
        Minimum number of expressing cells for a gene to be kept.
    n_top_genes : int
        Number of top highly-variable genes to select per target.

    Returns
    -------
    adata : anndata.AnnData
        The preprocessed dataset, with ``var['highly_variable']`` set.
    """
    # BUG: The incoming `adata` argument is discarded -- it is immediately
    # overwritten by `import_zebrafish_data()`. Remove this reassignment (or
    # remove the `adata` parameter) so the caller's object is actually used.
    adata = import_zebrafish_data()
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.var["gene_short_name"] = adata.var["gene_short_name"].astype(str)
    adata.var_names = adata.var["gene_short_name"].values
    adata.var_names_make_unique()

    targets = list(adata.obs['gene_target'].unique())
    hvg = pd.Series(False, index=adata.var_names)

    for gene_target in targets:
        adata_paired = adata[
            (adata.obs['gene_target'] == 'ctrl-inj')
            | (adata.obs['gene_target'] == gene_target)
        ]
        sc.pp.highly_variable_genes(adata_paired, n_top_genes=n_top_genes)
        hvg = hvg | adata_paired.var['highly_variable']

    adata.var['highly_variable'] = hvg
    return adata


# TODO: this omits all the weird interventions?  Like *-mut interventions?
def load_data(filename: str) -> ad.AnnData:
    """Load a saved dataset and keep HVGs plus the perturbation target genes.

    Parameters
    ----------
    filename : str
        Path to a saved ``.h5ad`` file.

    Returns
    -------
    adata : anndata.AnnData
        The loaded dataset restricted to highly-variable and target genes.
    """
    adata = sc.read(filename)
    targets = list(adata.obs['gene_target'].unique())
    hvg = adata.var['highly_variable']
    perturbed = adata.var['gene_short_name'].isin(targets)
    adata = adata[:, (hvg) | (perturbed)]
    return adata
