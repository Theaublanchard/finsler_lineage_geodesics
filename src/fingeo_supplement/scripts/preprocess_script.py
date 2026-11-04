"""Standalone preprocessing script: load the raw zebrafish data, preprocess,
and write out a saved ``.h5ad`` for later use.
"""

import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd

import os
import sys

from fingeo_supplement.utils.preprocess import *

# TODO: doesn't actually run yet, need to fix module stuff

if __name__ == "__main__":

    filename = "1k_hvg.h5ad"

    adata = import_zebrafish_data()
    # BUG: `preprocess` has no `exclude_highly_expressed` parameter, so this
    # call raises a TypeError. Either remove the kwarg or add the parameter
    # to `preprocess` in `utils/preprocess.py`.
    adata = preprocess(adata, n_top_genes=1000, exclude_highly_expressed=False)
    adata.write(filename)
