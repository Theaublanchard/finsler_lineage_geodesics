"""Build and validate lineage (differentiation) adjacency matrices for cell types.

Provides curated adjacency constants (zebrafish / arch neural, CITE) and
helpers to enforce a rooted DAG on a PAGA tree and to store an adjacency
matrix on an ``AnnData`` object for the Finsler lineage penalty.
"""

import itertools
from collections import deque
import numpy as np
import scanpy as sc
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

ZEBRAFISH_NEURAL_ADJACENCY = {
    'neural progenitor (telencephalon/diencephalon)': [
        'differentiating neuron 1',
        'differentiating neuron 2',
    ],
    'neural progenitor (MHB)': ['differentiating neuron 1', 'differentiating neuron 2'],
    'neural progenitor (hindbrain)': ['differentiating neuron (hindbrain)'],
    'neural progenitor (hindbrain R7/8)': [
        'differentiating neuron (hindbrain)',
        'differentiating neuron 1',
        'differentiating neuron 2',
        'motor neuron',
    ],
    'posterior spinal cord progenitors': [
        'neuron (+ spinal cord)',
        'dorsal spinal cord neuron',
        'motor neuron',
    ],
    'differentiating neuron (hindbrain)': [
        'hypophysis/locus coeruleus',
        'neuron (+ spinal cord)',
    ],
    'differentiating neuron 1': [
        'neuron (dopaminergic)',
        'neurons (gabaergic, glutamatergic; contains Purkinje)',
        'neurons (gabaergic, glutamatergic)',
        'neuron (telencephalon, glutamatergic)',
    ],
    'differentiating neuron 2': [
        'neuron (dopaminergic)',
        'neurons (gabaergic, glutamatergic; contains Purkinje)',
        'neurons (gabaergic, glutamatergic)',
        'neuron (telencephalon, glutamatergic)',
    ],
    'neurons (differentiating, contains peripheral)': [
        'neurons (gabaergic, glutamatergic; contains Purkinje)',
        'neurons (gabaergic, glutamatergic)',
        'neuron (telencephalon, glutamatergic)',
    ],
}

ARCH_NEURAL_ADJACENCY = {
    'head mesenchyme (maybe ventral, hand2+)': [
        'head mesenchyme/PA cartilage',
        'pharyngeal arch (NC-derived)',
    ],
    'pharyngeal arch (NC-derived)': [
        'jaw chondrocyte',
    ],
    'head mesenchyme/PA cartilage': [
        'pharyngeal arch (early)',
    ],
    'pharyngeal arch (early)': [
        'pharyngeal arch (contains muscle, early cartilage)',
    ],
    'pharyngeal arch (contains muscle, early cartilage)': [
        'jaw chondrocyte',
        'chondrocranium',
    ],
}

CITE_ADJACENCY = {'HSC': ['NeuP', 'EryP', 'MasP', 'MkP', 'MoP', 'BP']}


def enforce_dag_from_root(
    adata, root_cell_type: "str | list[str]", groups: str = 'cell_type', threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Prune the PAGA connectivities into a DAG rooted at a starting cell type.

    Performs a BFS from ``root_cell_type`` over the PAGA graph, assigning
    each node a level, then keeps only the edges that go from a lower level
    to a higher level (forward differentiation direction).

    Parameters
    ----------
    adata
        AnnData with a PAGA graph in ``uns['paga']``.
    root_cell_type : str | list[str]
        Cell-type label (or list of labels) to use as the DAG root(s).
    groups : str
        Obs column holding the cell-type labels.
    threshold : float
        Minimum PAGA connectivity to consider an edge present.

    Returns
    -------
    directed_connectivity : np.ndarray
        The pruned directed connectivity matrix.
    levels : np.ndarray
        BFS level of each node (-1 for unreachable).
    """

    connectivity = adata.uns['paga']['connectivities'].toarray()
    n_nodes = connectivity.shape[0]

    categories = adata.obs[groups].cat.categories

    # Support single root or list of roots
    if isinstance(root_cell_type, str):
        root_cell_type = [root_cell_type]
    root_idxs = [np.where(categories == r)[0][0] for r in root_cell_type]

    directed_connectivity = np.zeros_like(connectivity)

    levels = np.full(n_nodes, -1)
    queue = deque()
    for idx in root_idxs:
        levels[idx] = 0
        queue.append(idx)

    while queue:
        current = queue.popleft()
        current_level = levels[current]

        neighbors = np.where(connectivity[current] > threshold)[0]

        for neighbor in neighbors:
            if levels[neighbor] == -1:  # Unvisited
                levels[neighbor] = current_level + 1
                queue.append(neighbor)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if connectivity[i, j] > 0:  # There's an edge
                if levels[i] < levels[j]:  # i -> j (forward)
                    directed_connectivity[i, j] = connectivity[i, j]
                elif levels[i] > levels[j]:  # j -> i (reverse)
                    directed_connectivity[j, i] = connectivity[i, j]

    adata.uns['paga']['connectivities_dag'] = csr_matrix(directed_connectivity)
    adata.uns['paga']['levels'] = levels

    return directed_connectivity, levels


def run_paga_tree(
    adata, cell_type_key: str, threshold: float = 0.2, root_node: str = ""
) -> dict[str, list[str]]:
    """Run PAGA on an AnnData object and derive a lineage tree.

    Parameters
    ----------
    adata
        The (preprocessed) AnnData object.
    cell_type_key : str
        Obs column holding the cell-type labels.
    threshold : float
        Minimum adjacency weight kept in the tree.
    root_node : str
        Optional root cell type for the tree.
    """
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.paga(adata, groups=cell_type_key)
    categories = adata.obs[cell_type_key].cat.categories.tolist()

    directed_conn, _ = enforce_dag_from_root(
        adata, root_cell_type=root_node, groups='cell_type', threshold=threshold
    )

    # Build adjacency dictionary
    adj = {}
    n = len(categories)
    for i in range(n):
        neighbors = []
        for j in range(n):
            if directed_conn[i, j] > threshold:
                neighbors.append(categories[j])
        if neighbors:
            adj[categories[i]] = neighbors

    return adj


# TODO: we currently prune to only have cell_types in the tree
def incorporate_tree(adata, adj: dict[str, list[str]], cell_type_key: str):
    """Store a lineage adjacency matrix on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to annotate.
    adj : dict[str, list[str]]
        A mapping of cell types to their allowed descendants.
    cell_type_key : str
        Obs column holding the cell-type labels.
    """
    # cell_types = adata.obs[cell_type_key].unique().tolist()
    # tree_cell_types = list(adj.keys()) + list(itertools.chain(*adj.values()))
    # cell_types = list(set(cell_types + tree_cell_types))

    tree_cell_types = list(adj.keys()) + list(itertools.chain(*adj.values()))
    tree_cell_types = sorted(set(tree_cell_types))  # sort for deterministic stoi/itos ordering
    adata = adata[adata.obs[cell_type_key].isin(tree_cell_types)]
    cell_types = tree_cell_types

    stoi = {name: i for i, name in enumerate(cell_types)}
    itos = {i: name for i, name in enumerate(cell_types)}

    N = len(cell_types)
    tree = np.zeros((N, N))

    for u, neighbors in adj.items():
        for v in neighbors:
            tree[stoi[u], stoi[v]] = 1

    adata.uns['stoi'] = stoi
    adata.uns['itos'] = itos
    adata.uns['tree'] = tree
    adata.obs['cell_type_one_hot'] = [stoi[x] for x in adata.obs[cell_type_key]]
    return adata
