"""Synthetic 2D/3D trajectory datasets for testing the Finsler pipeline.

Each ``sample_*`` function builds a small trajectory with cell-type labels
(``y``), timepoints (``t``), and a lineage adjacency matrix (``tree``),
optionally with Gaussian noise. ``process_fake_data`` wraps the output into
an AnnData object compatible with the real-dataset pipeline.
"""

import math
import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torchdyn.datasets import generate_moons


def sample_quarter(
    size: int, inp: tuple[float, list[float]], sigma: float = 0.05, d: int = 2
) -> np.ndarray:
    """Sample points along a quarter-circle arc with Gaussian noise.

    Parameters
    ----------
    size : int
        Number of points to sample.
    inp : tuple[float, list[float]]
        A tuple of ``(angle, offset)``; ``angle`` scales the arc
        (multiples of pi/2) and ``offset`` is a 2D translation.
    sigma : float
        Standard deviation of the added Gaussian noise.
    d : int
        Output ambient dimension (2D is embedded in higher d with zeros).

    Returns
    -------
    np.ndarray
        Sampled points, shape ``(size, d)``.
    """
    i, o = inp  # angle, offset
    f = lambda x: (np.cos(x), np.sin(x))
    z = np.stack(f(np.pi / 2 * i + np.pi / 2 * np.random.uniform(size=size)), axis=1)
    if d > 2:
        x = np.zeros((z.shape[0], d))
        x[:, :2] = z
        z = x
    ep = sigma * np.random.normal(size=(size, d))
    offset = np.pad(np.array(o), (0, d - 2), 'constant')[None, :]
    return z + ep + offset


def sample_s(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample an 'S'-shaped trajectory made of six quarter-circle arcs.

    Parameters
    ----------
    size : int
        Total number of points (split evenly across six arcs).
    d : int
        Output ambient dimension.

    Returns
    -------
    x : np.ndarray
        Points, shape ``(size, d)``.
    y : np.ndarray
        Cell-type labels (all 0 here).
    t : np.ndarray
        Timepoint label per point (0-5).
    tree : np.ndarray
        Trivial 1x1 lineage adjacency matrix.
    """
    sizes = [size // 6] * 6
    angles = [0, 1, 2, 0, 3, 2]
    offsets = [[0, 1], [0, 1], [0, 1], [0, -1], [0, -1], [0, -1]]
    inputs = list(zip(angles, offsets))

    x = np.concatenate(
        [sample_quarter(sizes[i], inputs[i], d=d) for i in range(len(sizes))], axis=0
    )
    y = np.zeros(x.shape[0])
    t = np.repeat([0, 1, 2, 3, 4, 5], size // 6)
    tree = np.zeros((1, 1))

    return x, y, t, tree


def sample_s_imbalanced(
    size: int, d: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample an 'S'-shaped trajectory with a non-uniform per-arc size.

    Parameters
    ----------
    size : int
        Total number of points (split non-uniformly across six arcs).
    d : int
        Output ambient dimension.

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels, timepoints, and lineage matrix (see
        ``sample_s`` for the meaning of each).
    """
    sizes = [size // 20, size // 10, size, size // 20, size, size // 20]
    angles = [0, 1, 2, 0, 3, 2]
    offsets = [[0, 1], [0, 1], [0, 1], [0, -1], [0, -1], [0, -1]]
    inputs = list(zip(angles, offsets))

    x = np.concatenate(
        [sample_quarter(sizes[i], inputs[i], d=d) for i in range(len(sizes))], axis=0
    )
    y = np.zeros(x.shape[0])
    t = [[i] * sizes[i] for i in range(len(sizes))]
    t = np.concatenate(t, axis=0)
    tree = np.zeros((1, 1))

    return x, y, t, tree


def sample_omega(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample an 'omega'-shaped chain of six arcs with a linear lineage.

    Parameters
    ----------
    size : int
        Total number of points (split evenly across six segments).
    d : int
        Output ambient dimension.

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels (0-5), timepoints (0/1), and a 6x6
        chain adjacency matrix.
    """
    sizes = [size // 6] * 6
    sigma = 0.05

    x0 = np.zeros((sizes[0], d))
    x0[:, 0] = np.random.uniform(0, 2, size=sizes[0])
    x0[:, 1] = -1.0
    x0 += sigma * np.random.normal(size=(sizes[0], d))
    x1 = sample_quarter(sizes[1], (3, [0, 0]), d=d)
    x2 = sample_quarter(sizes[2], (0, [0, 0]), d=d)
    x3 = sample_quarter(sizes[3], (1, [0, 0]), d=d)
    x4 = sample_quarter(sizes[4], (2, [0, 0]), d=d)
    x5 = np.zeros((sizes[5], d))
    x5[:, 0] = np.random.uniform(-2, 0, size=sizes[5])
    x5[:, 1] = -1.0
    x5 += sigma * np.random.normal(size=(sizes[5], d))

    x = np.concatenate([x0, x1, x2, x3, x4, x5], axis=0)
    y = np.repeat([0, 1, 2, 3, 4, 5], size // 6)
    t = np.repeat([0, 0, 0, 1, 1, 1], size // 6)
    tree = np.zeros((6, 6))
    tree[0, 1] = 1
    tree[1, 2] = 1
    tree[2, 3] = 1
    tree[3, 4] = 1
    tree[4, 5] = 1
    return x, y, t, tree


def sample_wedge(size: int, d: int) -> np.ndarray:
    """Sample points uniformly from a quarter-unit disk (a "wedge").

    Parameters
    ----------
    size : int
        Number of points to sample.
    d : int
        Ambient dimension (only 2D sampling is implemented; the point is
        returned in the first two coordinates).

    Returns
    -------
    np.ndarray
        Points, shape ``(size, 2)``.
    """
    r = np.sqrt(np.random.uniform(0, 1, size=size))
    theta = np.pi / 2 * np.random.uniform(0, 1, size=size)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def sample_disk(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a full disk made of four reflected quarter-disk wedges.

    Parameters
    ----------
    size : int
        Total number of points (split evenly across four wedges).
    d : int
        Output ambient dimension (must be 2).

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels (0-3), timepoints (0/1), and a 4x4 chain
        adjacency matrix.
    """
    sizes = [size // 4] * 4
    sigma = 0.05
    assert d == 2

    x0 = sample_wedge(sizes[0], d)
    x1 = sample_wedge(sizes[1], d) * np.array([[-1, 1]])
    x2 = sample_wedge(sizes[2], d) * np.array([[-1, -1]])
    x3 = sample_wedge(sizes[3], d) * np.array([[1, -1]])

    x = np.concatenate([x0, x1, x2, x3], axis=0)
    y = np.repeat([0, 1, 2, 3], size // 4)
    t = np.repeat([0, 0, 1, 1], size // 4)
    tree = np.zeros((4, 4))
    tree[0, 1] = 1
    tree[1, 2] = 1
    tree[2, 3] = 1
    return x, y, t, tree


def sample_balance(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a balanced (symmetric) 4-cluster trajectory.

    Parameters
    ----------
    size : int
        Total number of points (split evenly across four clusters).
    d : int
        Output ambient dimension (must be 2).

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels, timepoints (0/1), and a 3x3 lineage
        matrix with one edge.
    """
    sizes = [size // 4] * 4
    sigma = 0.1
    assert d == 2

    x0 = sigma * np.random.normal(size=(sizes[0], d)) + np.array([[0, 0]])
    x1 = sigma * np.random.normal(size=(sizes[1], d)) + np.array([[0.5, 0]])
    x2 = sigma * np.random.normal(size=(sizes[2], d)) + np.array([[0, 0]])
    x3 = sigma * np.random.normal(size=(sizes[3], d)) + np.array([[0, -0.5]])

    x = np.concatenate([x0, x1, x2, x3], axis=0)
    y = np.repeat([0, 1, 0, 2], size // 4)
    t = np.repeat([0, 0, 1, 1], size // 4)
    tree = np.zeros((3, 3))
    tree[1, 2] = 1
    return x, y, t, tree


def sample_growth(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a 5-cluster "growth" trajectory with branching lineage.

    Parameters
    ----------
    size : int
        Total number of points (split evenly across five clusters).
    d : int
        Output ambient dimension (must be 2).

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels, timepoints, and a 4x4 lineage matrix.
    """
    sizes = [size // 5] * 5
    sigma = 0.1
    assert d == 2

    x0 = sigma * np.random.normal(size=(sizes[0], d)) + np.array([[0, 0]])
    x1 = sigma * np.random.normal(size=(sizes[1], d)) + np.array([[0.5, 0]])
    x2 = sigma * np.random.normal(size=(sizes[2], d)) + np.array([[0, 0]])
    x3 = sigma * np.random.normal(size=(sizes[3], d)) + np.array([[0, -0.5]])
    x4 = sigma * np.random.normal(size=(sizes[4], d)) + np.array([[0.5, -0.5]])

    x = np.concatenate([x0, x1, x2, x3, x4], axis=0)
    y = np.repeat([0, 1, 0, 2, 3], size // 5)
    t = np.repeat([0, 0, 1, 1, 1], size // 5)
    tree = np.zeros((4, 4))
    tree[1, 2] = 1
    tree[1, 3] = 1
    return x, y, t, tree


def sample_points(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample three fixed point clusters at increasing positions.

    Parameters
    ----------
    size : int
        Total number of points (split evenly across three clusters).
    d : int
        Output ambient dimension.

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels (all 0), timepoints (0/1/2), and a 1x1
        lineage matrix.
    """
    sizes = [size // 3] * 3

    x0 = np.zeros((sizes[0], d))
    x1 = np.zeros((sizes[1], d))
    x2 = np.zeros((sizes[2], d))
    x1[:, 0] = 1
    x1[:, 1] = 1
    x2[:, 0] = 2
    x = np.concatenate([x0, x1, x2], axis=0)

    y = np.zeros(x.shape[0])
    t = np.repeat([0, 1, 2], size // 3)
    tree = np.zeros((1, 1))

    return x, y, t, tree


def sample_path(
    size: int, d: int, sigma: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a branching "path" trajectory with a 5-cell lineage.

    Parameters
    ----------
    size : int
        Total number of points (split into six segments).
    d : int
        Ambient dimension.
    sigma : float
        Standard deviation of the added Gaussian noise.

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels, timepoints, and a 5x5 lineage matrix.
    """
    sizes = [size // 5, size // 5, size // 5, size // 100, size // 100, size // 100]

    x0 = sigma * np.random.normal(size=(sizes[0], d))
    x1 = sigma * np.random.normal(size=(sizes[1], d))
    x2 = sigma * np.random.normal(size=(sizes[2], d))
    x3 = sigma * np.random.normal(size=(sizes[3], d))
    x4 = sigma * np.random.normal(size=(sizes[4], d))
    x5 = sigma * np.random.normal(size=(sizes[4], d))

    x0[:, 0] -= 1.0
    x1[:, 1] -= 1.0
    x2[:, 0] += 1.0

    x3[:, 1] -= 1.0
    x5[:, 1] += 1.0
    x = np.concatenate([x0, x1, x2, x3, x4, x5], axis=0)

    y0 = np.zeros(x0.shape[0])
    y1 = np.zeros(x1.shape[0]) + 1
    y2 = np.zeros(x2.shape[0]) + 4
    y3 = np.zeros(x3.shape[0]) + 1
    y4 = np.zeros(x3.shape[0]) + 2
    y5 = np.zeros(x3.shape[0]) + 3
    y = np.concatenate([y0, y1, y2, y3, y4, y5], axis=0)

    t0 = np.zeros(x0.shape[0])
    t1 = np.zeros(x1.shape[0]) + 1
    t2 = np.zeros(x1.shape[0]) + 2
    t3 = np.random.choice(3, size=x3.shape[0])
    t4 = np.random.choice(3, size=x4.shape[0])
    t5 = np.random.choice(3, size=x5.shape[0])
    t = np.concatenate([t0, t1, t2, t3, t4, t5], axis=0)

    tree = np.zeros((5, 5))
    tree[0, 1] = 1
    tree[1, 4] = 1

    return x, y, t, tree


def sample_diamond(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a four-cluster "diamond" trajectory with crossing lineage.

    Parameters
    ----------
    size : int
        Total number of points (split evenly across four clusters).
    d : int
        Output ambient dimension.

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels (0-3), timepoints (0/1), and a 4x4
        lineage matrix.
    """

    sizes = [size // 4] * 4

    x0 = 0.1 * np.random.normal(size=(sizes[0], d))
    x1 = 0.1 * np.random.normal(size=(sizes[1], d))
    x2 = 0.1 * np.random.normal(size=(sizes[2], d))
    x3 = 0.1 * np.random.normal(size=(sizes[3], d))

    x0[:, 0] -= 1.0
    x1[:, 0] += 1.0
    x2[:, 1] -= 1.0
    x3[:, 1] += 1.0
    x = np.concatenate([x0, x1, x2, x3], axis=0)

    y0 = np.zeros(x0.shape[0])
    y1 = np.zeros(x1.shape[0]) + 1
    y2 = np.zeros(x2.shape[0]) + 2
    y3 = np.zeros(x3.shape[0]) + 3
    y = np.concatenate([y0, y1, y2, y3], axis=0)

    t0 = np.zeros(x0.shape[0])
    t1 = np.zeros(x1.shape[0])
    t2 = np.zeros(x2.shape[0]) + 1
    t3 = np.zeros(x3.shape[0]) + 1
    t = np.concatenate([t0, t1, t2, t3], axis=0)

    tree = np.zeros((4, 4))
    tree[0, 2] = 1
    tree[1, 3] = 1

    return x, y, t, tree


def sample_moons(size: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a two-cluster trajectory: ring points and torchdyn moons.

    Parameters
    ----------
    size : int
        Number of points per cluster.
    d : int
        Ambient dimension (only 2D is implemented).

    Returns
    -------
    x, y, t, tree : np.ndarray
        Points, cell-type labels (all 0), timepoints (0/1), and a 1x1
        lineage matrix.
    """

    assert d == 2, "Not implemented high dim moons yet"

    var = 0.1
    scale = 5
    m = torch.distributions.multivariate_normal.MultivariateNormal(
        torch.zeros(d), math.sqrt(var) * torch.eye(d)
    )
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
    ]
    centers = torch.tensor(centers) * scale
    noise = m.sample((size,))
    multi = torch.multinomial(torch.ones(8), size, replacement=True)
    data = []
    for i in range(size):
        data.append(centers[multi[i]] + noise[i])
    x0 = torch.stack(data).numpy()

    x1, _ = generate_moons(size, noise=0.2)
    x1 = x1.numpy()
    x1 = x1 * 3 - 1

    x = np.concatenate([x0, x1], axis=0)

    y = np.zeros(x.shape[0])
    t0 = np.zeros(x0.shape[0])
    t1 = np.zeros(x1.shape[0]) + 1
    t = np.concatenate([t0, t1], axis=0)

    tree = np.zeros((1, 1))

    return x, y, t, tree


def process_fake_data(size: int, d: int, sample_fn, normalize: bool = False) -> ad.AnnData:
    """Wrap a synthetic sample into an AnnData object for the pipeline.

    Parameters
    ----------
    size : int
        Number of points to sample.
    d : int
        Ambient dimension.
    sample_fn
        A ``sample_*`` function ``fn(size, d) -> (x, y, t, tree)``.
    normalize : bool
        If True, standardize ``X_pca`` by its per-dimension std (stored in
        ``uns['std']``).

    Returns
    -------
    anndata.AnnData
        The synthetic dataset, with ``obsm['X_pca']``, ``uns['tree']`` and
        the obs columns (``timepoint``, ``cell_type``, ...) set.
    """
    x, y, t, tree = sample_fn(size, d)

    adata = ad.AnnData(x)
    adata.obs['timepoint'] = t.astype(int)
    adata.obs['gene_target'] = pd.Categorical(["ctrl-inj"] * x.shape[0])
    adata.obs['cell_type'] = pd.Categorical(y)
    adata.obs['cell_type_one_hot'] = adata.obs['cell_type']

    # TODO: handle LabelEncoder for cell types properly

    adata.obsm['X_pca'] = adata.X
    adata.varm['PCs'] = np.zeros((d, d))
    adata.uns['std'] = np.ones((1, d))
    adata.uns['tree'] = tree

    if normalize:
        adata.uns['std'] = np.std(adata.obsm['X_pca'], axis=0, keepdims=True)
        adata.obsm['X_pca'] /= adata.uns['std']

    return adata
