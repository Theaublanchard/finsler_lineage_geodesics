"""Optimal-transport sampling utilities (geomloss + KeOps).

Helpers for sampling from entropic OT couplings between two point clouds,
with a memory-efficient batched (KeOps) implementation to avoid
N x M allocations.
"""

import torch
from geomloss import SamplesLoss
from pykeops.torch import LazyTensor


# https://github.com/getkeops/keops/issues/372
def uniform_lazy_tensor_2d(dim0: int, dim1: int, device: torch.device) -> LazyTensor:
    """Initialize a 2D lazy tensor with uniform random values.

    Parameters
    ----------
    dim0 : int
        Number of rows (broadcast along axis 0).
    dim1 : int
        Number of columns (broadcast along axis 1).
    device : torch.device
        Device on which to allocate the underlying random values.

    Returns
    -------
    LazyTensor
        A ``LazyTensor`` of shape ``(dim0, dim1, 1)`` with values in ``[0, 1]``.
    """
    rand_x = LazyTensor(torch.rand((dim0, 1, 1), device=device))
    rand_y = LazyTensor(torch.rand((1, dim1, 1), device=device))

    rand_xy = (rand_x * 12.9898 + rand_y * 78.233).sin() * 43758.5453123
    rand_xy_floor = (rand_xy - 0.5).round()
    rand_xy_fract = rand_xy - rand_xy_floor
    rand_xy_clamp = rand_xy_fract.clamp(0, 1)

    return rand_xy_clamp


@torch.no_grad()
def sample_from_coupling_keops_batched(
    X: torch.Tensor,
    Y: torch.Tensor,
    f: torch.Tensor,
    g: torch.Tensor,
    p: int,
    eps: float,
    n_samples: int,
    i_batch: int = 1024,  # tune: how many source rows to handle at once
):
    """Batched (over source rows) KeOps-based sampler avoiding N x M allocations.

    Parameters
    ----------
    X : torch.Tensor (N, d)
        Source point cloud (on device).
    Y : torch.Tensor (M, d)
        Target point cloud (on device).
    f : torch.Tensor (N,)
        Source Sinkhorn dual potentials.
    g : torch.Tensor (M,)
        Target Sinkhorn dual potentials.
    p : int
        Cost exponent (usually 2).
    eps : float
        Temperature (``blur**p``).
    n_samples : int
        Total number of samples to draw.
    i_batch : int
        Number of distinct source rows processed per iteration.

    Returns
    -------
    i_idx : torch.Tensor (n_samples,)
        Sampled source indices (long).
    j_idx : torch.Tensor (n_samples,)
        Sampled target indices (long).
    """
    device = X.device
    N, M = X.shape[0], Y.shape[0]

    i_idx = torch.randint(0, N, (n_samples,), device=device)

    pairs_j = []

    for start in range(0, n_samples, i_batch):
        end = min(start + i_batch, n_samples)
        idx_batch = i_idx[torch.arange(start, end)]  # (B,)
        B = idx_batch.shape[0]

        Xb = X[idx_batch]  # (B,d)
        fb = f[idx_batch]  # (B,)

        X_i = LazyTensor(Xb[:, None, :])  # (B, 1, d)
        Y_j = LazyTensor(Y[None, :, :])  # (1, M, d)

        C = ((X_i - Y_j).abs() ** p).sum(-1) / p

        F = LazyTensor(fb[:, None, None])  # (B,1,1)
        G = LazyTensor(g[None, :, None])  # (1,M,1)

        logits = (F + G - C) / eps

        U = uniform_lazy_tensor_2d(B, M, device)
        gumbel = -(-U.log()).log()
        noisy = logits + gumbel

        # argmax over M for each row
        j_batch = noisy.argmax(dim=1)  # (B,1) long on device

        pairs_j.append(j_batch.squeeze(-1))  # store chosen target indices (B,)

    j_idx = torch.cat(pairs_j, dim=0)

    return i_idx, j_idx


def sample_from_coupling(
    X: torch.Tensor,
    Y: torch.Tensor,
    n_samples: int,
    blur: float = 0.05,
    p: int = 2,
    scaling: float = 0.9,
    indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample pairs from an entropic OT coupling between two point clouds.

    Computes Sinkhorn dual potentials with ``geomloss.SamplesLoss`` and
    then Gumbel-max samples the coupling in a memory-efficient batched
    (KeOps) fashion.

    Parameters
    ----------
    X : torch.Tensor (N, d)
        Source point cloud.
    Y : torch.Tensor (M, d)
        Target point cloud.
    n_samples : int
        Number of coupled pairs to draw.
    blur : float
        Sinkhorn blur (regularization) amount.
    p : int
        Cost exponent (default 2).
    scaling : float
        Sinkhorn scaling parameter.
    indices : bool
        If True, return the sampled ``(i, j)`` indices instead of points.

    Returns
    -------
    If ``indices`` is False:
    X_i
        Sampled source points, shape ``(n_samples, d)``.
    Y_j
        Sampled target points, shape ``(n_samples, d)``.

    If ``indices`` is True:
    i
        Sampled source indices, shape ``(n_samples,)``.
    j
        Sampled target indices, shape ``(n_samples,)``.
    """
    eps = blur ** (1 / p)

    loss_fn = SamplesLoss(
        "sinkhorn", p=p, blur=blur, scaling=scaling, debias=False, potentials=True
    )
    alpha, beta = loss_fn(X, Y)
    alpha, beta = alpha.flatten(), beta.flatten()

    i, j = sample_from_coupling_keops_batched(
        X, Y, alpha, beta, p=p, eps=eps, n_samples=n_samples
    )

    if indices:
        return i, j
    return X[i], Y[j]


# X = torch.concat([torch.rand(100000, 2) + torch.tensor([[0.0, 5.0]]), torch.rand(20000, 2) + torch.tensor([[0.0, -5.0]])], axis=0)
# Y = torch.rand(200000, 2) + torch.tensor([[10.0, 0.0]])

# X_, Y_ = sample_from_coupling(X, Y, 200000)

# Z = 0.5 * X_ + 0.5 * Y_

# plt.scatter(X[:,0], X[:,1])
# plt.scatter(Z[:,0], Z[:,1])
# plt.scatter(Y[:,0], Y[:,1])
# plt.show()
