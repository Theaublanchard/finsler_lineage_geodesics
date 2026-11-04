"""Reconstruction of the trained Finsler metric and directed-graph utilities.

This module is the *recovery* side of the pipeline: it loads the artifact
bundle written by ``scripts.run_model.save_finsler_artifacts`` (a
``finsler_metric.pt`` file) and exposes:

* :class:`FinslerEmbedder` — a small, self-contained module that reproduces
  the trained embedding ``(phi, psi, beta)`` and can compute the pointwise
  Finsler norm ``F(x, v)`` (the classifier + tree target the embedding was
  regressed against).
* :func:`build_directed_graph` — cell-level and cell-type-level asymmetric
  directed cost matrices.
* :func:`export_directed_graphs` — writes each graph to the three formats
  promised by the plan (``.npz``, NetworkX pickle, edge-list ``.csv``).
* :func:`load_finsler_metric` — convenience wrapper that returns a ready-to-use
  :class:`FinslerEmbedder` from a ``finsler_metric.pt`` path.
* :func:`verify_finsler_norm` — checks positive homogeneity and
  non-degeneracy on a set of sample tangent vectors.

Everything runs in ``torch.no_grad`` except the JVP call inside
``finsler_norm`` (which needs autograd to differentiate the classifier).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

from fingeo_supplement.models.modules import SimpleDenseNet, SimpleEmbedNet

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]]


def _as_float_tensor(x: ArrayLike) -> torch.Tensor:
    """Coerce an input to a ``torch.float32`` tensor on CPU (shape preserved)."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).float()
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


class FinslerEmbedder(nn.Module):
    """Load-only reconstruction of the trained Finsler embedding ``phi/psi/beta``.

    The training code (``models/embed_models.py::FinslerEmbedNetTrainBase``)
    stores a single network ``embed_net`` whose output of size
    ``latent_dim`` is split at ``d = len(beta) = latent_dim // 2``:

    * ``phi(x) = z[:, d:]``  (vector, symmetric Riemannian part)
    * ``psi(x) = z[:, :d] @ beta``  (scalar, asymmetric Finsler part)

    The directed cost between two points is::

        d_F(x, y) = ||phi(x) - phi(y)||_2 + ReLU(psi(x) - psi(y))

    which is *not* squared (the training code squares it only to feed a
    ``p=2`` optimal-transport solver).
    """

    def __init__(
        self,
        embed_net: nn.Module,
        beta: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        tree: Optional[torch.Tensor] = None,
        stoi: Optional[Mapping[int, str]] = None,
        itos: Optional[Mapping[int, str]] = None,
        num_classes: int = 0,
        classifier_net: Optional[nn.Module] = None,
        lamb: float = 1.0,
        temp: float = 1.0,
    ) -> None:
        """Build an embedder from raw tensors/modules.

        Parameters
        ----------
        embed_net
            A ``SimpleEmbedNet`` (or compatible) module with weights loaded.
        beta
            The learned Finsler vector (length ``latent_dim // 2``).
        mean, std
            Input normalization buffers from the training run.
        tree
            The lineage adjacency matrix ``A`` of shape ``(n_classes, n_classes)``.
            Required only for :meth:`finsler_norm`.
        stoi, itos
            Class index <-> name mappings (required only for
            :meth:`build_directed_graph` type-level aggregation).
        num_classes
            Number of real (non-dummy) classes.
        classifier_net
            The trained classifier network. Required for :meth:`finsler_norm`.
        lamb, temp
            Finsler hyperparameters (weight and softmax temperature).
        """
        super().__init__()
        self.embed_net = embed_net
        self.register_buffer("beta", beta.detach().cpu().clone())
        self.register_buffer("mean", mean.detach().cpu().clone())
        self.register_buffer("std", std.detach().cpu().clone())
        if tree is not None:
            # The classifier outputs ``num_classes + 1`` (real classes + the
            # dummy "other" class) and, during training, ``FinslerMixin`` pads
            # the lineage adjacency to match. The artifact stores the *raw*
            # tree (``num_classes x num_classes``), so pad it here too —
            # otherwise ``Jf(x) v @ u.T`` in ``finsler_norm`` has a shape
            # mismatch. Mirrors ``models.metric_models.FinslerMixin._pad_tree``.
            self.register_buffer("tree", self._pad_tree(tree, int(num_classes) + 1))
        self.stoi = dict(stoi) if stoi else {}
        self.itos = {int(k): v for k, v in (itos or {}).items()}
        self.num_classes = int(num_classes)
        self.classifier_net = classifier_net
        self.lamb = float(lamb)
        self.temp = float(temp)

    @staticmethod
    def _pad_tree(tree: torch.Tensor, target: int) -> torch.Tensor:
        """Zero-pad the lineage adjacency to ``target x target`` (a no-op if
        it is already at least ``target`` wide).  This mirrors
        ``FinslerMixin._pad_tree`` so the recovered ``finsler_norm`` sees the
        same padded matrix the training code used."""
        t = tree.detach().cpu().clone()
        if t.shape[-1] >= target:
            return t
        padded = torch.zeros((target, target), dtype=t.dtype)
        rows = min(target, t.shape[0])
        cols = min(target, t.shape[1])
        padded[:rows, :cols] = t[:rows, :cols]
        return padded

    # ------------------------------------------------------------------ input

    def normalize(self, x: ArrayLike) -> torch.Tensor:
        """Apply the stored ``(x - mean) / std`` normalization."""
        xt = _as_float_tensor(x)
        if self.mean is not None and self.mean.numel() > 0:
            xt = (xt - self.mean.to(xt.dtype)) / self.std.to(xt.dtype)
        return xt

    # ----------------------------------------------------------------- embed

    @torch.no_grad()
    def phi_psi(self, x: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(phi(x), psi(x))`` where ``phi`` is ``(B, latent_dim-d)``
        and ``psi`` is ``(B,)``."""
        xt = self.normalize(x)
        z = self.embed_net(xt)
        d = self.beta.shape[0]
        phi = z[:, d:]
        psi = z[:, :d] @ self.beta
        return phi, psi

    @torch.no_grad()
    def directed_cost(
        self,
        X1: ArrayLike,
        X2: Optional[ArrayLike] = None,
    ) -> torch.Tensor:
        """Asymmetric Finsler cost matrix ``M[i, j]`` for the ordered pairs
        ``(X1[i], X2[j])`` (or ``(X1[i], X1[j])`` if ``X2 is None``).

        ``M[i, j] = ||phi_i - phi_j|| + ReLU(psi_i - psi_j)``.

        Returns a tensor of shape ``(len(X1), len(X2 or X1))``.
        """
        X2 = X1 if X2 is None else X2
        phi1, psi1 = self.phi_psi(X1)
        phi2, psi2 = self.phi_psi(X2)
        M = torch.cdist(phi1, phi2) + F.relu(psi1.unsqueeze(1) - psi2.unsqueeze(0))
        return M

    # ------------------------------------------------------------- F(x, v)

    def finsler_norm(self, x: ArrayLike, v: ArrayLike) -> torch.Tensor:
        """Pointwise Finsler norm ``F(x, v)`` as used in the training target.

        Replicates ``models.metric_models.FinslerMixin.forward``:
        ``F(x, v) = ||v|| + lamb * (||v|| / ||v||) * sum_c f_c(x) * ReLU(
            Jf(x) v @ u_c)`` where ``f = softmax(classifier(x) / temp)`` and
        ``u = 1 - (pad(tree) + I)``.

        Requires ``self.classifier_net`` and ``self.tree`` to be set.
        """
        if self.classifier_net is None or self.tree is None:
            raise RuntimeError(
                "finsler_norm requires classifier_net and tree; "
                "they were not included in the artifact bundle."
            )
        xt = self.normalize(x)
        vt = _as_float_tensor(v)
        if vt.shape != xt.shape:
            raise ValueError(
                "finsler_norm expects x and v to share the same shape (one "
                f"tangent per base point); got x={tuple(xt.shape)} and "
                f"v={tuple(vt.shape)}. Batch vectors along dim 0, not a new axis."
            )

        def classifier_fn(xx: torch.Tensor) -> torch.Tensor:
            return torch.softmax(self.classifier_net(xx) / self.temp, dim=-1)

        f_x, Jf_x_v = jvp(classifier_fn, (xt,), (vt,))
        n = self.tree.shape[0]
        tree = self.tree.to(xt.device, dtype=xt.dtype)
        u = 1 - (tree + torch.eye(n, device=xt.device, dtype=xt.dtype))
        # NOTE: we intentionally do NOT slice u to the classifier's output dim;
        # the training code uses the padded matrix as-is.
        finsler_term = torch.sum(f_x * F.relu(Jf_x_v @ u.T), dim=-1)
        v_norm = torch.norm(vt, dim=-1)
        scale = v_norm / v_norm.clamp_min(1e-8)  # = 1 wherever v != 0
        riemann_term = v_norm
        return riemann_term + self.lamb * scale * F.relu(finsler_term)

    # --------------------------------------------------------- directed graph

    @torch.no_grad()
    def build_directed_graph(
        self,
        X: ArrayLike,
        cell_type: Optional[Union[Sequence[str], np.ndarray]] = None,
    ) -> dict:
        """Build the cell-level and (optionally) cell-type-level directed
        cost matrices.

        Parameters
        ----------
        X
            Feature matrix of shape ``(n_cells, pc_dim)``.
        cell_type
            A per-cell type label (string) of length ``n_cells``.  If given,
            the cell-type-level aggregate matrix is also returned.

        Returns
        -------
        dict
            ``{"cell_matrix": (n, n) tensor, "type_matrix": (|C|, |C|) tensor
            or None, "cell_types": list[str] or None}``
        """
        M = self.directed_cost(X)
        out: dict = {"cell_matrix": M, "type_matrix": None, "cell_types": None}
        if cell_type is None:
            return out

        labels = list(cell_type)
        # Determine type order: prefer the artifact's `itos` ordering if present,
        # else fall back to sorted unique labels.
        if self.itos:
            types = [self.itos[i] for i in range(len(self.itos))]
        else:
            types = sorted(set(labels))
        type_index = {t: i for i, t in enumerate(types)}
        C = len(types)
        acc = np.zeros((C, C), dtype=np.float64)
        cnt = np.zeros((C, C), dtype=np.int64)
        M_np = M.detach().cpu().numpy()
        for i in range(len(labels)):
            a = type_index[labels[i]]
            for j in range(len(labels)):
                b = type_index[labels[j]]
                acc[a, b] += M_np[i, j]
                cnt[a, b] += 1
        # Average (leave zeros where no cells were observed in a pair).
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        out["type_matrix"] = torch.from_numpy(T).float()
        out["cell_types"] = types
        return out


# ---------------------------------------------------------------------------
# Artifact bundle I/O
# ---------------------------------------------------------------------------


def _cfg_get(cfg: Optional[Mapping], *path: str, default=None):
    """Read a config value that may be stored flat (``"a.b"``) or nested
    (``{"a": {"b": ...}}``).

    The artifact's ``config`` dict is produced by ``OmegaConf.to_container``
    and is *nested* (e.g. ``cfg["classifier"]["hidden_dim"]``), but the
    builders historically read it with dot-notation (``cfg["classifier.hidden_dim"]``),
    which silently missed and fell back to defaults — building a net with the
    wrong hidden size / depth and then failing ``load_state_dict``. This helper
    tries the dotted key first, then walks the nested dict, returning
    ``default`` if neither is present.
    """
    if cfg is None:
        return default
    dotted = ".".join(path)
    if dotted in cfg and cfg[dotted] is not None:
        return cfg[dotted]
    node: object = cfg
    for part in path:
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return default
    return node


def _build_classifier_net(
    cfg: Mapping, num_classes: int, sd: Mapping[str, torch.Tensor]
) -> Optional[nn.Module]:
    """Rebuild the classifier ``SimpleDenseNet`` from the artifact bundle."""
    if sd is None:
        return None
    hidden = int(_cfg_get(cfg, "classifier", "hidden_dim", default=20))
    layers = int(_cfg_get(cfg, "classifier", "num_layers", default=1))
    spectral = bool(_cfg_get(cfg, "classifier", "spectral_norm", default=False))
    try:
        net = SimpleDenseNet(
            input_dim=int(_cfg_get(cfg, "pc_dim", default=100)),
            output_dim=int(num_classes) + 1,
            hidden_dims=[hidden] * layers,
            layer_norm=True,
            use_spectral_norm=spectral,
        )
    except Exception as _e:  # pragma: no cover - configuration edge case
        print(f"[finsler_extract] could not rebuild classifier_net: {_e!r}")
        return None
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"[finsler_extract] classifier load: missing={missing} " f"unexpected={unexpected}"
        )
    return net


def _build_embed_net(
    cfg: Mapping, latent_dim: int, sd: Mapping[str, torch.Tensor]
) -> SimpleEmbedNet:
    """Rebuild the ``SimpleEmbedNet`` from the artifact bundle."""
    hidden = int(_cfg_get(cfg, "hidden_dim", default=256))
    layers = int(_cfg_get(cfg, "num_layers", default=3))
    net = SimpleEmbedNet(
        input_dim=int(_cfg_get(cfg, "pc_dim", default=100)),
        output_dim=int(latent_dim),
        layer_norm=True,
        hidden_dims=[hidden] * layers,
        rescale=float(_cfg_get(cfg, "rescale", default=0.5)),
        skip=bool(_cfg_get(cfg, "skip", default=False)),
    )
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"embed_net state_dict is missing keys: {missing}")
    if unexpected:
        print(f"[finsler_extract] embed_net unexpected keys: {unexpected}")
    return net


def load_finsler_metric(path: Union[str, Path]) -> FinslerEmbedder:
    """Load a ``finsler_metric.pt`` bundle and return a ready-to-use
    :class:`FinslerEmbedder`.

    Parameters
    ----------
    path
        Path to the bundle (e.g. ``outputs/cns/finsler_metric.pt``).
    """
    state = torch.load(str(path), map_location="cpu", weights_only=False)

    cfg = state["config"] or {}
    latent_dim = int(cfg.get("latent_dim", 100))
    classifier = state.get("finsler", {}).get("use", False)
    classifier_net = (
        _build_classifier_net(cfg, int(state["num_classes"]), state.get("classifier_net"))
        if classifier and state.get("classifier_net") is not None
        else None
    )
    embed_net = _build_embed_net(cfg, latent_dim, state["embed_net"])
    fin = state.get("finsler", {}) or {}
    return FinslerEmbedder(
        embed_net=embed_net,
        beta=state["beta"],
        mean=state["mean"],
        std=state["std"],
        tree=state.get("tree"),
        stoi=state.get("stoi"),
        itos=state.get("itos"),
        num_classes=int(state.get("num_classes", 0)),
        classifier_net=classifier_net,
        lamb=float(fin.get("lamb", 1.0)),
        temp=float(fin.get("temp", 1.0)),
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def _export_one_graph(
    M: torch.Tensor,
    names: Sequence[str],
    out_dir: Path,
    base: str,
) -> dict[str, Path]:
    """Write one directed cost matrix to ``.npz`` + NetworkX + edge-list CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    M_np = M.detach().cpu().numpy()

    # 1. .npz
    npz_path = out_dir / f"{base}_matrix.npz"
    np.savez_compressed(
        npz_path,
        matrix=M_np,
        node_names=np.asarray(list(names), dtype=object),
    )

    # 2. NetworkX DiGraph pickle
    import networkx as nx  # local import so the rest of the module can load

    # without a hard networkx import (defensive).

    G = nx.DiGraph()
    G.add_nodes_from(range(len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            G.add_edge(i, j, weight=float(M_np[i, j]))
    nx_path = out_dir / f"{base}_digraph.nx.pkl"
    nx.to_pickle(G, nx_path)

    # 3. Edge-list CSV
    csv_path = out_dir / f"{base}_edges.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_idx", "target_idx", "source_name", "target_name", "cost"])
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j:
                    continue
                writer.writerow(
                    [
                        i,
                        j,
                        str(names[i]),
                        str(names[j]),
                        f"{float(M_np[i, j]):.6e}",
                    ]
                )
    return {"npz": npz_path, "nx": nx_path, "csv": csv_path}


def export_directed_graphs(
    embedder: FinslerEmbedder,
    X: ArrayLike,
    cell_type: Optional[Sequence[str]],
    out_dir: Union[str, Path],
    tissue: Optional[str] = None,
) -> dict:
    """Build and export both the cell-level and cell-type-level directed
    graphs for one tissue.

    Files written under ``out_dir/``::

        cell_matrix.npz, cell_digraph.nx.pkl, cell_edges.csv
        type_matrix.npz, type_digraph.nx.pkl, type_edges.csv

    Parameters
    ----------
    embedder
        A loaded :class:`FinslerEmbedder`.
    X
        Feature matrix of shape ``(n_cells, pc_dim)``.
    cell_type
        Per-cell type label (string) of length ``n_cells``.  If ``None``,
        only the cell-level graph is written (no type-level aggregation).
    out_dir
        Directory in which to write the artifact files.
    tissue
        Optional tissue tag; if given, ``out_dir`` is treated as a root and
        the files are written to ``out_dir/<tissue>/``.

    Returns
    -------
    dict
        ``{"cell": {paths}, "type": {paths} or None}``
    """
    out_root = Path(out_dir)
    if tissue:
        out_root = out_root / tissue
    result: dict = {"cell": None, "type": None}

    n = int(X.shape[0]) if isinstance(X, torch.Tensor) else len(X)
    cell_names = [f"cell_{i}" for i in range(n)]
    cell_graph = _export_one_graph(
        embedder.directed_cost(X), cell_names, out_root, base="cell"
    )
    result["cell"] = cell_graph

    if cell_type is not None:
        out = embedder.build_directed_graph(X, cell_type)
        T = out["type_matrix"]
        types = out["cell_types"]
        if T is not None and types:
            result["type"] = _export_one_graph(T, types, out_root, base="type")
    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_finsler_norm(
    embedder: FinslerEmbedder,
    X: ArrayLike,
    n_samples: int = 32,
    n_vectors: int = 16,
) -> dict:
    """Sanity-check the Finsler norm ``F(x, v)`` against its defining
    properties (positive homogeneity and non-degeneracy).

    Parameters
    ----------
    embedder
        A loaded :class:`FinslerEmbedder` (must have ``classifier_net`` and
        ``tree`` set).
    X
        Feature matrix to sample points and tangent vectors from.
    n_samples
        Number of base points to sample.
    n_vectors
        Number of random tangent vectors per base point.

    Returns
    -------
    dict
        ``{"positive_homogeneity_max_rel_err": float,
           "non_degeneracy_min_ratio": float,
           "passed": bool}``
    """
    X_np = embedder.normalize(X)
    idx = torch.randperm(X_np.shape[0])[:n_samples]
    xs = X_np[idx]

    # Random tangent vectors of comparable magnitude to a typical step.
    typical_scale = float((X_np.max() - X_np.min()).mean()) if X_np.numel() > 0 else 1.0
    vs = torch.randn(n_samples, n_vectors, X_np.shape[1]) * typical_scale

    # ``FinslerEmbedder.finsler_norm`` expects ``v`` to have the SAME shape as
    # ``x`` (one tangent per base point), so fold the ``n_vectors`` axis into the
    # batch dimension: (n_samples, n_vectors, dim) -> (n_samples*n_vectors, dim).
    xs_pairs = einops.repeat(xs, "n d -> (n k) d", k=n_vectors)
    vs_pairs = einops.rearrange(vs, "n k d -> (n k) d")

    hom_errs = []
    non_deg_ratios = []
    with torch.no_grad():
        for a in [0.5, 2.0]:
            f_v = embedder.finsler_norm(xs_pairs, vs_pairs)
            f_av = embedder.finsler_norm(xs_pairs, a * vs_pairs)
            rel = (f_av - a * f_v).abs() / f_v.clamp_min(1e-8)
            hom_errs.append(float(rel.max()))
        for a in [0.1, 1.0, 10.0]:
            f_v = embedder.finsler_norm(xs_pairs, a * vs_pairs)
            v_norm = (a * vs_pairs).norm(dim=-1)
            # Non-degeneracy: F(x, v) >= ||v|| (the Riemannian part)
            ratio = f_v / v_norm.clamp_min(1e-8)
            non_deg_ratios.append(float(ratio.min()))

    return {
        "positive_homogeneity_max_rel_err": max(hom_errs),
        "non_degeneracy_min_ratio": min(non_deg_ratios),
        "passed": (max(hom_errs) < 1e-3) and (min(non_deg_ratios) >= 1.0 - 1e-3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    """CLI entry point: rebuild a FinslerEmbedder from a saved bundle and
    print a summary of its shape / parameters."""
    import argparse

    ap = argparse.ArgumentParser(description="finsler_extract")
    ap.add_argument("artifact", help="path to finsler_metric.pt")
    ap.add_argument("--data", default=None, help="optional .h5ad path to load X_pca")
    ap.add_argument("--out", default=None, help="optional output dir for graphs")
    args = ap.parse_args()

    emb = load_finsler_metric(args.artifact)
    print(f"[finsler_extract] loaded {args.artifact}")
    print(f"  beta shape: {tuple(emb.beta.shape)}")
    print(f"  mean shape: {tuple(emb.mean.shape)}")
    print(f"  std:        {float(emb.std):.4e}")
    print(f"  tree:       {None if emb.tree is None else tuple(emb.tree.shape)}")
    print(f"  num_classes:{emb.num_classes}")
    print(f"  lamb:       {emb.lamb}")
    print(f"  temp:       {emb.temp}")

    if args.data:
        import anndata  # local import

        adata = anndata.read_h5ad(args.data)
        X = adata.obsm["X_pca"]
        ct = adata.obs["cell_type"].tolist() if "cell_type" in adata.obs else None
        print(f"[finsler_extract] loaded {args.data}: X {X.shape}")
        if args.out:
            tissue = args.artifact.parent.name if args.artifact.parent.name else None
            res = export_directed_graphs(emb, X, ct, args.out, tissue=tissue)
            for k, v in res.items():
                if v:
                    print(f"  {k}:")
                    for kind, p in v.items():
                        print(f"    {kind}: {p}")
        else:
            out = emb.build_directed_graph(X, ct)
            print(f"  cell_matrix: {tuple(out['cell_matrix'].shape)}")
            if out["type_matrix"] is not None:
                print(f"  type_matrix: {tuple(out['type_matrix'].shape)}")
    print("[finsler_extract] done")


if __name__ == "__main__":
    _main()
