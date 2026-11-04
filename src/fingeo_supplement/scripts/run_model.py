"""End-to-end model building and training pipeline for the Finsler embedding.

Wires together the classifier / metric / embedding (and optional flow) models
from a Hydra config, and drives the staged training loop (train-then-freeze
per phase) under wandb logging and a Lightning trainer.
"""

from __future__ import annotations

import numpy as np
import torch
import wandb
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import WeightedRandomSampler
import random

from torch.utils.data import Dataset, Sampler, DataLoader
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, Sampler, DataLoader
import pytorch_lightning as pl
from fingeo_supplement.models.fm_models import *
from fingeo_supplement.models.metric_models import *
from fingeo_supplement.models.embed_models import *
from fingeo_supplement.utils.preprocess import *
from fingeo_supplement.datasets.dataset import *
from fingeo_supplement.utils.frozen import *
from fingeo_supplement.utils.callback import *
from fingeo_supplement.models.modules import *
from fingeo_supplement.models.ema import *
from fingeo_supplement.models.classifier_models import *
from fingeo_supplement.datasets.process import *
from omegaconf import OmegaConf
from fingeo_supplement.eval.eval import *


def build_classifier(config):
    """Build the cell-type classifier (a plain MLP + training wrapper).

    Parameters
    ----------
    config
        The resolved Hydra config; reads ``pc_dim``, ``num_classes`` and the
        ``classifier`` sub-config.

    Returns
    -------
    ClassifierNetTrainBase
        The wrapped classifier model.
    """
    classifier_net = SimpleDenseNet(
        input_dim=config.pc_dim,
        output_dim=config.num_classes + 1,
        hidden_dims=[config.classifier.hidden_dim] * config.classifier.num_layers,
        layer_norm=True,
        use_spectral_norm=config.classifier.spectral_norm,
    )

    classifier_model = ClassifierNetTrainBase(classifier_net=classifier_net, config=config)
    return classifier_model


def build_metric(config, tree, classifier_model):
    """Build the Finsler metric model.

    Selects the CFM / MFM (optionally Finsler) implementation based on
    ``config.metric`` and ``config.finsler.use``.

    Parameters
    ----------
    config
        The resolved Hydra config.
    tree
        The lineage adjacency matrix (|C| x |C|) numpy array.
    classifier_model
        The (frozen) classifier used by the Finsler lineage penalty.

    Returns
    -------
    metric model
        One of the ``MetricNet*`` / ``Finsler*`` training-base models.
    """

    if config.metric == "cfm":
        if not config.finsler.use:
            metric_model = MetricNetCFM(config=config)
        else:
            metric_model = FinslerCFM(
                config=config,
                classifier_model=classifier_model,
                tree=torch.from_numpy(tree).float(),
                temp=config.finsler.temp,
                lamb=config.finsler.lamb,
            )

    elif config.metric == "mfm":
        # TODO: pass the dataloader?
        if not config.finsler.use:
            metric_model = MetricNetMFM(
                K=config.mfm.K,
                kappa=config.mfm.kappa,
                alpha=config.mfm.alpha,
                epsilon=config.mfm.epsilon,
                config=config,
            )
        else:
            metric_model = FinslerMFM(
                config=config,
                classifier_model=classifier_model,
                tree=torch.from_numpy(tree).float(),
                temp=config.finsler.temp,
                lamb=config.finsler.lamb,
                K=config.mfm.K,
                kappa=config.mfm.kappa,
                alpha=config.mfm.alpha,
                epsilon=config.mfm.epsilon,
            )
    elif config.metric == "gaga":
        # BUG: This branch is a no-op `pass`, so `metric_model` is never
        # assigned here and the `return metric_model` below raises a
        # NameError. Implement the GAGA metric model (e.g. FinslerGAGA) or
        # raise NotImplementedError for this metric.
        pass

    else:
        raise NotImplementedError(f"Metric model {config.metric} not implemented.")

    return metric_model


def build_embed(config, timepoints, metric_model):
    """Build the embedding model (embedding net + geodesic net + wrapper).

    Parameters
    ----------
    config
        The resolved Hydra config.
    timepoints
        The list of available timepoints (used to bound the global time
        interval of the geodesic net).
    metric_model
        The metric model used to cost geodesics.

    Returns
    -------
    embed model
        ``FinslerEmbedNetTrainBase`` or ``EmbedNetTrainBase``.
    """

    embed_net = SimpleEmbedNet(
        input_dim=config.pc_dim,
        output_dim=config.latent_dim,
        layer_norm=True,
        hidden_dims=[config.hidden_dim] * config.num_layers,
        rescale=config.rescale,
        skip=config.skip,
    )

    geo_net = SinNet(
        input_dim=2 * config.pc_dim,
        output_dim=config.pc_dim,
        num_freq=config.num_freq,
        layer_norm=True,
        hidden_dims=[config.hidden_dim] * config.num_layers,
        rescale=config.rescale,
    )

    t_global_min, t_global_max = min(timepoints), max(timepoints)

    if config.finsler.use:
        embed_model = FinslerEmbedNetTrainBase(
            metric_model=metric_model,
            geo_net=geo_net,
            embed_net=embed_net,
            config=config,
            t_global_min=t_global_min,
            t_global_max=t_global_max,
        )
    else:
        embed_model = EmbedNetTrainBase(
            metric_model=metric_model,
            geo_net=geo_net,
            embed_net=embed_net,
            config=config,
            t_global_min=t_global_min,
            t_global_max=t_global_max,
        )

    return embed_model


def build_flow(config, timepoints, embed_model):
    """Build the (optional) flow-matching model on top of the embedding.

    Parameters
    ----------
    config
        The resolved Hydra config.
    timepoints
        The list of available timepoints.
    embed_model
        The embedding model the flow is conditioned on.

    Returns
    -------
    FlowNetTrainBase
        The wrapped flow-matching model.
    """

    flow_net = SinNet(
        input_dim=config.pc_dim,
        output_dim=config.pc_dim,
        num_freq=config.num_freq,
        layer_norm=True,
        hidden_dims=[config.hidden_dim] * config.num_layers,
    )

    t_global_min, t_global_max = min(timepoints), max(timepoints)

    flow_model = FlowNetTrainBase(
        flow_net=flow_net,
        embed_model=embed_model,
        config=config,
        t_global_min=t_global_min,
        t_global_max=t_global_max,
    )

    return flow_model


def build_trainer(config, wandb_logger, phase):
    """Build a Lightning trainer for a given training phase.

    Selects the max-epochs and callbacks for the requested phase and
    configures the accelerator / logging / gradient-clip options.

    Parameters
    ----------
    config
        The resolved Hydra config.
    wandb_logger
        A wandb logger (or False) to attach to the trainer.
    phase
        One of ``"classifier"``, ``"metric"``, ``"embed"``, ``"flow"``.

    Returns
    -------
    pytorch_lightning.Trainer
        A configured trainer for the requested phase.
    """
    callbacks = []
    if phase == "classifier":
        max_epochs = config.classifier_max_epochs
    elif phase == "metric":
        max_epochs = config.metric_max_epochs
    elif phase == "embed":
        max_epochs = config.embed_max_epochs
        callbacks.append(BestModelCallback("train_loss_geo_embed"))
    elif phase == "flow":
        max_epochs = config.flow_max_epochs
    else:
        raise NotImplementedError(f"Phase {phase} not implemented.")
    trainer = pl.Trainer(
        accelerator="cpu" if config.force_cpu else "gpu",
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
        max_epochs=max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        enable_checkpointing=False,
        detect_anomaly=config.detect_anomaly,
        enable_progress_bar=True,
    )
    return trainer


def build_singleton_dataloader(config, adata):
    """Build the dataloader for the classifier / metric (singleton) phase.

    Extracts a singleton dataset from ``adata`` and returns a DataLoader,
    optionally using a class-balanced ``WeightedRandomSampler``.

    Parameters
    ----------
    config
        The resolved Hydra config.
    adata
        The preprocessed AnnData object.

    Returns
    -------
    DataLoader
        A DataLoader yielding ``(X, y)`` batches.
    """
    X, y = extract_singleton_dataset(adata)
    train_dataset = TensorDataset(X, y)
    batch_size = min(len(train_dataset), config.score_batch_size)
    if config.balance_classes:

        class_counts = torch.bincount(y) + 1
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[y]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, drop_last=True, shuffle=True
        )

    return train_dataloader


def build_paired_dataloader(config, adata):
    """Build the dataloader for the embedding (paired / OT) phase.

    Extracts a paired dataset from ``adata`` and wraps it in either a
    ``ShufflingOTDataset`` (when ``config.fast_ot``) or a plain
    ``ShufflingDataset``.

    Parameters
    ----------
    config
        The resolved Hydra config.
    adata
        The preprocessed AnnData object.

    Returns
    -------
    DataLoader
        A DataLoader yielding coupled ``(x0, x1)`` batches.
    """
    dataset = extract_paired_dataset(adata)

    if config.fast_ot:
        # TODO: hard-coded?  Actually validate if this works sensibly
        # TODO: we don't need to update during flow btw
        update_epoch_rate = 50
        train_dataset = ShufflingOTDataset(dataset, config.flow_batch_size, update_epoch_rate)
    else:
        train_dataset = ShufflingDataset(dataset, config.flow_batch_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.loader_batch_size, shuffle=True
    )
    return train_dataloader


def run_full_model(
    config,
    project,
    singleton_dataloader,
    paired_dataloader,
    timepoints,
    tree,
    wandb_logger=None,
):
    """Build and run the full staged training pipeline.

    Builds the classifier / metric / embed / flow models and trains them
    in sequence (each phase is trained then frozen before the next),
    optionally logging to wandb.

    Parameters
    ----------
    config
        The resolved Hydra config.
    project
        The wandb project name.
    singleton_dataloader
        Dataloader for the classifier / metric phase.
    paired_dataloader
        Dataloader for the embedding phase.
    timepoints
        The list of available timepoints.
    tree
        The lineage adjacency matrix (numpy array).
    wandb_logger
        An external wandb logger to reuse across phases, or None for
        per-phase runs (when ``config.use_wandb``).

    Returns
    -------
    classifier_model
        The trained (frozen) classifier.
    metric_model
        The trained (frozen) metric model.
    embed_model
        The trained (frozen) embedding model.
    flow_model
        The trained (frozen) flow model.
    """

    print(config)

    classifier_model = build_classifier(config)
    metric_model = build_metric(config, tree, classifier_model)
    embed_model = build_embed(config, timepoints, metric_model)
    flow_model = build_flow(config, timepoints, embed_model)

    if config.normalize:
        X = singleton_dataloader.dataset.tensors[0]
        mean = torch.mean(X, dim=0, keepdim=True)
        std = torch.max(torch.std(X, dim=0)) * np.sqrt(config.pc_dim)
        for model in [classifier_model, metric_model, embed_model, flow_model]:
            model.mean = mean
            model.std = std

    if not (config.finsler.use or config.force_train_classifier):
        config.classifier_max_epochs = 2

    if config.metric == "mfm":
        metric_model.train_dataloader = singleton_dataloader
    else:
        config.metric_max_epochs = 2

    if config.no_learning:
        assert config.metric == "cfm" and not config.finsler.use, "you need to learn a metric"
        return classifier_model, metric_model, embed_model, flow_model

    # phase_list =  ['classifier', 'metric', 'embed', 'flow']
    phase_list = ['classifier', 'metric', 'embed']

    for i, phase in enumerate(phase_list):

        print(f"Running phase {phase}:.......")

        # Determine the logger for this phase
        if wandb_logger is not None:
            # External logger (sweep path) — reuse the single run
            phase_logger = wandb_logger
        elif config.use_wandb:
            # Notebook path — separate wandb run per phase
            config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            phase_logger = WandbLogger(project=project, name=phase, log_model=True)
            wandb.init(config=config_dict, project=project, reinit=True)
        else:
            phase_logger = False

        ### train ###
        trainer = build_trainer(config, phase_logger, phase)
        model = {
            'classifier': classifier_model,
            'metric': metric_model,
            'embed': embed_model,
            'flow': flow_model,
        }[phase]

        train_dataloader = (
            singleton_dataloader if phase in ["classifier", "metric"] else paired_dataloader
        )

        # Log gradients to wandb for this phase
        if phase_logger:
            phase_logger.watch(model, log="gradients", log_freq=50)

        trainer.fit(model=model, train_dataloaders=train_dataloader)

        # Stop watching before the next phase (or cleanup)
        if phase_logger:
            wandb.unwatch(model)

        # Cleanup per-phase wandb run (notebook path only)
        if wandb_logger is None and config.use_wandb:
            wandb.finish()

        ### cleanup ###
        model.eval()
        freeze_params(model)

    return classifier_model, metric_model, embed_model, flow_model


def remove_all_forward_hooks(model):
    """Clear all registered forward hooks on every submodule of a model.

    Parameters
    ----------
    model
        A ``torch.nn.Module`` whose submodules' forward hooks should be
        removed.
    """
    for module in model.modules():
        module._forward_hooks.clear()


def tissue_slug(config) -> str:
    """Derive a short, filesystem-safe tissue tag from the config.

    Used to lay artifacts out under ``outputs/<tissue>/``.
    """
    slug = str(getattr(config, "dataset", "") or "").lower()
    if "cns" in slug:
        return "cns"
    if "arch" in slug or "pa" in slug:
        return "pa"
    return slug or "tissue"


def save_finsler_artifacts(config, adata, classifier_model, embed_model, out_root="outputs"):
    """Persist everything needed to reconstruct the trained Finsler metric.

    Writes a single ``outputs/<tissue>/finsler_metric.pt`` bundle containing the
    embedding net (which encodes both ``phi`` and ``psi``), the learned vector
    ``beta``, the input normalization buffers, the classifier net, the lineage
    adjacency matrix ``A`` (with ``stoi``/``itos``), the Finsler hyperparameters
    and the full resolved config.

    Parameters
    ----------
    config
        The resolved Hydra config.
    adata
        The preprocessed AnnData (source of the tree and cell types).
    classifier_model
        The trained classifier (used to evaluate ``F(x, v)``).
    embed_model
        The trained :class:`FinslerEmbedNetTrainBase` (source of ``embed_net``
        and ``beta``).
    out_root
        Root directory under which ``<tissue>/finsler_metric.pt`` is written.

    Returns
    -------
    Path
        The path the artifact bundle was written to.
    """
    out_dir = Path(out_root) / tissue_slug(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "finsler_metric.pt"

    try:
        config_dict = OmegaConf.to_container(config, resolve=True)
    except Exception:
        config_dict = OmegaConf.to_container(config, resolve=False)

    state = {
        "embed_net": embed_model.embed_net.state_dict(),
        "beta": embed_model.beta.detach().cpu(),
        "mean": embed_model.mean.detach().cpu(),
        "std": embed_model.std.detach().cpu(),
        "tree": torch.as_tensor(adata.uns["tree"], dtype=torch.float32).cpu(),
        "stoi": dict(adata.uns["stoi"]),
        "itos": {int(k): v for k, v in dict(adata.uns["itos"]).items()},
        "num_classes": int(adata.obs["cell_type"].nunique()),
        "finsler": {
            "use": bool(config.finsler.use),
            "lamb": float(config.finsler.lamb),
            "temp": float(config.finsler.temp),
        },
        "classifier_net": None,
        "config": config_dict,
    }
    if classifier_model is not None:
        try:
            state["classifier_net"] = classifier_model.classifier_net.state_dict()
        except Exception:
            pass

    torch.save(state, out_path)
    print(f"[save_finsler_artifacts] saved Finsler metric -> {out_path}")
    return out_path


def train(config, project, wandb_logger=None):
    """Run the full train + evaluate pipeline for a given config.

    Loads and preprocesses the data, builds the dataloaders, trains the
    four-stage model, then evaluates the embedding model's W1 distance at
    held-out and intermediate timepoints.

    Parameters
    ----------
    config
        The resolved Hydra config.
    project
        The wandb project name.
    wandb_logger
        An external wandb logger to reuse, or None.

    Returns
    -------
    classifier_model, metric_model, embed_model, flow_model
        The trained (frozen) models.
    w1_scores
        The W1 distances at intermediate timepoints (tensor).
    w1_val_scores
        The W1 distance at the held-out validation timepoint (tensor).
    """

    if config.seeded:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    adata = process_data(
        pc_dim=config.pc_dim,
        t0_index=config.t0_index,
        t1_index=config.t1_index,
        data=config.data,
        use_paga=config.use_paga,
        paga_threshold=config.paga_threshold,
        tissue=config.tissue,
    )
    config.num_classes = adata.obs['cell_type'].nunique()

    timepoints = sorted(adata.obs['timepoint'].unique().tolist())
    tree = adata.uns['tree']

    t0, t1 = timepoints[config.t0_index], timepoints[config.t1_index]

    adata = adata[(adata.obs['timepoint'] >= t0) & (adata.obs['timepoint'] <= t1)]
    adata_train = adata[adata.obs['timepoint'].isin([t0, t1])]

    singleton_dataloader = build_singleton_dataloader(config, adata_train)
    paired_dataloader = build_paired_dataloader(config, adata_train)

    classifier_model, metric_model, embed_model, flow_model = run_full_model(
        config=config,
        project=project,
        singleton_dataloader=singleton_dataloader,
        paired_dataloader=paired_dataloader,
        timepoints=timepoints,
        tree=tree,
        wandb_logger=wandb_logger,
    )

    remove_all_forward_hooks(classifier_model)
    remove_all_forward_hooks(metric_model)
    remove_all_forward_hooks(embed_model)
    remove_all_forward_hooks(flow_model)

    # Persist the trained Finsler metric so it can be recovered for downstream
    # analysis (directed-graph construction, F(x, v) evaluation, ...).
    if getattr(config, "finsler", None) is not None and config.finsler.use:
        try:
            save_finsler_artifacts(config, adata, classifier_model, embed_model)
        except Exception as _e:  # never let a save failure kill a finished run
            print(f"[save_finsler_artifacts] WARNING: failed to save: {_e!r}")

    t_val = timepoints[config.t0_index + 1]
    sample_val = (
        adata[adata.obs['timepoint'] == t_val].obs[config.sample_key].unique().tolist()[0]
    )
    subset_val = (adata.obs['timepoint'] != t_val) | (
        adata.obs[config.sample_key] == sample_val
    )
    adata_val = adata[subset_val]  # only evaluate sample_val at t_val
    adata = adata[adata.obs[config.sample_key] != sample_val]  # exclude sample_val

    w1_val_scores = []
    w1 = predict(
        embed_model,
        adata_val,
        t0,
        t_val,
        t1,
        num_traj=6000,
        library="pot",
        icfm=config.force_icfm,
    )
    w1_val_scores.append(w1)
    w1_val_scores = torch.tensor(w1_val_scores)

    w1_scores = []
    for index in range(config.t0_index + 1, config.t1_index):
        t = timepoints[index]
        w1 = predict(
            embed_model, adata, t0, t, t1, num_traj=6000, library="pot", icfm=config.force_icfm
        )
        w1_scores.append(w1)
    w1_scores = torch.tensor(w1_scores)

    return classifier_model, metric_model, embed_model, flow_model, w1_scores, w1_val_scores
