"""Reusable neural-network building blocks.

Small MLP / residual / embedding primitives (``SinNet``, ``ResidualBlock``,
``SimpleEmbedNet``, ``SimpleDenseNet``, ``LinearNet``) shared across the
classifier, metric, and embedding models.
"""

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
from typing import List, Optional
import math
from torch.nn.utils import spectral_norm


class SinNet(nn.Module):
    """MLP with a sinusoidal time encoding prepended to the input.

    The time scalar ``t`` is expanded into ``2 * num_freq`` cos/sin
    features and concatenated with the spatial input before the dense
    network. Used as the geodesic deformation ``eta`` and the flow
    vector field ``v(x, t)``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_freq: int,
        activation: str = "selu",
        layer_norm: bool = False,
        hidden_dims: List[int] = [],
        rescale: float = 1.0,
    ) -> None:
        """Create the sinusoidal-time MLP.

        Parameters
        ----------
        input_dim : int
            Spatial input dimension (time features are appended).
        output_dim : int
            Output dimension.
        num_freq : int
            Number of sinusoidal frequencies for the time encoding.
        activation : str
            Activation name (see :data:`ACTIVATION_MAP`).
        layer_norm : bool
            Whether to insert LayerNorm after each linear layer.
        hidden_dims : List[int]
            Hidden layer sizes.
        rescale : float
            Global output scale factor.
        """
        super().__init__()

        self.num_freq = num_freq
        self.model = SimpleDenseNet(
            input_dim=input_dim + 2 * num_freq,
            output_dim=output_dim,
            activation=activation,
            layer_norm=layer_norm,
            hidden_dims=hidden_dims,
        )
        self.rescale = rescale

    def sinusoidal_time_encoding(
        self, t: torch.Tensor, max_period: float = 10000
    ) -> torch.Tensor:
        """Expand a scalar time into a cos/sin feature vector.

        Parameters
        ----------
        t : torch.Tensor (b,) or (b, 1)
            Time value(s).
        max_period : float
            Period of the highest frequency.

        Returns
        -------
        torch.Tensor (..., 2 * num_freq)
            Cos/sin time-encoding features.
        """
        device = next(self.parameters()).device

        if t.ndim == 1:
            t = t.unsqueeze(1)

        freqs = (
            torch.exp(
                -math.log(max_period)
                * torch.arange(start=0, end=self.num_freq, dtype=torch.float32)
                / self.num_freq
            )
            .unsqueeze(0)
            .to(device)
        )

        scaled_t = t * freqs
        return torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the sinusoidal-time MLP.

        Parameters
        ----------
        x : torch.Tensor (b, d_in)
            Spatial input points.
        t : torch.Tensor (b,) or (b, 1)
            Time value(s).

        Returns
        -------
        torch.Tensor (b, d_out)
            Dense-network output on ``[x, time-encoding]``, rescaled.
        """
        t = self.sinusoidal_time_encoding(t)
        return self.rescale * self.model(torch.cat([x, t], dim=1))


# https://github.com/kksniak/metric-flow-matching/blob/main/mfm/networks/mlp_base.py


class swish(nn.Module):
    """Swish / SiLU activation ``x * sigmoid(x)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the swish activation.

        Parameters
        ----------
        x : torch.Tensor (...)
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor (...)
            ``x * sigmoid(x)``.
        """
        return x * torch.sigmoid(x)


ACTIVATION_MAP = {
    "identity": nn.Identity,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
    "swish": swish,
}


class ResidualBlock(nn.Module):
    """Residual block: ``x + Linear-LN-Act-Linear(x)``."""

    def __init__(self, dim: int, activation: type) -> None:
        """Create a residual block.

        Parameters
        ----------
        dim : int
            Feature dimension.
        activation : type
            Activation class (instantiated once).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Parameters
        ----------
        x : torch.Tensor (b, dim)
            Input tensor.

        Returns
        -------
        torch.Tensor (b, dim)
            ``x + net(x)``.
        """
        return x + self.net(x)


class SimpleEmbedNet(nn.Module):
    """Dense embedding network with an optional linear skip projection.

    When ``skip=True``, adds a (near-)norm-preserving linear map of the
    input to the output, improving the conditioning of the embedding.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "selu",
        layer_norm: bool = False,
        hidden_dims: List[int] = [],
        rescale: float = 1.0,
        skip: bool = False,
    ) -> None:
        """Create the embedding network.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output (latent) dimension.
        activation : str
            Activation name.
        layer_norm : bool
            Insert LayerNorm after each linear layer.
        hidden_dims : List[int]
            Hidden layer sizes.
        rescale : float
            Global output scale.
        skip : bool
            If True, add a linear skip projection ``Q``.
        """
        super().__init__()
        self.model = SimpleDenseNet(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            layer_norm=layer_norm,
            hidden_dims=hidden_dims,
        )

        self.rescale = rescale
        self.skip = skip

        if skip:
            if input_dim <= output_dim:
                self.Q = torch.eye(output_dim, input_dim)
            else:
                # approximately norm-preserving projection
                self.Q = 1 / np.sqrt(output_dim) * torch.randn(output_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the embedding network.

        Parameters
        ----------
        x : torch.Tensor (b, d_in)
            Input points.

        Returns
        -------
        torch.Tensor (b, d_out)
            ``skip_embed + rescale * model(x)``.
        """
        skip_embed = 0
        if self.skip:
            device = next(self.parameters()).device
            skip_embed = x @ self.Q.T.to(device)
        return skip_embed + self.rescale * self.model(x)


class SimpleResNet(nn.Module):
    """MLP with residual blocks in the interior."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "selu",
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        """Create the residual MLP.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        activation : str
            Activation name.
        hidden_dim : int
            Hidden feature dimension.
        num_layers : int
            Total number of linear layers (including input/output).
        """
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(ACTIVATION_MAP[activation]())
        for i in range(1, num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, ACTIVATION_MAP[activation]))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the residual MLP.

        Parameters
        ----------
        x : torch.Tensor (b, d_in)
            Input points.

        Returns
        -------
        torch.Tensor (b, d_out)
            Output tensor.
        """
        return self.model(x)


class SimpleDenseNet(nn.Module):
    """Plain dense (fully-connected) MLP.

    Stacks linear layers (optionally spectral-normalized) with
    activation, optional LayerNorm, and optional dropout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "selu",
        layer_norm: bool = False,
        use_spectral_norm: bool = False,
        dropout: float = 0.0,
        hidden_dims: List[int] = [],
    ) -> None:
        """Create the dense MLP.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        activation : str
            Activation name.
        layer_norm : bool
            Insert LayerNorm after each linear layer.
        use_spectral_norm : bool
            Apply spectral normalization to every linear layer.
        dropout : float
            Dropout probability (0 to disable).
        hidden_dims : List[int]
            Hidden layer sizes.
        """
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 2):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        linear = nn.Linear(dims[-2], dims[-1])
        if use_spectral_norm:
            linear = spectral_norm(linear)
        layers.append(linear)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the dense MLP.

        Parameters
        ----------
        x : torch.Tensor (b, d_in)
            Input points.

        Returns
        -------
        torch.Tensor (b, d_out)
            Output tensor.
        """
        return self.model(x)


class LinearNet(nn.Module):
    """Stack of plain linear layers (no activation, no normalization)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]) -> None:
        """Create the linear network.

        Parameters
        ----------
        input_dim : int
            Input dimension.
        output_dim : int
            Output dimension.
        hidden_dims : List[int]
            Hidden layer sizes.
        """
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the linear network.

        Parameters
        ----------
        x : torch.Tensor (b, d_in)
            Input points.

        Returns
        -------
        torch.Tensor (b, d_out)
            Output tensor.
        """
        return self.model(x)
