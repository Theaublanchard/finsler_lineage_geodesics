"""Parameter-freezing helpers.

``frozen_params`` is a context manager that temporarily detaches a model
from the autograd graph; ``freeze_params`` sets every parameter's
``requires_grad`` to False after a phase completes.
"""

import torch
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def frozen_params(module: torch.nn.Module) -> Iterator[None]:
    """Context manager that temporarily freezes all of a module's parameters.

    Disables ``requires_grad`` on every parameter for the duration of the
    block, then restores the original flags on exit.

    Parameters
    ----------
    module : torch.nn.Module
        The module whose parameters to freeze/restore.

    Yields
    ------
    None
    """
    old_flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(False)
    yield
    for p, old in zip(module.parameters(), old_flags):
        p.requires_grad_(old)


def freeze_params(module: torch.nn.Module) -> None:
    """Permanently disable ``requires_grad`` on all of a module's parameters.

    Parameters
    ----------
    module : torch.nn.Module
        The module whose parameters to freeze.
    """
    for p in module.parameters():
        p.requires_grad_(False)
