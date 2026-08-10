"""
General utility functions.
"""

import os
import numpy as np
import random
import torch


def reset_random_seeds(seed):
    # Let me know if I'm missing something here :) - MV
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    gen = torch.manual_seed(seed)
    return gen


def perf_option(config, name, default):
    """
    Read a flag from the config's `performance` block, tolerating its absence.
    """
    perf = config.get("performance", None) if config is not None else None
    if perf is None:
        return default
    value = perf.get(name, default)
    return default if value is None else value


def configure_performance(config=None):
    """
    Apply the `performance` block of the config to the torch backends.

    TODO(Max): see configs/config.yaml -- trim these comments and decide which
    flags are worth keeping configurable.

    Must be called *after* reset_random_seeds(), which unconditionally enables
    determinism and disables cuDNN autotuning; these settings override it.

    See configs/config.yaml for what each flag costs. The one that surprises
    people is fill_uninitialized_memory: torch turns it on as a side effect of
    use_deterministic_algorithms(True), it is only a debugging aid, and it is
    more expensive than any single kernel in this model's training step.
    """
    if not perf_option(config, "validate_distributions", False):
        torch.distributions.Distribution.set_default_validate_args(False)

    deterministic = perf_option(config, "deterministic", True)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic

    torch.backends.cudnn.benchmark = perf_option(config, "cudnn_benchmark", False)
    torch.utils.deterministic.fill_uninitialized_memory = perf_option(
        config, "fill_uninitialized_memory", False
    )


def numerical_stability_check(cov, device, epsilon=1e-6):
    """
    Check for numerical stability of covariance matrix.
    If not stable (i.e., not positive definite), add epsilon to diagonal.

    Parameters:
    cov (Tensor): The covariance matrix to check.
    epsilon (float, optional): The value to add to the diagonal if the matrix is not positive definite. Default is 1e-6.

    Returns:
    Tensor: The potentially adjusted covariance matrix.
    """
    num_added = 0
    if cov.dim() == 2:
        cov = (cov + cov.transpose(dim0=0, dim1=1)) / 2
    else:
        cov = (cov + cov.transpose(dim0=1, dim1=2)) / 2

    while True:
        try:
            # Attempt Cholesky decomposition; if it fails, the matrix is not positive definite
            torch.linalg.cholesky(cov)
            if num_added > 0.0001:
                print(
                    "Added {} to the diagonal of the covariance matrix.".format(
                        num_added
                    )
                )
            break
        except RuntimeError:
            # Add epsilon to the diagonal
            if cov.dim() == 2:
                cov = cov + epsilon * torch.eye(cov.size(0), device=device)
            else:
                cov = cov + epsilon * torch.eye(cov.size(1), device=device)
            num_added += epsilon
            epsilon *= 2
    return cov
