"""MMD scoring helpers for query-adaptive reference sets.

The repository's original :func:`utils.mmd_utils.MMD_batch2` recomputes the
reference/reference kernel term for every query.  Adaptive reference selection
cannot share one such term across the whole test set, but it can precompute the
kernel matrix for the complete candidate pool once and index the appropriate
submatrix for each query.

The functions in this module deliberately preserve the biased MMD estimator
used by ``MMD_batch2``::

    mean(K_ref_ref) - 2 * mean(K_ref_query) + K_query_query

For a single query, ``K_query_query`` is exactly one for both kernel variants
implemented by the original code.
"""

from __future__ import annotations

from typing import Union

import torch

from utils.mmd_utils import MMD_batch2, Pdist2


Scalar = Union[float, int, torch.Tensor]


def _require_feature_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have a batch dimension and feature dimensions")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    return tensor


def _flatten_samples(tensor: torch.Tensor, name: str) -> torch.Tensor:
    tensor = _require_feature_tensor(tensor, name)
    return tensor.reshape(tensor.shape[0], -1)


def _validate_feature_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape[1] != right.shape[1]:
        raise ValueError(
            f"{left_name} and {right_name} feature dimensions differ: "
            f"{left.shape[1]} != {right.shape[1]}"
        )
    if left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must be on the same device")
    if left.dtype != right.dtype:
        raise ValueError(f"{left_name} and {right_name} must use the same dtype")


def _scalar_parameter(
    value: Scalar,
    name: str,
    reference: torch.Tensor,
    *,
    strictly_positive: bool = False,
) -> torch.Tensor:
    scalar = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if scalar.numel() != 1:
        raise ValueError(f"{name} must be a scalar")
    scalar = scalar.reshape(())
    if not torch.isfinite(scalar).item():
        raise ValueError(f"{name} must be finite")
    if strictly_positive and scalar.item() <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return scalar


def _validated_kernel_parameters(
    sigma: Scalar,
    sigma0: Scalar,
    epsilon: Scalar,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_tensor = _scalar_parameter(
        sigma, "sigma", reference, strictly_positive=True
    )
    sigma0_tensor = _scalar_parameter(
        sigma0, "sigma0", reference, strictly_positive=True
    )
    epsilon_tensor = _scalar_parameter(epsilon, "epsilon", reference)
    if not 0 <= epsilon_tensor.item() <= 1:
        raise ValueError("epsilon must be in the closed interval [0, 1]")
    return sigma_tensor, sigma0_tensor, epsilon_tensor


def composite_kernel_matrix(
    left_deep: torch.Tensor,
    right_deep: torch.Tensor,
    left_raw: torch.Tensor,
    right_raw: torch.Tensor,
    sigma: Scalar,
    sigma0: Scalar = 0.1,
    epsilon: Scalar = 1e-10,
    is_smooth: bool = True,
) -> torch.Tensor:
    """Compute the exact pairwise kernel used by ``MMD_batch2``.

    ``deep`` tensors must be two-dimensional. ``raw`` tensors may retain their
    original video dimensions; all dimensions after the batch dimension are
    flattened exactly as in the repository's evaluation code.
    """

    left_deep = _require_feature_tensor(left_deep, "left_deep")
    right_deep = _require_feature_tensor(right_deep, "right_deep")
    if left_deep.ndim != 2 or right_deep.ndim != 2:
        raise ValueError("left_deep and right_deep must be two-dimensional")
    _validate_feature_pair(left_deep, right_deep, "left_deep", "right_deep")

    sigma_tensor, sigma0_tensor, epsilon_tensor = _validated_kernel_parameters(
        sigma, sigma0, epsilon, left_deep
    )
    deep_distance = Pdist2(left_deep, right_deep)

    if is_smooth:
        left_raw_flat = _flatten_samples(left_raw, "left_raw")
        right_raw_flat = _flatten_samples(right_raw, "right_raw")
        if left_raw_flat.shape[0] != left_deep.shape[0]:
            raise ValueError("left_raw and left_deep batch sizes differ")
        if right_raw_flat.shape[0] != right_deep.shape[0]:
            raise ValueError("right_raw and right_deep batch sizes differ")
        _validate_feature_pair(
            left_raw_flat, right_raw_flat, "left_raw", "right_raw"
        )
        if left_raw_flat.device != left_deep.device:
            raise ValueError("raw and deep features must be on the same device")
        if left_raw_flat.dtype != left_deep.dtype:
            raise ValueError("raw and deep features must use the same dtype")

        raw_distance = Pdist2(left_raw_flat, right_raw_flat)
        kernel = (1 - epsilon_tensor) * torch.exp(
            -deep_distance / sigma0_tensor - raw_distance / sigma_tensor
        ) + epsilon_tensor * torch.exp(-raw_distance / sigma_tensor)
    else:
        kernel = torch.exp(-deep_distance / sigma0_tensor)

    if not torch.isfinite(kernel).all().item():
        raise ValueError("computed kernel contains NaN or infinity")
    return kernel


@torch.no_grad()
def precompute_candidate_kernel_matrix(
    candidate_deep: torch.Tensor,
    candidate_raw: torch.Tensor,
    sigma: Scalar,
    sigma0: Scalar = 0.1,
    epsilon: Scalar = 1e-10,
    is_smooth: bool = True,
    block_size: int = 16,
) -> torch.Tensor:
    """Precompute the candidate/candidate kernel matrix in row blocks."""

    candidate_deep = _require_feature_tensor(candidate_deep, "candidate_deep")
    if candidate_deep.ndim != 2:
        raise ValueError("candidate_deep must be two-dimensional")
    candidate_raw = _require_feature_tensor(candidate_raw, "candidate_raw")
    if candidate_raw.shape[0] != candidate_deep.shape[0]:
        raise ValueError("candidate_raw and candidate_deep batch sizes differ")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    blocks = []
    for start in range(0, candidate_deep.shape[0], block_size):
        stop = min(start + block_size, candidate_deep.shape[0])
        blocks.append(
            composite_kernel_matrix(
                candidate_deep[start:stop],
                candidate_deep,
                candidate_raw[start:stop],
                candidate_raw,
                sigma,
                sigma0,
                epsilon,
                is_smooth,
            )
        )
    return torch.cat(blocks, dim=0)


def _validate_scoring_inputs(
    candidate_deep: torch.Tensor,
    candidate_raw: torch.Tensor,
    query_deep: torch.Tensor,
    query_raw: torch.Tensor,
    selected_reference_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_deep = _require_feature_tensor(candidate_deep, "candidate_deep")
    query_deep = _require_feature_tensor(query_deep, "query_deep")
    if candidate_deep.ndim != 2 or query_deep.ndim != 2:
        raise ValueError("candidate_deep and query_deep must be two-dimensional")
    _validate_feature_pair(
        candidate_deep, query_deep, "candidate_deep", "query_deep"
    )

    candidate_raw = _flatten_samples(candidate_raw, "candidate_raw")
    query_raw = _flatten_samples(query_raw, "query_raw")
    if candidate_raw.shape[0] != candidate_deep.shape[0]:
        raise ValueError("candidate_raw and candidate_deep batch sizes differ")
    if query_raw.shape[0] != query_deep.shape[0]:
        raise ValueError("query_raw and query_deep batch sizes differ")
    _validate_feature_pair(candidate_raw, query_raw, "candidate_raw", "query_raw")
    if candidate_raw.device != candidate_deep.device:
        raise ValueError("raw and deep features must be on the same device")
    if candidate_raw.dtype != candidate_deep.dtype:
        raise ValueError("raw and deep features must use the same dtype")

    if not isinstance(selected_reference_indices, torch.Tensor):
        raise TypeError("selected_reference_indices must be a torch.Tensor")
    if selected_reference_indices.ndim != 2:
        raise ValueError("selected_reference_indices must have shape [queries, references]")
    if selected_reference_indices.shape[0] != query_deep.shape[0]:
        raise ValueError(
            "selected_reference_indices and query features have different batch sizes"
        )
    if selected_reference_indices.shape[1] == 0:
        raise ValueError("each query must select at least one reference")
    if selected_reference_indices.dtype == torch.bool or selected_reference_indices.dtype.is_floating_point:
        raise TypeError("selected_reference_indices must use an integer dtype")

    indices = selected_reference_indices.to(
        device=candidate_deep.device, dtype=torch.long
    )
    if (indices < 0).any().item() or (indices >= candidate_deep.shape[0]).any().item():
        raise IndexError("selected_reference_indices contains an out-of-range index")
    for query_index, row in enumerate(indices):
        if torch.unique(row).numel() != row.numel():
            raise ValueError(
                f"selected_reference_indices contains duplicates for query {query_index}"
            )

    return candidate_deep, candidate_raw, query_deep, query_raw, indices


@torch.no_grad()
def slow_adaptive_mmd_scores(
    candidate_deep: torch.Tensor,
    candidate_raw: torch.Tensor,
    query_deep: torch.Tensor,
    query_raw: torch.Tensor,
    selected_reference_indices: torch.Tensor,
    sigma: Scalar,
    sigma0: Scalar = 0.1,
    epsilon: Scalar = 1e-10,
    is_smooth: bool = True,
) -> torch.Tensor:
    """Reference implementation that invokes ``MMD_batch2`` per query."""

    (
        candidate_deep,
        candidate_raw,
        query_deep,
        query_raw,
        indices,
    ) = _validate_scoring_inputs(
        candidate_deep,
        candidate_raw,
        query_deep,
        query_raw,
        selected_reference_indices,
    )
    sigma_tensor, sigma0_tensor, epsilon_tensor = _validated_kernel_parameters(
        sigma, sigma0, epsilon, candidate_deep
    )

    scores = []
    for query_index, reference_indices in enumerate(indices):
        reference_deep = candidate_deep.index_select(0, reference_indices)
        reference_raw = candidate_raw.index_select(0, reference_indices)
        scores.append(
            MMD_batch2(
                torch.cat((reference_deep, query_deep[query_index : query_index + 1])),
                reference_deep.shape[0],
                torch.cat((reference_raw, query_raw[query_index : query_index + 1])),
                sigma_tensor,
                sigma0_tensor,
                epsilon_tensor,
                is_smooth=is_smooth,
            )[0]
        )
    return torch.stack(scores)


@torch.no_grad()
def fast_adaptive_mmd_scores(
    candidate_deep: torch.Tensor,
    candidate_raw: torch.Tensor,
    query_deep: torch.Tensor,
    query_raw: torch.Tensor,
    selected_reference_indices: torch.Tensor,
    candidate_kernel_matrix: torch.Tensor,
    sigma: Scalar,
    sigma0: Scalar = 0.1,
    epsilon: Scalar = 1e-10,
    is_smooth: bool = True,
) -> torch.Tensor:
    """Score queries by indexing a precomputed candidate kernel matrix."""

    (
        candidate_deep,
        candidate_raw,
        query_deep,
        query_raw,
        indices,
    ) = _validate_scoring_inputs(
        candidate_deep,
        candidate_raw,
        query_deep,
        query_raw,
        selected_reference_indices,
    )
    _validated_kernel_parameters(sigma, sigma0, epsilon, candidate_deep)

    if not isinstance(candidate_kernel_matrix, torch.Tensor):
        raise TypeError("candidate_kernel_matrix must be a torch.Tensor")
    expected_shape = (candidate_deep.shape[0], candidate_deep.shape[0])
    if tuple(candidate_kernel_matrix.shape) != expected_shape:
        raise ValueError(
            "candidate_kernel_matrix must have shape "
            f"{expected_shape}, got {tuple(candidate_kernel_matrix.shape)}"
        )
    if candidate_kernel_matrix.device != candidate_deep.device:
        raise ValueError("candidate_kernel_matrix and features must be on the same device")
    if candidate_kernel_matrix.dtype != candidate_deep.dtype:
        raise ValueError("candidate_kernel_matrix and features must use the same dtype")
    if not torch.isfinite(candidate_kernel_matrix).all().item():
        raise ValueError("candidate_kernel_matrix contains NaN or infinity")

    scores = []
    for query_index, reference_indices in enumerate(indices):
        reference_kernel = candidate_kernel_matrix.index_select(
            0, reference_indices
        ).index_select(1, reference_indices)
        cross_kernel = composite_kernel_matrix(
            candidate_deep.index_select(0, reference_indices),
            query_deep[query_index : query_index + 1],
            candidate_raw.index_select(0, reference_indices),
            query_raw[query_index : query_index + 1],
            sigma,
            sigma0,
            epsilon,
            is_smooth,
        )
        query_self_kernel = torch.ones(
            (), device=candidate_deep.device, dtype=candidate_deep.dtype
        )
        scores.append(reference_kernel.mean() - 2 * cross_kernel.mean() + query_self_kernel)
    return torch.stack(scores)
