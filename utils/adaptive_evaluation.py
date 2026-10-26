"""Orchestration helpers for query-adaptive reference evaluation."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from utils.adaptive_mmd import fast_adaptive_mmd_scores
from utils.reference_selection import (
    ReferenceSelection,
    select_clip_topn,
    select_random_topn,
    validate_reference_selection_mode,
)


@dataclass(frozen=True)
class AdaptiveReferencePool:
    """Aligned candidate IDs, NSG tensors, deep features, and kernel matrix."""

    video_ids: Tuple[str, ...]
    raw_features: torch.Tensor
    deep_features: torch.Tensor
    kernel_matrix: torch.Tensor
    clip_embeddings: Optional[torch.Tensor] = None

    def __post_init__(self):
        candidate_count = len(self.video_ids)
        if candidate_count == 0:
            raise ValueError("reference pool must contain at least one candidate")
        if len(set(self.video_ids)) != candidate_count:
            raise ValueError("reference pool video IDs must be unique")
        if self.raw_features.shape[0] != candidate_count:
            raise ValueError("raw feature count does not match reference IDs")
        if self.deep_features.ndim != 2 or self.deep_features.shape[0] != candidate_count:
            raise ValueError("deep feature count does not match reference IDs")
        if tuple(self.kernel_matrix.shape) != (candidate_count, candidate_count):
            raise ValueError("kernel matrix shape does not match reference IDs")
        if self.clip_embeddings is not None:
            if self.clip_embeddings.ndim != 2:
                raise ValueError("CLIP embeddings must be two-dimensional")
            if self.clip_embeddings.shape[0] != candidate_count:
                raise ValueError("CLIP embedding count does not match reference IDs")


@dataclass(frozen=True)
class AdaptiveBatchResult:
    scores: torch.Tensor
    selections: Tuple[Tuple[ReferenceSelection, ...], ...]


@torch.no_grad()
def select_and_score_adaptive_batch(
    reference_pool: AdaptiveReferencePool,
    query_ids: Sequence[str],
    query_raw_features: torch.Tensor,
    query_deep_features: torch.Tensor,
    mode: str,
    top_n: int,
    random_seed: int,
    sigma,
    sigma0,
    epsilon,
    is_smooth: bool,
    query_clip_embeddings: Optional[torch.Tensor] = None,
) -> AdaptiveBatchResult:
    """Select one reference set per query and compute its exact MMD score."""

    mode = validate_reference_selection_mode(mode)
    if mode == "original":
        raise ValueError("adaptive batch scoring only supports random_topn or clip_topn")
    if len(query_ids) != query_raw_features.shape[0]:
        raise ValueError("query ID count does not match raw query features")
    if len(query_ids) != query_deep_features.shape[0]:
        raise ValueError("query ID count does not match deep query features")

    if mode == "clip_topn":
        if reference_pool.clip_embeddings is None:
            raise ValueError("clip_topn requires candidate CLIP embeddings")
        if query_clip_embeddings is None:
            raise ValueError("clip_topn requires query CLIP embeddings")
        if query_clip_embeddings.ndim != 2:
            raise ValueError("query CLIP embeddings must be two-dimensional")
        if query_clip_embeddings.shape[0] != len(query_ids):
            raise ValueError("query CLIP embedding count does not match query IDs")

    batch_selections = []
    for query_index, query_id in enumerate(query_ids):
        if mode == "random_topn":
            selections = select_random_topn(
                reference_pool.video_ids,
                query_id,
                top_n,
                random_seed,
            )
        else:
            selections = select_clip_topn(
                reference_pool.video_ids,
                reference_pool.clip_embeddings,
                query_id,
                query_clip_embeddings[query_index],
                top_n,
            )
        batch_selections.append(selections)

    selected_indices = torch.tensor(
        [
            [selection.candidate_index for selection in selections]
            for selections in batch_selections
        ],
        dtype=torch.long,
        device=query_deep_features.device,
    )
    scores = fast_adaptive_mmd_scores(
        reference_pool.deep_features,
        reference_pool.raw_features,
        query_deep_features,
        query_raw_features,
        selected_indices,
        reference_pool.kernel_matrix,
        sigma,
        sigma0,
        epsilon,
        is_smooth=is_smooth,
    )
    return AdaptiveBatchResult(scores=scores, selections=tuple(batch_selections))
