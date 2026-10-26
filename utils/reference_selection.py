"""Deterministic reference selection utilities for adaptive MMD evaluation.

This module deliberately has no OpenCLIP dependency.  It operates on stable
video identifiers and already-normalized video embeddings so that selection
logic can be tested independently from feature extraction and model loading.
"""

from dataclasses import dataclass
from collections import Counter
import hashlib
import math
from pathlib import PurePosixPath
import random
from typing import Optional, Sequence, Tuple

import torch


REFERENCE_SELECTION_MODES = frozenset({"original", "random_topn", "clip_topn"})
_VIDEO_EXTENSIONS = frozenset({".avi", ".gif", ".mkv", ".mov", ".mp4", ".webm"})


@dataclass(frozen=True)
class ReferenceSelection:
    """One ranked reference selected for a query video."""

    reference_id: str
    candidate_index: int
    rank: int
    similarity: Optional[float] = None


def validate_reference_selection_mode(mode: str) -> str:
    """Validate and return a normalized reference-selection mode."""

    if not isinstance(mode, str):
        raise TypeError(f"Reference-selection mode must be a string, got {type(mode).__name__}.")
    normalized_mode = mode.strip().lower()
    if normalized_mode not in REFERENCE_SELECTION_MODES:
        supported = ", ".join(sorted(REFERENCE_SELECTION_MODES))
        raise ValueError(
            f"Unsupported reference-selection mode {mode!r}. Supported modes: {supported}."
        )
    return normalized_mode


def normalize_video_id(video_id: str) -> str:
    """Return a stable ID used for equality checks and deterministic seeding.

    Directory components and a common video extension are removed.  Matching
    is case-insensitive so an absolute path and its split-file ID identify the
    same video.
    """

    if not isinstance(video_id, str):
        raise TypeError(f"video_id must be a string, got {type(video_id).__name__}.")

    cleaned = video_id.strip().replace("\\", "/")
    if not cleaned:
        raise ValueError("video_id must not be empty.")

    name = PurePosixPath(cleaned).name
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in _VIDEO_EXTENSIONS:
        name = name[: -len(suffix)]
    if not name:
        raise ValueError(f"video_id {video_id!r} does not contain a usable identifier.")
    return name.casefold()


def _validate_candidates(candidate_ids: Sequence[str]) -> Tuple[str, ...]:
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a sequence of video IDs, not a string.")
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty.")

    normalized_ids = tuple(normalize_video_id(candidate_id) for candidate_id in candidate_ids)
    duplicate_ids = sorted(
        normalized_id
        for normalized_id, count in Counter(normalized_ids).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "candidate_ids contain duplicate normalized IDs: " + ", ".join(duplicate_ids)
        )
    return normalized_ids


def _eligible_candidate_indices(
    candidate_ids: Sequence[str], query_id: str, top_n: int
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError(f"top_n must be an integer, got {type(top_n).__name__}.")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}.")

    normalized_ids = _validate_candidates(candidate_ids)
    normalized_query_id = normalize_video_id(query_id)
    eligible_indices = tuple(
        index
        for index, normalized_id in enumerate(normalized_ids)
        if normalized_id != normalized_query_id
    )
    if len(eligible_indices) < top_n:
        raise ValueError(
            f"Cannot select top_n={top_n} references for query {query_id!r}: "
            f"only {len(eligible_indices)} of {len(candidate_ids)} candidates remain "
            "after excluding the query itself."
        )
    return normalized_ids, eligible_indices


def _query_seed(random_seed: int, normalized_query_id: str) -> int:
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise TypeError(
            f"random_seed must be an integer, got {type(random_seed).__name__}."
        )
    digest = hashlib.sha256(
        f"{random_seed}\0{normalized_query_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def select_random_topn(
    candidate_ids: Sequence[str],
    query_id: str,
    top_n: int,
    random_seed: int,
) -> Tuple[ReferenceSelection, ...]:
    """Select references without replacement using a per-query stable seed."""

    normalized_ids, eligible_indices = _eligible_candidate_indices(
        candidate_ids, query_id, top_n
    )
    rng = random.Random(_query_seed(random_seed, normalize_video_id(query_id)))
    selected_indices = rng.sample(eligible_indices, top_n)

    return tuple(
        ReferenceSelection(
            reference_id=candidate_ids[candidate_index],
            candidate_index=candidate_index,
            rank=rank,
        )
        for rank, candidate_index in enumerate(selected_indices, start=1)
    )


def _validate_normalized_embeddings(
    candidate_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
    candidate_count: int,
    normalization_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(candidate_embeddings, torch.Tensor):
        raise TypeError("candidate_embeddings must be a torch.Tensor.")
    if not isinstance(query_embedding, torch.Tensor):
        raise TypeError("query_embedding must be a torch.Tensor.")
    if candidate_embeddings.ndim != 2:
        raise ValueError(
            "candidate_embeddings must have shape [num_candidates, embedding_dim], "
            f"got {tuple(candidate_embeddings.shape)}."
        )
    if candidate_embeddings.shape[0] != candidate_count:
        raise ValueError(
            f"Received {candidate_count} candidate IDs but "
            f"{candidate_embeddings.shape[0]} candidate embeddings."
        )

    if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding = query_embedding.squeeze(0)
    if query_embedding.ndim != 1:
        raise ValueError(
            "query_embedding must have shape [embedding_dim] or [1, embedding_dim], "
            f"got {tuple(query_embedding.shape)}."
        )
    if query_embedding.shape[0] != candidate_embeddings.shape[1]:
        raise ValueError(
            "Candidate and query embedding dimensions differ: "
            f"{candidate_embeddings.shape[1]} != {query_embedding.shape[0]}."
        )
    if not candidate_embeddings.is_floating_point() or not query_embedding.is_floating_point():
        raise TypeError("Candidate and query embeddings must use a floating-point dtype.")
    if not torch.isfinite(candidate_embeddings).all() or not torch.isfinite(query_embedding).all():
        raise ValueError("Candidate and query embeddings must contain only finite values.")
    if normalization_tolerance <= 0 or not math.isfinite(normalization_tolerance):
        raise ValueError("normalization_tolerance must be a positive finite number.")

    candidate_norms = torch.linalg.vector_norm(candidate_embeddings, dim=1)
    query_norm = torch.linalg.vector_norm(query_embedding)
    ones = torch.ones_like(candidate_norms)
    if not torch.allclose(
        candidate_norms, ones, atol=normalization_tolerance, rtol=normalization_tolerance
    ):
        max_error = torch.max(torch.abs(candidate_norms - 1)).item()
        raise ValueError(
            "candidate_embeddings must be L2-normalized; "
            f"maximum norm error is {max_error:.3e}."
        )
    if not torch.isclose(
        query_norm,
        torch.ones_like(query_norm),
        atol=normalization_tolerance,
        rtol=normalization_tolerance,
    ):
        raise ValueError(
            "query_embedding must be L2-normalized; "
            f"its norm is {query_norm.item():.6f}."
        )

    return candidate_embeddings, query_embedding


def select_clip_topn(
    candidate_ids: Sequence[str],
    candidate_embeddings: torch.Tensor,
    query_id: str,
    query_embedding: torch.Tensor,
    top_n: int,
    normalization_tolerance: float = 1e-4,
) -> Tuple[ReferenceSelection, ...]:
    """Select the highest-cosine-similarity references with stable tie-breaking."""

    normalized_ids, eligible_indices = _eligible_candidate_indices(
        candidate_ids, query_id, top_n
    )
    candidate_embeddings, query_embedding = _validate_normalized_embeddings(
        candidate_embeddings,
        query_embedding,
        len(candidate_ids),
        normalization_tolerance,
    )

    similarities = torch.mv(candidate_embeddings, query_embedding).detach().cpu().tolist()
    ranked_indices = sorted(
        eligible_indices,
        key=lambda index: (-similarities[index], normalized_ids[index], index),
    )[:top_n]

    return tuple(
        ReferenceSelection(
            reference_id=candidate_ids[candidate_index],
            candidate_index=candidate_index,
            rank=rank,
            similarity=float(similarities[candidate_index]),
        )
        for rank, candidate_index in enumerate(ranked_indices, start=1)
    )
