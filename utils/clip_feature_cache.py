"""Validated, incremental storage for video-level OpenCLIP embeddings."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch

from utils.reference_selection import normalize_video_id


CLIP_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ClipFeatureCacheSpec:
    """Configuration that uniquely identifies compatible cached embeddings."""

    model_name: str
    pretrained: str
    open_clip_version: str
    embedding_dim: int
    num_frames: int
    frame_source: str
    preprocess_fingerprint: str
    split_sha256: str
    dataset_name: str
    label: str
    generation_model: str
    split: str
    dtype: str = "float32"

    def __post_init__(self):
        required_strings = {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "open_clip_version": self.open_clip_version,
            "frame_source": self.frame_source,
            "preprocess_fingerprint": self.preprocess_fingerprint,
            "split_sha256": self.split_sha256,
            "dataset_name": self.dataset_name,
            "label": self.label,
            "generation_model": self.generation_model,
            "split": self.split,
            "dtype": self.dtype,
        }
        empty_fields = [name for name, value in required_strings.items() if not value]
        if empty_fields:
            raise ValueError(f"Cache spec fields must not be empty: {empty_fields}.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive.")
        if self.dtype != "float32":
            raise ValueError("Only float32 CLIP feature caches are currently supported.")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_preprocess(preprocess) -> str:
    """Fingerprint the concrete preprocessing transform representation."""

    return hashlib.sha256(repr(preprocess).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary_path = Path(file_handle.name)
            json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            temporary_path = Path(file_handle.name)
            np.save(file_handle, array, allow_pickle=False)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class ClipFeatureCache:
    """Store one normalized embedding and one provenance record per video."""

    def __init__(
        self,
        cache_dir: Path,
        spec: ClipFeatureCacheSpec,
        normalization_tolerance: float = 1e-4,
    ):
        self.cache_dir = Path(cache_dir)
        self.spec = spec
        self.normalization_tolerance = normalization_tolerance
        if normalization_tolerance <= 0 or not math.isfinite(normalization_tolerance):
            raise ValueError("normalization_tolerance must be a positive finite number.")

        self.embedding_dir = self.cache_dir / "embeddings"
        self.record_dir = self.cache_dir / "records"
        self.metadata_path = self.cache_dir / "metadata.json"
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_or_validate_metadata()

    @classmethod
    def open_existing(
        cls,
        cache_dir: Path,
        normalization_tolerance: float = 1e-4,
    ) -> "ClipFeatureCache":
        """Open a cache from its recorded spec without loading OpenCLIP.

        Evaluation only consumes previously generated embeddings.  Reading the
        authoritative cache metadata here avoids loading the CLIP model merely
        to reconstruct its embedding dimension, preprocessing fingerprint, or
        installed-package version.
        """

        cache_dir = Path(cache_dir)
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"CLIP cache metadata does not exist: {metadata_path}")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read CLIP cache metadata {metadata_path}: {error}") from error
        if metadata.get("format_version") != CLIP_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported CLIP cache format in {metadata_path}: "
                f"{metadata.get('format_version')!r}"
            )
        try:
            spec = ClipFeatureCacheSpec(**metadata["spec"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid CLIP cache spec in {metadata_path}: {error}") from error
        return cls(cache_dir, spec, normalization_tolerance=normalization_tolerance)

    def _initialize_or_validate_metadata(self) -> None:
        expected = {
            "format_version": CLIP_CACHE_FORMAT_VERSION,
            "spec": asdict(self.spec),
        }
        if not self.metadata_path.exists():
            _atomic_write_json(self.metadata_path, expected)
            return

        try:
            actual = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read CLIP cache metadata {self.metadata_path}: {error}") from error

        if actual != expected:
            actual_spec = actual.get("spec", {}) if isinstance(actual, dict) else {}
            expected_spec = expected["spec"]
            differing_keys = sorted(
                key
                for key in set(actual_spec) | set(expected_spec)
                if actual_spec.get(key) != expected_spec.get(key)
            )
            if actual.get("format_version") != CLIP_CACHE_FORMAT_VERSION:
                differing_keys.insert(0, "format_version")
            raise ValueError(
                f"CLIP cache metadata mismatch in {self.metadata_path}. "
                f"Differing fields: {differing_keys}. Use a separate cache directory."
            )

    @staticmethod
    def _cache_key(video_id: str) -> str:
        normalized_id = normalize_video_id(video_id)
        return hashlib.sha256(normalized_id.encode("utf-8")).hexdigest()

    def _paths(self, video_id: str) -> Tuple[Path, Path]:
        cache_key = self._cache_key(video_id)
        return (
            self.embedding_dir / f"{cache_key}.npy",
            self.record_dir / f"{cache_key}.json",
        )

    def _validate_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor.")
        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        if embedding.ndim != 1 or embedding.shape[0] != self.spec.embedding_dim:
            raise ValueError(
                f"Embedding must have shape [{self.spec.embedding_dim}], "
                f"got {tuple(embedding.shape)}."
            )
        if not embedding.is_floating_point():
            raise TypeError("embedding must use a floating-point dtype.")
        embedding = embedding.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(embedding).all():
            raise ValueError("embedding must contain only finite values.")
        norm = torch.linalg.vector_norm(embedding)
        if not torch.isclose(
            norm,
            torch.ones_like(norm),
            atol=self.normalization_tolerance,
            rtol=self.normalization_tolerance,
        ):
            raise ValueError(
                f"embedding must be L2-normalized; its norm is {norm.item():.6f}."
            )
        return embedding

    def contains(self, video_id: str, validate: bool = False) -> bool:
        embedding_path, record_path = self._paths(video_id)
        exists = embedding_path.is_file() and record_path.is_file()
        if exists and validate:
            self.load(video_id)
        return exists

    def missing(self, video_ids: Iterable[str]) -> Tuple[str, ...]:
        missing_ids = []
        for video_id in video_ids:
            if not self.contains(video_id):
                missing_ids.append(video_id)
                continue
            # Existing entries must be readable and compatible. Corruption or
            # stale records are errors rather than silent cache misses.
            self.load(video_id)
        return tuple(missing_ids)

    def put(
        self,
        video_id: str,
        embedding: torch.Tensor,
        relative_frame_dir: str,
        frame_names: Sequence[str],
    ) -> None:
        normalized_id = normalize_video_id(video_id)
        embedding = self._validate_embedding(embedding)
        if not relative_frame_dir:
            raise ValueError("relative_frame_dir must not be empty.")
        if len(frame_names) != self.spec.num_frames:
            raise ValueError(
                f"Expected {self.spec.num_frames} frame names, got {len(frame_names)}."
            )
        if len(set(frame_names)) != len(frame_names):
            raise ValueError("frame_names must be unique.")

        embedding_path, record_path = self._paths(video_id)
        _atomic_write_numpy(embedding_path, embedding.numpy())
        record = {
            "format_version": CLIP_CACHE_FORMAT_VERSION,
            "video_id": video_id,
            "normalized_id": normalized_id,
            "relative_frame_dir": relative_frame_dir,
            "frame_names": list(frame_names),
            "embedding_file": embedding_path.name,
        }
        _atomic_write_json(record_path, record)

    def load(self, video_id: str) -> torch.Tensor:
        normalized_id = normalize_video_id(video_id)
        embedding_path, record_path = self._paths(video_id)
        if not embedding_path.is_file() or not record_path.is_file():
            raise FileNotFoundError(f"No complete CLIP cache entry for video {video_id!r}.")

        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read CLIP cache record {record_path}: {error}") from error
        if record.get("format_version") != CLIP_CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported cache record format in {record_path}.")
        if record.get("normalized_id") != normalized_id:
            raise ValueError(
                f"Cache record ID mismatch for {video_id!r}: "
                f"{record.get('normalized_id')!r} != {normalized_id!r}."
            )
        if record.get("embedding_file") != embedding_path.name:
            raise ValueError(f"Cache record points to an unexpected embedding file: {record_path}.")
        if len(record.get("frame_names", [])) != self.spec.num_frames:
            raise ValueError(f"Cache record has the wrong frame count: {record_path}.")

        try:
            array = np.load(embedding_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read CLIP embedding {embedding_path}: {error}") from error
        if array.dtype != np.float32:
            raise ValueError(f"Cached embedding must be float32, got {array.dtype}.")
        return self._validate_embedding(torch.from_numpy(array.copy()))

    def load_many(self, video_ids: Sequence[str]) -> torch.Tensor:
        if not video_ids:
            raise ValueError("video_ids must not be empty.")
        return torch.stack([self.load(video_id) for video_id in video_ids], dim=0)
