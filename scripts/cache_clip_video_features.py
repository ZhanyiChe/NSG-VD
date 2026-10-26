"""Cache normalized OpenCLIP video embeddings from existing NSG-VD frames."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from hydra.utils import to_absolute_path
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm

from utils.clip_feature_cache import (
    ClipFeatureCache,
    ClipFeatureCacheSpec,
    fingerprint_preprocess,
    sha256_file,
)
from utils.clip_video_features import (
    clip_cache_model_key,
    encode_video_frames,
    get_image_embedding_dim,
    load_open_clip_model,
    resolve_video_frame_paths,
)
from utils.reference_selection import normalize_video_id


def _read_split_ids(split_file: Path):
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    video_ids = [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip().endswith(".mp4")
    ]
    if not video_ids:
        raise ValueError(f"Split file contains no .mp4 video IDs: {split_file}")
    normalized_ids = [normalize_video_id(video_id) for video_id in video_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError(f"Split file contains duplicate normalized video IDs: {split_file}")
    return video_ids


def _select_video_ids(split_file: Path, start_idx: int, limit):
    """Match VideoFramesDataset: filter MP4 IDs, then apply the requested slice."""

    if isinstance(start_idx, bool) or not isinstance(start_idx, int) or start_idx < 0:
        raise ValueError("clip_cache.start_idx must be a non-negative integer.")
    video_ids = _read_split_ids(split_file)
    if limit is None:
        selected = video_ids[start_idx:]
    else:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("clip_cache.limit must be a positive integer or null.")
        selected = video_ids[start_idx : start_idx + limit]
    if not selected:
        raise ValueError(
            f"No .mp4 video IDs selected from {split_file} with "
            f"start_idx={start_idx}, limit={limit}."
        )
    return selected


def _frame_directory_name(video_id: str) -> str:
    return Path(video_id.replace("\\", "/")).stem


@hydra.main(
    config_path="../configs/nsg-vd-224x224",
    config_name="clip_cache.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    logger.info("Resolved configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    split_file = Path(to_absolute_path(cfg.clip_cache.split_file))
    frame_root = Path(to_absolute_path(cfg.clip_cache.frame_root))
    base_cache_dir = Path(to_absolute_path(cfg.reference_selection.feature_cache_dir))
    start_idx = int(cfg.clip_cache.start_idx)
    limit = None if cfg.clip_cache.limit is None else int(cfg.clip_cache.limit)
    video_ids = _select_video_ids(split_file, start_idx, limit)

    frame_paths_by_id = {}
    invalid_frames = []
    for video_id in video_ids:
        frame_dir = frame_root / _frame_directory_name(video_id)
        try:
            frame_paths_by_id[video_id] = resolve_video_frame_paths(
                frame_dir, cfg.reference_selection.clip_num_frames
            )
        except (FileNotFoundError, ValueError) as error:
            invalid_frames.append((video_id, str(error)))

    if invalid_frames and cfg.clip_cache.strict_frames:
        preview = "\n".join(f"- {video_id}: {error}" for video_id, error in invalid_frames[:10])
        raise RuntimeError(
            f"{len(invalid_frames)} of {len(video_ids)} videos do not have the required "
            f"{cfg.reference_selection.clip_num_frames} cached frames. First failures:\n{preview}"
        )
    if invalid_frames:
        logger.warning(
            f"Skipping {len(invalid_frames)} videos without complete frame caches because "
            "clip_cache.strict_frames=false."
        )
    valid_video_ids = [video_id for video_id in video_ids if video_id in frame_paths_by_id]
    if not valid_video_ids:
        raise RuntimeError("No videos with complete frame caches are available for CLIP encoding.")

    requested_device = str(cfg.clip_cache.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"clip_cache.device={requested_device!r}, but CUDA is not available."
        )
    device = torch.device(requested_device)
    model, preprocess, open_clip_version = load_open_clip_model(
        cfg.reference_selection.clip_model_name,
        cfg.reference_selection.clip_pretrained,
        device,
    )
    embedding_dim = get_image_embedding_dim(model)

    cache_dir = (
        base_cache_dir
        / clip_cache_model_key(
            cfg.reference_selection.clip_model_name,
            cfg.reference_selection.clip_pretrained,
        )
        / cfg.clip_cache.label
        / cfg.clip_cache.generation_model
        / cfg.clip_cache.split
    )
    spec = ClipFeatureCacheSpec(
        model_name=cfg.reference_selection.clip_model_name,
        pretrained=cfg.reference_selection.clip_pretrained,
        open_clip_version=open_clip_version,
        embedding_dim=embedding_dim,
        num_frames=cfg.reference_selection.clip_num_frames,
        frame_source=cfg.clip_cache.frame_source,
        preprocess_fingerprint=fingerprint_preprocess(preprocess),
        split_sha256=sha256_file(split_file),
        dataset_name=cfg.clip_cache.dataset_name,
        label=cfg.clip_cache.label,
        generation_model=cfg.clip_cache.generation_model,
        split=cfg.clip_cache.split,
    )
    cache = ClipFeatureCache(cache_dir, spec)
    missing_ids = cache.missing(valid_video_ids)
    logger.info(
        f"CLIP cache: directory={cache_dir}, split_videos={len(video_ids)}, "
        f"valid_frames={len(valid_video_ids)}, cached={len(valid_video_ids) - len(missing_ids)}, "
        f"missing={len(missing_ids)}"
    )

    for video_id in tqdm(missing_ids, desc="Encoding CLIP video features", unit="video"):
        frame_paths = frame_paths_by_id[video_id]
        embedding = encode_video_frames(
            model,
            preprocess,
            frame_paths,
            device,
            cfg.reference_selection.clip_batch_size,
        )
        frame_dir = frame_paths[0].parent
        cache.put(
            video_id,
            embedding,
            relative_frame_dir=str(frame_dir.relative_to(frame_root.parent)),
            frame_names=[path.name for path in frame_paths],
        )

    remaining = cache.missing(valid_video_ids)
    if remaining:
        raise RuntimeError(f"CLIP cache is incomplete after extraction: {len(remaining)} missing.")
    logger.success(
        f"CLIP cache complete: {len(valid_video_ids)} videos, "
        f"embedding_dim={embedding_dim}, path={cache_dir}"
    )


if __name__ == "__main__":
    main()
