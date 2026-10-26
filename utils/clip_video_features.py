"""OpenCLIP loading and video-level visual feature extraction helpers."""

from pathlib import Path
import re
from typing import Sequence, Tuple

from PIL import Image
import torch
import torch.nn.functional as F


_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def load_open_clip_model(model_name: str, pretrained: str, device: torch.device):
    """Load a frozen OpenCLIP image encoder and its matching preprocess."""

    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError(
            "open_clip_torch is required for CLIP reference retrieval. "
            "Install and record a compatible open_clip_torch version before running this script."
        ) from error

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    version = getattr(open_clip, "__version__", "unknown")
    return model, preprocess, version


def get_image_embedding_dim(model) -> int:
    """Return the declared OpenCLIP image embedding dimension."""

    visual = getattr(model, "visual", None)
    output_dim = getattr(visual, "output_dim", None)
    if not isinstance(output_dim, int) or output_dim <= 0:
        raise ValueError(
            "Cannot determine the OpenCLIP image embedding dimension from model.visual.output_dim."
        )
    return output_dim


def resolve_video_frame_paths(frame_dir: Path, num_frames: int) -> Tuple[Path, ...]:
    """Return a stable list of exactly ``num_frames`` cached RGB frame files."""

    frame_dir = Path(frame_dir)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")

    frame_paths = tuple(
        sorted(
            path
            for path in frame_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
    )
    if len(frame_paths) != num_frames:
        raise ValueError(
            f"Expected exactly {num_frames} image frames in {frame_dir}, "
            f"found {len(frame_paths)}."
        )
    return frame_paths


@torch.inference_mode()
def encode_video_frames(
    model,
    preprocess,
    frame_paths: Sequence[Path],
    device: torch.device,
    frame_batch_size: int,
) -> torch.Tensor:
    """Encode frames, normalize per frame, mean pool, then normalize per video."""

    if not frame_paths:
        raise ValueError("frame_paths must not be empty.")
    if frame_batch_size <= 0:
        raise ValueError("frame_batch_size must be positive.")

    processed_frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            processed = preprocess(image.convert("RGB"))
        if not isinstance(processed, torch.Tensor) or processed.ndim != 3:
            raise ValueError(
                "OpenCLIP preprocess must return a [channels, height, width] tensor."
            )
        processed_frames.append(processed)

    frame_embeddings = []
    for start in range(0, len(processed_frames), frame_batch_size):
        batch = torch.stack(
            processed_frames[start : start + frame_batch_size], dim=0
        ).to(device)
        embeddings = model.encode_image(batch)
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise ValueError("OpenCLIP encode_image must return a [batch, embedding_dim] tensor.")
        if not torch.isfinite(embeddings).all():
            raise ValueError("OpenCLIP produced non-finite frame embeddings.")
        frame_embeddings.append(F.normalize(embeddings.float(), dim=-1))

    frame_embeddings = torch.cat(frame_embeddings, dim=0)
    if frame_embeddings.shape[0] != len(frame_paths):
        raise RuntimeError("The number of CLIP frame embeddings does not match the input frames.")
    video_embedding = F.normalize(frame_embeddings.mean(dim=0), dim=0)
    if not torch.isfinite(video_embedding).all():
        raise ValueError("Video-level CLIP embedding is non-finite.")
    return video_embedding.detach().to(device="cpu", dtype=torch.float32)


def clip_cache_model_key(model_name: str, pretrained: str) -> str:
    """Return a filesystem-safe directory component for a model/weight pair."""

    value = f"{model_name}__{pretrained}"
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not key:
        raise ValueError("model_name and pretrained do not form a usable cache key.")
    return key
