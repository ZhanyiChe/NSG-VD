from utils.experiment_utils import set_seed
from data.utils import get_generation_models, get_label_from_generation_model
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator
from utils.train_utils import *
from utils.data_utils import *
from utils.adaptive_evaluation import (
    AdaptiveReferencePool,
    select_and_score_adaptive_batch,
)
from utils.adaptive_mmd import precompute_candidate_kernel_matrix
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
from utils.reference_selection import (
    normalize_video_id,
    validate_reference_selection_mode,
)
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import hydra
from tabulate import tabulate
import torch
import copy
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


_CLIP_REFERENCE_MODES = frozenset({"clip_topn", "clip_bottomn"})
_CLIP_FRAME_SOURCE = "existing_nsg_vd_8_frame_cache"


def _evaluation_mode(cfg):
    evaluation_cfg = cfg.get("evaluation", None)
    mode = "standard" if evaluation_cfg is None else evaluation_cfg.get("mode", "standard")
    if mode not in {"standard", "real_only"}:
        raise ValueError("evaluation.mode must be one of: standard, real_only")
    return mode


def _reference_selection_mode(cfg):
    selection_cfg = cfg.get("reference_selection", None)
    if selection_cfg is None:
        return "original"
    return validate_reference_selection_mode(selection_cfg.get("mode", "original"))


def _clip_cache_directory(cfg, label, generation_model, split):
    base_dir = Path(to_absolute_path(str(cfg.reference_selection.feature_cache_dir)))
    return (
        base_dir
        / clip_cache_model_key(
            cfg.reference_selection.clip_model_name,
            cfg.reference_selection.clip_pretrained,
        )
        / label
        / generation_model
        / split
    )


def _open_validated_clip_cache(cfg, label, generation_model, split):
    cache_dir = _clip_cache_directory(cfg, label, generation_model, split)
    cache = ClipFeatureCache.open_existing(cache_dir)
    expected = {
        "model_name": str(cfg.reference_selection.clip_model_name),
        "pretrained": str(cfg.reference_selection.clip_pretrained),
        "num_frames": int(cfg.reference_selection.clip_num_frames),
        "dataset_name": str(cfg.data.dataset_name),
        "label": label,
        "generation_model": generation_model,
        "split": split,
    }
    mismatches = {
        field: (getattr(cache.spec, field), expected_value)
        for field, expected_value in expected.items()
        if getattr(cache.spec, field) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"CLIP cache metadata does not match the evaluation request at {cache_dir}: "
            f"{mismatches}"
        )
    return cache


def _dataset_video_ids(dataloader):
    """Return the exact ordered video IDs exposed by a query dataloader."""

    video_ids = getattr(dataloader.dataset, "video_ids", None)
    if video_ids is None:
        raise ValueError(
            "CLIP query pre-caching requires dataloader.dataset.video_ids so cached "
            "embeddings remain aligned with NSG features."
        )
    raw_ids = tuple(str(video_id) for video_id in video_ids)
    normalized_ids = [normalize_video_id(video_id) for video_id in raw_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("Query dataset contains duplicate normalized video IDs.")
    return raw_ids


def _merge_clip_cache_requests(requests):
    """Merge repeated datasets while preserving first-seen query-ID order."""

    merged = {}
    for label, generation_model, split, video_ids in requests:
        key = (str(label), str(generation_model), str(split))
        if key not in merged:
            merged[key] = []
        known = {normalize_video_id(video_id) for video_id in merged[key]}
        for video_id in video_ids:
            raw_id = str(video_id)
            normalized_id = normalize_video_id(raw_id)
            if normalized_id not in known:
                merged[key].append(raw_id)
                known.add(normalized_id)
    return [(*key, tuple(video_ids)) for key, video_ids in merged.items()]


def _collect_query_clip_cache_requests(
        evaluation_mode,
        cfg,
        real_only_loader=None,
        test_dataloaders=None):
    """Describe only the query videos that the current evaluation will score."""

    if evaluation_mode == "real_only":
        return [
            (
                "real",
                str(cfg.real_only.real_model),
                str(cfg.real_only.split),
                _dataset_video_ids(real_only_loader),
            )
        ]

    requests = []
    for fake_model, loaders in test_dataloaders.items():
        requests.extend([
            (
                "real",
                str(cfg.data.test_real_model),
                "test",
                _dataset_video_ids(loaders["real"]),
            ),
            (
                "fake",
                str(fake_model),
                "test",
                _dataset_video_ids(loaders["fake"]),
            ),
        ])
    return _merge_clip_cache_requests(requests)


def _cache_query_clip_features(cfg, requests, device):
    """Incrementally cache missing CLIP query features before MMD evaluation.

    Existing entries are validated and skipped. OpenCLIP is loaded only when at
    least one requested query embedding is absent, and one model instance is
    reused for every missing query dataset in this evaluation run.
    """

    requests = list(requests)
    if not requests:
        return

    pending = []
    total_requested = 0
    total_cached = 0
    for label, generation_model, split, video_ids in requests:
        if not video_ids:
            raise ValueError(
                f"CLIP query set is empty: {label}/{generation_model}/{split}"
            )
        total_requested += len(video_ids)
        try:
            cache = _open_validated_clip_cache(
                cfg, label, generation_model, split
            )
        except FileNotFoundError:
            cache = None
            missing_ids = tuple(video_ids)
        else:
            missing_ids = cache.missing(video_ids)
            total_cached += len(video_ids) - len(missing_ids)
        if missing_ids:
            pending.append(
                (label, generation_model, split, tuple(video_ids), missing_ids)
            )

    logger.info(
        f"CLIP query pre-cache check: requested={total_requested}, "
        f"cached={total_cached}, missing={sum(len(item[4]) for item in pending)}"
    )
    if not pending:
        logger.info("All query CLIP features are already cached; skipping extraction")
        return

    if not bool(
            cfg.reference_selection.get("auto_cache_query_features", True)):
        first = pending[0]
        raise RuntimeError(
            "Query CLIP cache is incomplete and automatic extraction is disabled: "
            f"{sum(len(item[4]) for item in pending)} missing; "
            f"first dataset={first[0]}/{first[1]}/{first[2]}, "
            f"first IDs={list(first[4][:10])}"
        )

    clip_model = None
    try:
        logger.info(
            "Loading OpenCLIP to extract missing query features: "
            f"model={cfg.reference_selection.clip_model_name}, "
            f"pretrained={cfg.reference_selection.clip_pretrained}, device={device}"
        )
        clip_model, preprocess, open_clip_version = load_open_clip_model(
            str(cfg.reference_selection.clip_model_name),
            str(cfg.reference_selection.clip_pretrained),
            device,
        )
        embedding_dim = get_image_embedding_dim(clip_model)
        preprocess_fingerprint = fingerprint_preprocess(preprocess)
        data_root = Path(to_absolute_path(str(cfg.data.data_path)))
        num_frames = int(cfg.reference_selection.clip_num_frames)
        frame_batch_size = int(cfg.reference_selection.clip_batch_size)

        for label, generation_model, split, video_ids, _ in pending:
            split_file = (
                data_root
                / "split"
                / label
                / generation_model
                / f"{split}_ids.txt"
            )
            if not split_file.is_file():
                raise FileNotFoundError(
                    f"CLIP query split file does not exist: {split_file}"
                )
            frame_root = (
                data_root
                / "video_frames"
                / label
                / generation_model
                / split
            )
            spec = ClipFeatureCacheSpec(
                model_name=str(cfg.reference_selection.clip_model_name),
                pretrained=str(cfg.reference_selection.clip_pretrained),
                open_clip_version=open_clip_version,
                embedding_dim=embedding_dim,
                num_frames=num_frames,
                frame_source=_CLIP_FRAME_SOURCE,
                preprocess_fingerprint=preprocess_fingerprint,
                split_sha256=sha256_file(split_file),
                dataset_name=str(cfg.data.dataset_name),
                label=label,
                generation_model=generation_model,
                split=split,
            )
            cache = ClipFeatureCache(
                _clip_cache_directory(cfg, label, generation_model, split),
                spec,
            )
            missing_ids = cache.missing(video_ids)
            logger.info(
                "Caching query CLIP features: "
                f"dataset={label}/{generation_model}/{split}, "
                f"queries={len(video_ids)}, missing={len(missing_ids)}"
            )
            for video_id in tqdm(
                    missing_ids,
                    desc=f"Caching CLIP {generation_model}/{split}",
                    unit="video"):
                frame_paths = resolve_video_frame_paths(
                    frame_root / video_id,
                    num_frames,
                )
                embedding = encode_video_frames(
                    clip_model,
                    preprocess,
                    frame_paths,
                    device,
                    frame_batch_size,
                )
                cache.put(
                    video_id,
                    embedding,
                    relative_frame_dir=str(
                        frame_paths[0].parent.relative_to(frame_root.parent)
                    ),
                    frame_names=[path.name for path in frame_paths],
                )

            remaining_ids = cache.missing(video_ids)
            if remaining_ids:
                raise RuntimeError(
                    f"CLIP query cache remains incomplete for "
                    f"{label}/{generation_model}/{split}: "
                    f"{len(remaining_ids)} missing."
                )
            logger.success(
                "Query CLIP cache ready: "
                f"dataset={label}/{generation_model}/{split}, "
                f"queries={len(video_ids)}, path={cache.cache_dir}"
            )
    finally:
        if clip_model is not None:
            del clip_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def _build_adaptive_reference_pool(
        model,
        candidate_dataloader,
        device,
        kernel_block_size,
        clip_cache=None):
    raw_batches = []
    deep_batches = []
    candidate_ids = []
    for inputs, labels, batch_ids in tqdm(
            candidate_dataloader,
            desc="Building reference candidate pool",
            unit="batch"):
        if not torch.all(labels == 0):
            raise ValueError("reference candidate pool contains non-real labels")
        inputs = inputs.float()
        _, deep_features = model.net(inputs.to(device), out_feature=True)
        raw_batches.append(inputs)
        deep_batches.append(deep_features.detach().cpu())
        candidate_ids.extend(str(video_id) for video_id in batch_ids)

    if not candidate_ids:
        raise RuntimeError("reference candidate pool is empty")
    normalized_ids = [normalize_video_id(video_id) for video_id in candidate_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("reference candidate pool contains duplicate normalized video IDs")

    candidate_raw = torch.cat(raw_batches, dim=0).to(device)
    candidate_deep = torch.cat(deep_batches, dim=0).to(device)
    if candidate_raw.shape[0] != len(candidate_ids):
        raise RuntimeError("candidate NSG features and IDs are misaligned")

    candidate_kernel = precompute_candidate_kernel_matrix(
        candidate_deep,
        candidate_raw,
        model.sigma,
        model.sigma0_u,
        model.ep,
        is_smooth=model.is_smooth,
        block_size=kernel_block_size,
    )
    candidate_clip_embeddings = None
    if clip_cache is not None:
        missing_ids = clip_cache.missing(candidate_ids)
        if missing_ids:
            raise RuntimeError(
                f"CLIP candidate cache is incomplete: {len(missing_ids)} missing; "
                f"first IDs: {list(missing_ids[:10])}"
            )
        candidate_clip_embeddings = clip_cache.load_many(candidate_ids)

    return AdaptiveReferencePool(
        video_ids=tuple(candidate_ids),
        raw_features=candidate_raw,
        deep_features=candidate_deep,
        kernel_matrix=candidate_kernel,
        clip_embeddings=candidate_clip_embeddings,
    )


def _classification_metrics(real_scores, fake_scores, regression_model=None):
    raw_predict = torch.cat([real_scores, fake_scores], dim=0).cpu()
    if regression_model is None:
        predict = (raw_predict > 1).int()
    else:
        predict = torch.as_tensor(
            regression_model.predict(raw_predict.reshape(-1, 1).numpy())
        )
    labels = torch.cat(
        [torch.zeros(len(real_scores)), torch.ones(len(fake_scores))], dim=0
    )
    return {
        "precision": precision_score(labels.numpy(), predict.numpy()),
        "recall": recall_score(labels.numpy(), predict.numpy()),
        "f1": f1_score(labels.numpy(), predict.numpy()),
        "accuracy": accuracy_score(labels.numpy(), predict.numpy()),
        "auroc": roc_auc_score(labels.numpy(), raw_predict.numpy()),
    }


def _real_only_metrics(real_scores, regression_model=None):
    """Summarize false positives for an all-real query set."""

    real_scores = torch.as_tensor(real_scores).reshape(-1).cpu()
    if real_scores.numel() == 0:
        raise ValueError("real_scores must not be empty")
    if regression_model is None:
        predictions = (real_scores > 1).int()
    else:
        predictions = torch.as_tensor(
            regression_model.predict(real_scores.reshape(-1, 1).numpy())
        ).reshape(-1).int()
    if predictions.shape[0] != real_scores.shape[0]:
        raise RuntimeError("real-only predictions and scores are misaligned")

    false_positives = int((predictions == 1).sum().item())
    true_negatives = int((predictions == 0).sum().item())
    num_real = int(real_scores.numel())
    tnr = true_negatives / num_real
    fpr = false_positives / num_real
    return {
        "num_real": num_real,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "real_accuracy": tnr,
        "tnr": tnr,
        "fpr": fpr,
        "mean_mmd_score": float(real_scores.mean().item()),
        "predictions": predictions,
    }


@torch.no_grad()
def _score_adaptive_query_loader(
        model,
        query_dataloader,
        reference_pool,
        selection_mode,
        selection_cfg,
        query_label,
        generator_name,
        checkpoint_path,
        checkpoint_hash,
        clip_cache=None):
    device = next(model.parameters()).device
    scores = []
    query_rows = []
    selection_rows = []
    for inputs, labels, query_ids in tqdm(
            query_dataloader,
            desc=f"Scoring {generator_name} label={query_label}",
            leave=False):
        expected_label = float(query_label)
        if not torch.all(labels == expected_label):
            raise ValueError(
                f"query loader for label={query_label} contains unexpected labels"
            )
        query_ids = [str(query_id) for query_id in query_ids]
        query_raw = inputs.float().to(device)
        _, query_deep = model.net(query_raw, out_feature=True)
        query_clip_embeddings = None
        if selection_mode in _CLIP_REFERENCE_MODES:
            if clip_cache is None:
                raise ValueError(f"{selection_mode} requires a query CLIP cache")
            missing_ids = clip_cache.missing(query_ids)
            if missing_ids:
                raise RuntimeError(
                    f"CLIP query cache is incomplete for {generator_name}: "
                    f"{len(missing_ids)} missing; first IDs: {list(missing_ids[:10])}"
                )
            query_clip_embeddings = clip_cache.load_many(query_ids)

        batch_result = select_and_score_adaptive_batch(
            reference_pool,
            query_ids,
            query_raw,
            query_deep,
            selection_mode,
            int(selection_cfg.top_n),
            int(selection_cfg.random_seed),
            model.sigma,
            model.sigma0_u,
            model.ep,
            model.is_smooth,
            query_clip_embeddings=query_clip_embeddings,
        )
        batch_scores = batch_result.scores.detach().cpu()
        batch_mean_krr = batch_result.mean_krr.detach().cpu()
        batch_mean_krq = batch_result.mean_krq.detach().cpu()
        batch_kqq = batch_result.kqq.detach().cpu()
        reconstructed_scores = batch_mean_krr - 2 * batch_mean_krq + batch_kqq
        if not torch.allclose(
                batch_scores,
                reconstructed_scores,
                rtol=1e-5,
                atol=1e-6):
            max_error = torch.max(
                torch.abs(batch_scores - reconstructed_scores)
            ).item()
            raise RuntimeError(
                "Adaptive MMD component reconstruction failed: "
                f"maximum absolute error={max_error:.3e}"
            )
        batch_predictions = (batch_scores > 1).int()
        scores.append(batch_scores)

        for index, query_id in enumerate(query_ids):
            query_rows.append({
                "generator": generator_name,
                "query_id": query_id,
                "query_label": query_label,
                "mmd_score": float(batch_scores[index]),
                "mean_krr": float(batch_mean_krr[index]),
                "mean_krq": float(batch_mean_krq[index]),
                "kqq": float(batch_kqq[index]),
                "prediction": int(batch_predictions[index]),
                "threshold": 1.0,
                "reference_mode": selection_mode,
                "top_n": int(selection_cfg.top_n),
                "random_seed": int(selection_cfg.random_seed),
                "candidate_pool_size": len(reference_pool.video_ids),
                "checkpoint_path": checkpoint_path,
                "checkpoint_hash": checkpoint_hash,
            })
            for selection in batch_result.selections[index]:
                selection_rows.append({
                    "generator": generator_name,
                    "query_id": query_id,
                    "query_label": query_label,
                    "reference_id": selection.reference_id,
                    "candidate_index": selection.candidate_index,
                    "rank": selection.rank,
                    "clip_similarity": selection.similarity,
                    "reference_mode": selection_mode,
                    "top_n": int(selection_cfg.top_n),
                    "random_seed": int(selection_cfg.random_seed),
                })

    if not scores:
        raise RuntimeError(f"query loader is empty for {generator_name}, label={query_label}")
    return torch.cat(scores), query_rows, selection_rows


@torch.no_grad()
def _score_original_real_query_loader(
        model,
        query_dataloader,
        feature_ref,
        ref_data,
        dataset_name,
        checkpoint_path,
        checkpoint_hash,
        regression_model=None):
    """Score one ordered all-real loader with the legacy fixed reference set."""

    model.eval()
    device = next(model.parameters()).device
    scores = []
    query_ids = []
    for batch in tqdm(
            query_dataloader,
            desc=f"Scoring {dataset_name} real-only",
            leave=False):
        if len(batch) < 3:
            raise ValueError("real-only evaluation requires ID-bearing batches")
        inputs, labels, batch_ids = batch
        if not torch.all(labels == 0):
            raise ValueError("real-only query loader contains non-real labels")
        inputs = inputs.float().to(device)
        _, deep_features = model.net(inputs, out_feature=True)
        batch_scores = MMD_batch2(
            torch.cat([feature_ref, deep_features], dim=0),
            feature_ref.shape[0],
            torch.cat([ref_data, inputs], dim=0).view(
                ref_data.shape[0] + inputs.shape[0], -1
            ),
            model.sigma,
            model.sigma0_u,
            model.ep,
            is_smooth=model.is_smooth,
        ).cpu()
        scores.append(batch_scores)
        query_ids.extend(str(video_id) for video_id in batch_ids)

    if not scores:
        raise RuntimeError(f"real-only query loader is empty for {dataset_name}")
    scores = torch.cat(scores)
    if len(query_ids) != len(scores):
        raise RuntimeError("real-only video IDs and original MMD scores are misaligned")
    metrics = _real_only_metrics(scores, regression_model)
    rows = []
    for query_id, score, prediction in zip(
            query_ids, scores, metrics["predictions"]):
        rows.append({
            "generator": dataset_name,
            "query_id": query_id,
            "query_label": 0,
            "mmd_score": float(score),
            "prediction": int(prediction),
            "threshold": 1.0,
            "reference_mode": "original",
            "top_n": int(feature_ref.shape[0]),
            "random_seed": None,
            "candidate_pool_size": int(feature_ref.shape[0]),
            "checkpoint_path": checkpoint_path,
            "checkpoint_hash": checkpoint_hash,
        })
    return scores, rows, metrics

@hydra.main(
    config_path="configs/nsg-vd-224x224",
    config_name="test.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    # Setup Logging
    logger.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    evaluation_mode = _evaluation_mode(cfg)
    reference_mode = _reference_selection_mode(cfg)
    is_adaptive = reference_mode != "original"
    record_original_scores = (
        reference_mode == "original"
        and bool(cfg.reference_selection.get("save_original_query_scores", False))
    )
    logger.info(f"Evaluation mode: {evaluation_mode}")
    logger.info(f"Reference-selection mode: {reference_mode}")
    if record_original_scores:
        logger.info("Original per-query score logging is enabled")

    # ----------------------------------- Model ---------------------------------- #
    logger.info(f"Loading model: {cfg.model.name}")
    if cfg.model.name in ["Velocity_Single_TALL_MMD", "Score_Single_TALL_MMD"]:
        logger.info("Using single layer of TALL_MMD model")
        discriminator = SingleSwinBlockDiscriminator(num_features=cfg.model.feature_dim)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")
    
    model = deep_MMD(discriminator=discriminator, 
                        sigma=cfg.model.sigma, 
                        sigma0=cfg.model.sigma0, 
                        epsilon=cfg.model.epsilon, 
                        img_size=cfg.model.img_size, 
                        is_yy_zero=cfg.model.is_yy_zero,
                        is_smooth=cfg.model.is_smooth)
    model.load_state_dict(torch.load(cfg.ckpt_path, weights_only=True))
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model.net = nn.DataParallel(model.net, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
    model = model.to(device)
    model.eval()
    
    # ----------------------------------- Data ----------------------------------- #
    generation_models = get_generation_models(cfg.data.dataset_name)
    if is_adaptive:
        if cfg.data.get("ref_models", None) is not None:
            raise ValueError(
                "Adaptive reference selection currently requires one explicit "
                "MSR-VTT candidate pool; multi-source ref_models is unsupported."
            )
        candidate_dataloader = get_reference_candidate_dataloader(
            cfg.data,
            cfg.data.ref_model,
            mode=cfg.data.ref_mode,
            load_len=int(cfg.reference_selection.candidate_pool_size),
            resolution_size=cfg.data.resolution_size,
        )
    elif cfg.data.get("ref_models", None) is not None:
        ref_dataloaders = get_ref_dataloaders(
            cfg.data,
            cfg.data.ref_models,
            mode="test",
        )
    else:
        ref_dataloader = get_ref_dataloader(cfg.data, 
                                        cfg.data.ref_model,
                                        mode="val",
                                        resolution_size=cfg.data.resolution_size)
    if evaluation_mode == "real_only":
        real_model = str(cfg.real_only.real_model)
        real_split = str(cfg.real_only.split)
        if get_label_from_generation_model(real_model) != "real":
            raise ValueError(
                f"real_only.real_model must identify a real dataset, got {real_model}"
            )
        real_only_loader = get_real_score_dataloader(
            cfg.data,
            real_model,
            real_split,
            load_len=int(cfg.real_only.load_len),
            start_idx=int(cfg.real_only.start_idx),
            batch_size=int(cfg.real_only.batch_size),
            resolution_size=cfg.data.resolution_size,
            return_video_id=True,
        )
        logger.info(
            "Real-only query set: "
            f"model={real_model}, split={real_split}, "
            f"start_idx={cfg.real_only.start_idx}, requested={cfg.real_only.load_len}, "
            f"effective={len(real_only_loader.dataset)}, batch_size={cfg.real_only.batch_size}"
        )
    else:
        test_dataloaders = {}
        for fake_model in generation_models["fake"]["test"]:
            for real_model in get_generation_models(cfg.data.dataset_name)["real"]["test"]:
                real_model = cfg.data.test_real_model
                if fake_model == "Sora":
                    load_len = 56
                else:
                    load_len = cfg.data.test_load_len
                test_datasets = get_score_datasets(cfg.data,
                                                  "test",
                                                  load_len=load_len,
                                                  real_model=real_model,
                                                  generation_model=fake_model,
                                                  filter=False,
                                                  resolution_size=cfg.data.resolution_size,
                                                  return_video_id=(is_adaptive or record_original_scores))
                test_loader = get_data_loaders_for_mmd(
                    cfg.data,
                    test_datasets,
                    batch_size=cfg.data.val_batch_size,
                    shuffle=not is_adaptive,
                )
                test_dataloaders[f"{fake_model}"] = test_loader

    if reference_mode in _CLIP_REFERENCE_MODES:
        query_cache_requests = _collect_query_clip_cache_requests(
            evaluation_mode,
            cfg,
            real_only_loader=(
                real_only_loader if evaluation_mode == "real_only" else None
            ),
            test_dataloaders=(
                test_dataloaders if evaluation_mode == "standard" else None
            ),
        )
        _cache_query_clip_features(cfg, query_cache_requests, device)

    checkpoint_path = str(Path(to_absolute_path(str(cfg.ckpt_path))).resolve())
    checkpoint_hash = sha256_file(Path(checkpoint_path))
    logger.info(f"Checkpoint: path={checkpoint_path}, sha256={checkpoint_hash}")
    logger.info(
        f"Kernel parameters: epsilon={model.ep.item():.10g}, "
        f"sigma={model.sigma.item():.10g}, sigma0={model.sigma0_u.item():.10g}, "
        f"is_smooth={model.is_smooth}, threshold=1"
    )

    if is_adaptive:
        candidate_clip_cache = None
        if reference_mode in _CLIP_REFERENCE_MODES:
            candidate_clip_cache = _open_validated_clip_cache(
                cfg,
                "real",
                cfg.data.ref_model,
                cfg.data.ref_mode,
            )
            logger.info(
                "Using cached OpenCLIP embeddings: "
                f"model={candidate_clip_cache.spec.model_name}, "
                f"pretrained={candidate_clip_cache.spec.pretrained}, "
                f"open_clip={candidate_clip_cache.spec.open_clip_version}"
            )
        reference_pool = _build_adaptive_reference_pool(
            model,
            candidate_dataloader,
            device,
            int(cfg.reference_selection.kernel_block_size),
            clip_cache=candidate_clip_cache,
        )
        logger.info(
            f"Reference candidate pool: requested={cfg.reference_selection.candidate_pool_size}, "
            f"effective={len(reference_pool.video_ids)}, top_n={cfg.reference_selection.top_n}, "
            f"split={cfg.data.ref_mode}"
        )
    elif cfg.data.get("ref_models", None) is not None and cfg.data.get("ref_ratio", None) is not None:
        feature_ref, ref_data = get_ref_features_multi_source(
            model,
            ref_dataloaders,
            cfg.data.ref_ratio,
            cfg.data.ref_load_len,
        )
    else:
        feature_ref, ref_data = get_ref_features(model, ref_dataloader, cfg.data.ref_load_len)
    if not is_adaptive:
        feature_ref = feature_ref.cuda()
    regression_model = None

    if evaluation_mode == "real_only":
        all_selection_rows = []
        if is_adaptive:
            query_clip_cache = None
            if reference_mode in _CLIP_REFERENCE_MODES:
                query_clip_cache = _open_validated_clip_cache(
                    cfg,
                    "real",
                    real_model,
                    real_split,
                )
            real_scores, query_rows, all_selection_rows = _score_adaptive_query_loader(
                model,
                real_only_loader,
                reference_pool,
                reference_mode,
                cfg.reference_selection,
                query_label=0,
                generator_name=real_model,
                checkpoint_path=checkpoint_path,
                checkpoint_hash=checkpoint_hash,
                clip_cache=query_clip_cache,
            )
            real_metrics = _real_only_metrics(real_scores, regression_model)
        else:
            real_scores, query_rows, real_metrics = _score_original_real_query_loader(
                model,
                real_only_loader,
                feature_ref,
                ref_data,
                real_model,
                checkpoint_path,
                checkpoint_hash,
                regression_model,
            )

        os.makedirs(cfg.log_path, exist_ok=True)
        summary_path = os.path.join(
            cfg.log_path,
            cfg.real_only.summary_output_file,
        )
        summary_row = {
            "dataset": real_model,
            "split": real_split,
            "start_idx": int(cfg.real_only.start_idx),
            "requested_load_len": int(cfg.real_only.load_len),
            "evaluated_real": real_metrics["num_real"],
            "true_negatives": real_metrics["true_negatives"],
            "false_positives": real_metrics["false_positives"],
            "real_accuracy_percent": 100 * real_metrics["real_accuracy"],
            "tnr_percent": 100 * real_metrics["tnr"],
            "fpr_percent": 100 * real_metrics["fpr"],
            "mean_mmd_score": real_metrics["mean_mmd_score"],
            "threshold": 1.0,
            "reference_mode": reference_mode,
            "reference_size": (
                int(cfg.reference_selection.top_n)
                if is_adaptive else int(feature_ref.shape[0])
            ),
            "checkpoint_path": checkpoint_path,
            "checkpoint_hash": checkpoint_hash,
        }
        pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

        query_score_path = os.path.join(
            cfg.log_path,
            cfg.reference_selection.query_score_output_file,
        )
        pd.DataFrame(query_rows).to_csv(query_score_path, index=False)
        if is_adaptive:
            selection_path = os.path.join(
                cfg.log_path,
                cfg.reference_selection.selection_output_file,
            )
            pd.DataFrame(all_selection_rows).to_csv(selection_path, index=False)
            logger.success(f"Reference selections saved to {selection_path}")

        logger.info(
            f"Real-only result: dataset={real_model}/{real_split} | "
            f"N={real_metrics['num_real']} | TN={real_metrics['true_negatives']} | "
            f"FP={real_metrics['false_positives']} | "
            f"TNR={real_metrics['tnr']:.4f} | FPR={real_metrics['fpr']:.4f} | "
            f"Mean MMD={real_metrics['mean_mmd_score']:.6f}"
        )
        logger.success(f"Real-only summary saved to {summary_path}")
        logger.success(f"Query scores saved to {query_score_path}")
        return
        
    #  Evaluate Model
    results = []
    all_query_rows = []
    all_selection_rows = []
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for name, test_loader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset"):
            if is_adaptive:
                real_clip_cache = None
                fake_clip_cache = None
                if reference_mode in _CLIP_REFERENCE_MODES:
                    real_clip_cache = _open_validated_clip_cache(
                        cfg,
                        "real",
                        cfg.data.test_real_model,
                        "test",
                    )
                    fake_clip_cache = _open_validated_clip_cache(
                        cfg,
                        "fake",
                        name,
                        "test",
                    )
                test_results, query_rows, selection_rows = test_adaptive_dMMD(
                    model,
                    test_loader,
                    reference_pool,
                    reference_mode,
                    cfg.reference_selection,
                    name,
                    checkpoint_path,
                    checkpoint_hash,
                    real_clip_cache=real_clip_cache,
                    fake_clip_cache=fake_clip_cache,
                    regression_model=regression_model,
                )
                all_query_rows.extend(query_rows)
                all_selection_rows.extend(selection_rows)
                real_count = sum(row["query_label"] == 0 for row in query_rows)
                fake_count = sum(row["query_label"] == 1 for row in query_rows)
                logger.info(
                    f"Dataset: {name} | evaluated real={real_count}, fake={fake_count}"
                )
            else:
                legacy_result = test_dMMD(
                    model,
                    test_loader,
                    feature_ref,
                    ref_data,
                    regression_model,
                    return_query_rows=record_original_scores,
                    generator_name=name,
                    checkpoint_path=checkpoint_path,
                    checkpoint_hash=checkpoint_hash,
                )
                if record_original_scores:
                    test_results, query_rows = legacy_result
                    all_query_rows.extend(query_rows)
                    real_count = sum(row["query_label"] == 0 for row in query_rows)
                    fake_count = sum(row["query_label"] == 1 for row in query_rows)
                    logger.info(
                        f"Dataset: {name} | recorded original scores "
                        f"real={real_count}, fake={fake_count}"
                    )
                else:
                    test_results = legacy_result
            results.append([name, 
                            test_results["recall"], 
                            test_results["accuracy"], 
                            test_results["f1"], 
                            test_results["auroc"],
                            test_results["precision"], 
                            ])
            logger.info(
                f"Dataset: {name} | Recall: {test_results['recall']:.4f} | F1: {test_results['f1']:.4f} | "
                f"Accuracy: {test_results['accuracy']:.4f} | Precision: {test_results['precision']:.4f} | "
                f"AUROC: {test_results['auroc']:.4f}"
            )
    # Save results to CSV
    headers = ["Dataset", "Recall", "Accuracy", "F1", "AUROC", "Precision"]
    # Calculate mean of all metrics
    mean_metrics = ["Avg."]
    ref_result = [cfg.data.ref_load_len]
    for i in range(1, len(headers)):
        mean_metrics.append(sum(result[i] for result in results) / len(results))
        ref_result.append(sum(result[i] for result in results) / len(results))
    results.append(mean_metrics)
    logger.info("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    
    csv_path = os.path.join(cfg.log_path, cfg.save_csv_file)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Save results to CSV with four decimal places
    df = pd.DataFrame(results, columns=headers)
    df = df.applymap(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
    df = df.transpose()  # Transpose the table
    df.to_csv(csv_path, index=True, header=False)
    logger.success(f"Test results saved to {csv_path}")

    if is_adaptive or record_original_scores:
        query_score_path = os.path.join(
            cfg.log_path,
            cfg.reference_selection.query_score_output_file,
        )
        pd.DataFrame(all_query_rows).to_csv(query_score_path, index=False)
        logger.success(f"Query scores saved to {query_score_path}")
    if is_adaptive:
        selection_path = os.path.join(
            cfg.log_path,
            cfg.reference_selection.selection_output_file,
        )
        pd.DataFrame(all_selection_rows).to_csv(selection_path, index=False)
        logger.success(f"Reference selections saved to {selection_path}")

@torch.no_grad()
def test_dMMD(
        model,
        test_dataloader,
        feature_ref,
        ref_data,
        regression_model=None,
        *,
        return_query_rows=False,
        generator_name=None,
        checkpoint_path=None,
        checkpoint_hash=None):
    model.eval()
    is_smooth = model.is_smooth
    sigma = model.sigma
    sigma0_u = model.sigma0_u
    ep = model.ep
    net = model.net    
    
    fake_dataloader = test_dataloader["fake"]
    real_dataloader = test_dataloader["real"]
    dt_clean = []
    dt_adv = []
    real_ids = []
    fake_ids = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx, (real_data, fake_data) in tqdm(enumerate(zip(real_dataloader, fake_dataloader)), total=len(real_dataloader), desc="Evaluating", leave=False):
            x_real = real_data[0].float().to(device)
            x_fake = fake_data[0].float().to(device)
            if return_query_rows:
                if len(real_data) < 3 or len(fake_data) < 3:
                    raise ValueError(
                        "return_query_rows=True requires ID-bearing real and fake batches"
                    )
                batch_real_ids = [str(video_id) for video_id in real_data[2]]
                batch_fake_ids = [str(video_id) for video_id in fake_data[2]]
                if len(batch_real_ids) != x_real.shape[0]:
                    raise RuntimeError("real video IDs and NSG features are misaligned")
                if len(batch_fake_ids) != x_fake.shape[0]:
                    raise RuntimeError("fake video IDs and NSG features are misaligned")
                real_ids.extend(batch_real_ids)
                fake_ids.extend(batch_fake_ids)
            
            _,feature_cln = net(x_real,out_feature=True)
            _,feature_adv = net(x_fake,out_feature=True)
            dt_clean.append(MMD_batch2(torch.cat([feature_ref,feature_cln],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_real],dim=0).view(ref_data.shape[0]+x_real.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())
            
            dt_adv.append(MMD_batch2(torch.cat([feature_ref,feature_adv],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_fake],dim=0).view(ref_data.shape[0]+x_fake.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())

        dt_clean = torch.cat(dt_clean)
        dt_adv = torch.cat(dt_adv)
        raw_predict = torch.cat([dt_clean, dt_adv], dim=0)
        if regression_model is None:
            predict = (raw_predict > 1).int()
        else:
            predict = regression_model.predict(raw_predict.reshape(-1,1).cpu())
        labels = torch.cat([torch.zeros(len(dt_clean)), torch.ones(len(dt_adv))], dim=0)
        try:
            auroc = roc_auc_score(labels.cpu(), raw_predict.cpu())
            precision = precision_score(labels, predict)
            recall = recall_score(labels, predict)
            f1 = f1_score(labels, predict)
            acc = accuracy_score(labels, predict)
            # auroc, info = plot_mi(dt_clean, dt_adv, plot=False)
        except ValueError as e:
            logger.error(f"DeepMMD testing failed due to {e}. Exiting the program.")
        
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
    }
    if not return_query_rows:
        return metrics

    if generator_name is None:
        raise ValueError("generator_name is required when return_query_rows=True")
    if len(real_ids) != len(dt_clean) or len(fake_ids) != len(dt_adv):
        raise RuntimeError("recorded video IDs and original MMD scores are misaligned")

    predictions = torch.as_tensor(predict).reshape(-1).cpu()
    query_rows = []
    for query_label, query_ids, query_scores, query_predictions in (
            (0, real_ids, dt_clean, predictions[:len(dt_clean)]),
            (1, fake_ids, dt_adv, predictions[len(dt_clean):])):
        for query_id, score, prediction in zip(
                query_ids, query_scores, query_predictions):
            query_rows.append({
                "generator": generator_name,
                "query_id": query_id,
                "query_label": query_label,
                "mmd_score": float(score),
                "prediction": int(prediction),
                "threshold": 1.0,
                "reference_mode": "original",
                "top_n": int(feature_ref.shape[0]),
                "random_seed": None,
                "candidate_pool_size": int(feature_ref.shape[0]),
                "checkpoint_path": checkpoint_path,
                "checkpoint_hash": checkpoint_hash,
            })
    return metrics, query_rows


@torch.no_grad()
def test_adaptive_dMMD(
        model,
        test_dataloader,
        reference_pool,
        selection_mode,
        selection_cfg,
        generator_name,
        checkpoint_path,
        checkpoint_hash,
        real_clip_cache=None,
        fake_clip_cache=None,
        regression_model=None):
    """Evaluate real and fake queries with one selected reference set per query."""

    model.eval()
    real_scores, real_query_rows, real_selection_rows = _score_adaptive_query_loader(
        model,
        test_dataloader["real"],
        reference_pool,
        selection_mode,
        selection_cfg,
        query_label=0,
        generator_name=generator_name,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        clip_cache=real_clip_cache,
    )
    fake_scores, fake_query_rows, fake_selection_rows = _score_adaptive_query_loader(
        model,
        test_dataloader["fake"],
        reference_pool,
        selection_mode,
        selection_cfg,
        query_label=1,
        generator_name=generator_name,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_hash,
        clip_cache=fake_clip_cache,
    )
    metrics = _classification_metrics(real_scores, fake_scores, regression_model)
    return (
        metrics,
        real_query_rows + fake_query_rows,
        real_selection_rows + fake_selection_rows,
    )

if __name__ == "__main__":
    main()
