from utils.experiment_utils import set_seed
from data.utils import get_generation_models
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
from utils.clip_feature_cache import ClipFeatureCache, sha256_file
from utils.clip_video_features import clip_cache_model_key
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
        if selection_mode == "clip_topn":
            if clip_cache is None:
                raise ValueError("clip_topn requires a query CLIP cache")
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
        batch_predictions = (batch_scores > 1).int()
        scores.append(batch_scores)

        for index, query_id in enumerate(query_ids):
            query_rows.append({
                "generator": generator_name,
                "query_id": query_id,
                "query_label": query_label,
                "mmd_score": float(batch_scores[index]),
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
                    "rank": selection.rank,
                    "clip_similarity": selection.similarity,
                    "reference_mode": selection_mode,
                    "top_n": int(selection_cfg.top_n),
                    "random_seed": int(selection_cfg.random_seed),
                })

    if not scores:
        raise RuntimeError(f"query loader is empty for {generator_name}, label={query_label}")
    return torch.cat(scores), query_rows, selection_rows

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
    reference_mode = _reference_selection_mode(cfg)
    is_adaptive = reference_mode != "original"
    logger.info(f"Reference-selection mode: {reference_mode}")

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
                                              return_video_id=is_adaptive)
            test_loader = get_data_loaders_for_mmd(
                cfg.data,
                test_datasets,
                batch_size=cfg.data.val_batch_size,
                shuffle=not is_adaptive,
            )
            test_dataloaders[f"{fake_model}"] = test_loader

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
        if reference_mode == "clip_topn":
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
                if reference_mode == "clip_topn":
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
                test_results = test_dMMD(
                    model, test_loader, feature_ref, ref_data,
                    regression_model
                )
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

    if is_adaptive:
        query_score_path = os.path.join(
            cfg.log_path,
            cfg.reference_selection.query_score_output_file,
        )
        selection_path = os.path.join(
            cfg.log_path,
            cfg.reference_selection.selection_output_file,
        )
        pd.DataFrame(all_query_rows).to_csv(query_score_path, index=False)
        pd.DataFrame(all_selection_rows).to_csv(selection_path, index=False)
        logger.success(f"Query scores saved to {query_score_path}")
        logger.success(f"Reference selections saved to {selection_path}")

@torch.no_grad()
def test_dMMD(model, test_dataloader, feature_ref, ref_data, regression_model=None):
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
    with torch.no_grad():
        for idx, (real_data, fake_data) in tqdm(enumerate(zip(real_dataloader, fake_dataloader)), total=len(real_dataloader), desc="Evaluating", leave=False):
            x_real = real_data[0].float().cuda()
            x_fake = fake_data[0].float().cuda()
            
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
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
    }


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
