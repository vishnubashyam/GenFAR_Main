from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import FeatureKey, ModelFeatureSpec, load_feature_config
from .data_io import ScanItem, discover_scans, filter_scans_by_study, load_volume
from .feature_extractor import extract_features
from .model_loader import load_checkpoint
from .patch_sampler import PatchMetadata, normalize_patch, sample_patch

LOG = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    data_dir: Path
    models_dir: Path
    output_dir: Path
    config_path: Path
    device: Optional[str] = None
    seed: Optional[int] = None
    override_feature_key: Optional[FeatureKey] = None
    csv_path: Optional[Path] = None
    study_names: Optional[List[str]] = None


def run_pipeline(cfg: PipelineConfig) -> None:
    _ensure_output_dir(cfg.output_dir)
    feature_config = load_feature_config(cfg.config_path, cfg.models_dir)
    scans = discover_scans(cfg.data_dir)
    
    if cfg.csv_path and cfg.study_names:
        scans = filter_scans_by_study(scans, cfg.csv_path, cfg.study_names)
    
    if not scans:
        LOG.warning("No scans located in %s; exiting.", cfg.data_dir)
        return

    device = _resolve_device(cfg.device)
    LOG.info("Using device: %s", device)

    model_dfs: Dict[str, pd.DataFrame] = {}
    
    for model_name, spec in feature_config.models.items():
        feature_key = cfg.override_feature_key or spec.feature_key
        LOG.info("Processing model %s with feature key %s", model_name, feature_key.value)
        loaded = load_checkpoint(spec.checkpoint_path, device)
        rows = _process_scans_for_model(
            scans=scans,
            model_name=model_name,
            spec=spec,
            feature_key=feature_key,
            model=loaded.model,
            device=device,
            seed=cfg.seed,
        )
        if not rows:
            LOG.warning("No rows produced for model %s", model_name)
            continue
        model_dfs[model_name] = pd.DataFrame(rows)
        LOG.info("Processed %d scans for model %s", len(model_dfs[model_name]), model_name)
    
    if model_dfs:
        combined_df = _combine_model_features(model_dfs)
        output_path = cfg.output_dir / "features_combined.csv"
        combined_df.to_csv(output_path, index=False)
        LOG.info("Wrote combined features for %d scans to %s", len(combined_df), output_path)
    else:
        LOG.warning("No features were extracted from any model")


def _process_scans_for_model(
    scans: List[ScanItem],
    model_name: str,
    spec: ModelFeatureSpec,
    feature_key: FeatureKey,
    model: torch.nn.Module,
    device: torch.device,
    seed: Optional[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scan in tqdm(scans, desc=f"{model_name}", unit="scan"):
        try:
            row = _run_single_scan(
                scan=scan,
                model_name=model_name,
                spec=spec,
                feature_key=feature_key,
                model=model,
                device=device,
                seed=seed,
            )
            rows.append(row)
        except Exception as exc:
            LOG.exception("Failed to process scan %s with model %s: %s", scan.scan_id, model_name, exc)
    return rows


def _run_single_scan(
    scan: ScanItem,
    model_name: str,
    spec: ModelFeatureSpec,
    feature_key: FeatureKey,
    model: torch.nn.Module,
    device: torch.device,
    seed: Optional[int],
) -> Dict[str, object]:
    volume = load_volume(scan.path)
    rng = _rng_for(scan.scan_id, model_name, seed)
    patch, patch_meta = sample_patch(volume, rng)
    patch = normalize_patch(patch)

    tensor = torch.from_numpy(patch[None, None, ...])
    tensor = tensor.to(device=device, non_blocking=True)

    features = extract_features(model, tensor, feature_key)

    features_np = features.detach().cpu().numpy().reshape(-1).astype(np.float32)

    row: Dict[str, object] = {
        "scan_id": scan.scan_id,
        "scan_path": str(scan.path),
        "model_name": model_name,
        "checkpoint_path": str(spec.checkpoint_path),
        "feature_key": feature_key.value,
        "feature_dim": int(features_np.shape[0]),
        "seed": _seed_for(scan.scan_id, model_name, seed),
        "patch_origin_z": patch_meta.origin[0],
        "patch_origin_y": patch_meta.origin[1],
        "patch_origin_x": patch_meta.origin[2],
        "patch_strategy": patch_meta.strategy,
    }



    for idx, value in enumerate(features_np):
        row[f"feature_{idx:03d}"] = float(value)

    return row


def _combine_model_features(model_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not model_dfs:
        return pd.DataFrame()

    first_model_name = list(model_dfs.keys())[0]
    clean_first_model_name = first_model_name.replace("best_model_", "")
    combined = model_dfs[first_model_name].copy()

    feature_cols = [col for col in combined.columns if col.startswith("feature_") and col not in ["feature_key", "feature_dim"]]
    rename_dict = {col: f"{clean_first_model_name}_{col}" for col in feature_cols}
    combined = combined.rename(columns=rename_dict)

    cols_to_keep = ["scan_id", "scan_path"] + [f"{clean_first_model_name}_{col}" for col in feature_cols]
    combined = combined[cols_to_keep]

    for model_name in list(model_dfs.keys())[1:]:
        clean_model_name = model_name.replace("best_model_", "")
        model_df = model_dfs[model_name].copy()

        feature_cols = [col for col in model_df.columns if col.startswith("feature_") and col not in ["feature_key", "feature_dim"]]
        rename_dict = {col: f"{clean_model_name}_{col}" for col in feature_cols}
        model_df = model_df.rename(columns=rename_dict)

        cols_to_keep = ["scan_id"] + [f"{clean_model_name}_{col}" for col in feature_cols]
        model_df = model_df[cols_to_keep]

        combined = combined.merge(model_df, on="scan_id", how="outer")

    metadata_cols = ["scan_id", "scan_path"]
    feature_columns = [col for col in combined.columns if col not in metadata_cols]
    feature_columns.sort()

    final_column_order = metadata_cols + feature_columns
    combined = combined[final_column_order]

    return combined


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        if requested == "auto":
            return _resolve_device(None)
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_for(scan_id: str, model_name: str, seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    combined = (hash((seed, model_name, scan_id)) % (2**32))
    return int(combined)


def _rng_for(scan_id: str, model_name: str, seed: Optional[int]) -> np.random.Generator:
    derived = _seed_for(scan_id, model_name, seed)
    if derived is None:
        return np.random.default_rng()
    return np.random.default_rng(derived)
