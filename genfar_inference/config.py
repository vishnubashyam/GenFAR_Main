from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

CONFIG_FILENAME = "model_features.yaml"


class ConfigError(RuntimeError):
    pass


class FeatureKey(Enum):
    MLP_64 = "mlp_64"
    MLP_512 = "mlp_512"

    @property
    def dimension(self) -> int:
        if self is FeatureKey.MLP_64:
            return 64
        if self is FeatureKey.MLP_512:
            return 512
        raise ValueError(f"Unsupported feature key: {self.value}")


@dataclass(frozen=True)
class ModelFeatureSpec:
    checkpoint_path: Path
    feature_key: FeatureKey


@dataclass(frozen=True)
class FeatureConfig:
    models: Dict[str, ModelFeatureSpec]
    default_feature_key: FeatureKey


def _load_yaml(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise ConfigError(f"Configuration root must be a mapping, got {type(data)}")
    return data


def parse_feature_key(value: str, *, context: str) -> FeatureKey:
    try:
        return FeatureKey(value)
    except ValueError as exc:
        allowed = ", ".join(k.value for k in FeatureKey)
        raise ConfigError(f"Invalid feature key '{value}' in {context}; allowed: {allowed}") from exc


def load_feature_config(config_path: Path, models_dir: Path) -> FeatureConfig:
    raw = _load_yaml(config_path)
    defaults = raw.get("defaults", {}) or {}
    if not isinstance(defaults, MutableMapping):
        raise ConfigError("`defaults` section must be a mapping")

    default_key_raw = defaults.get("feature_key", FeatureKey.MLP_512.value)
    default_feature_key = parse_feature_key(str(default_key_raw), context="defaults.feature_key")

    raw_models = raw.get("models", {}) or {}
    if not isinstance(raw_models, MutableMapping):
        raise ConfigError("`models` section must be a mapping")

    resolved: Dict[str, ModelFeatureSpec] = {}
    for model_name, model_cfg in raw_models.items():
        if not isinstance(model_cfg, MutableMapping):
            raise ConfigError(f"Model entry '{model_name}' must be a mapping")
        feature_key_raw = model_cfg.get("feature_key")
        feature_key = (
            parse_feature_key(str(feature_key_raw), context=f"models.{model_name}.feature_key")
            if feature_key_raw
            else default_feature_key
        )
        checkpoint_path = _resolve_checkpoint_path(model_name, models_dir)
        resolved[model_name] = ModelFeatureSpec(checkpoint_path=checkpoint_path, feature_key=feature_key)

    return FeatureConfig(models=resolved, default_feature_key=default_feature_key)


def _resolve_checkpoint_path(model_stem: str, models_dir: Path) -> Path:
    candidates = list(models_dir.glob(f"{model_stem}.pth"))
    if not candidates:
        raise ConfigError(f"Checkpoint for model '{model_stem}' not found in {models_dir}")
    if len(candidates) > 1:
        raise ConfigError(
            f"Multiple checkpoints found for model '{model_stem}' in {models_dir}: {candidates}"
        )
    return candidates[0]


def infer_model_names(checkpoint_paths: Iterable[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in checkpoint_paths:
        stem = path.stem
        if stem in mapping and mapping[stem] != path:
            raise ConfigError(f"Duplicate checkpoint stem '{stem}' for paths {mapping[stem]} and {path}")
        mapping[stem] = path
    return mapping
