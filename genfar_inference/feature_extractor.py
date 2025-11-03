from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import nn

from .config import FeatureKey

LOG = logging.getLogger(__name__)


def run_model(
    model: nn.Module,
    tensor: torch.Tensor,
    feature_key: FeatureKey,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        prediction = model(tensor)
        backbone = _forward_backbone(model, tensor)
        features = _select_feature(model, backbone, feature_key)
    return prediction, features

def extract_features(
    model: nn.Module,
    tensor: torch.Tensor,
    feature_key: FeatureKey,
) -> torch.Tensor:
    with torch.inference_mode():
        backbone = _forward_backbone(model, tensor)
        features = _select_feature(model, backbone, feature_key)
    return features

def _forward_backbone(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        return getattr(model, "forward_features")(tensor)
    if hasattr(model, "forward_backbone"):
        return getattr(model, "forward_backbone")(tensor)
    if hasattr(model, "forward_features_at_layer"):
        return getattr(model, "forward_features_at_layer")(tensor, "backbone")
    raise AttributeError("Model does not provide a forward_features method required for embeddings.")


def _select_feature(model: nn.Module, backbone: torch.Tensor, feature_key: FeatureKey) -> torch.Tensor:
    if hasattr(model, "get_features"):
        return getattr(model, "get_features")(backbone, feature_key)

    # Attempt to use provided helper if available.
    if hasattr(model, "forward_features_at_layer"):
        layer_name = "mlp_512" if feature_key == FeatureKey.MLP_512 else "mlp_64"
        return getattr(model, "forward_features_at_layer")(backbone, layer_name)

    head = getattr(model, "head", None)
    if isinstance(head, nn.Sequential):
        if feature_key == FeatureKey.MLP_512:
            return head[:4](backbone)
        if feature_key == FeatureKey.MLP_64:
            return head[:8](backbone)

    raise AttributeError(f"Unable to extract feature key {feature_key.value} from model.")
