from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from .config import FeatureKey

LOG = logging.getLogger(__name__)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.se = SqueezeExcitation(planes)
        self.downsample: Optional[nn.Module] = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class SEResNet3D(nn.Module):
    def __init__(self, layers: Tuple[int, int, int, int], num_outputs: int) -> None:
        super().__init__()
        base_channels = 32
        self.stem = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, layers[3], stride=2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        backbone_dim = base_channels * 8
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, num_outputs),
        )

        self._initialize_weights()

    def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [SEBasicBlock(inplanes, planes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)

    def get_features(self, features: torch.Tensor, feature_key: FeatureKey) -> torch.Tensor:
        if feature_key == FeatureKey.MLP_512:
            return self.head[:4](features)
        if feature_key == FeatureKey.MLP_64:
            return self.head[:8](features)
        raise ValueError(f"Unsupported feature key {feature_key}")


@dataclass
class LoadedModel:
    model: nn.Module
    output_dim: int


def load_checkpoint(path: Path, device: torch.device) -> LoadedModel:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, nn.Module):
        model = raw.to(device)
        model.eval()
        output_dim = _infer_output_dim_from_module(model)
        return LoadedModel(model=model, output_dim=output_dim)

    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected checkpoint type at {path}: {type(raw)}")

    state_dict = raw.get("model_state_dict", raw)
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Checkpoint at {path} does not contain a valid state dict (got {type(state_dict)})"
        )

    consume_prefix_in_state_dict_if_present(state_dict, "module.")

    # Fix checkpoint compatibility: Rename 'conv1.*' -> 'stem.*' if needed
    # The training codebase uses 'conv1' but SEResNet3D uses 'stem'
    if any(k.startswith('conv1.') for k in state_dict.keys()):
        LOG.info("Renaming 'conv1.*' -> 'stem.*' for checkpoint compatibility")
        renamed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('conv1.'):
                new_key = key.replace('conv1.', 'stem.', 1)
                renamed_state_dict[new_key] = value
            else:
                renamed_state_dict[key] = value
        state_dict = renamed_state_dict

    output_dim = _infer_output_dim_from_state_dict(state_dict)
    model = SEResNet3D(layers=(2, 2, 2, 2), num_outputs=output_dim)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Check for critical missing keys that would leave layers uninitialized
    critical_missing = [k for k in missing if k.startswith('stem.') and not k.endswith('num_batches_tracked')]
    if critical_missing:
        raise RuntimeError(
            f"Critical layers not loaded from checkpoint {path}: {critical_missing}. "
            f"This would result in random weights! Please check checkpoint compatibility."
        )

    # Filter out expected missing keys (SE bias terms if trained without bias, and num_batches_tracked)
    expected_missing = [k for k in missing if '.se.fc.' in k and k.endswith('.bias')]
    expected_missing += [k for k in missing if k.endswith('num_batches_tracked')]
    unexpected_missing = [k for k in missing if k not in expected_missing]

    if unexpected_missing:
        LOG.warning("Unexpected missing keys when loading %s: %s", path, unexpected_missing)
    if len(expected_missing) > 0:
        LOG.info("Expected missing keys (SE bias terms, likely trained with bias=False): %d keys", len(expected_missing))
    if unexpected:
        LOG.warning("Unexpected keys in checkpoint %s: %s", path, unexpected)

    model.to(device)
    model.eval()
    return LoadedModel(model=model, output_dim=output_dim)


def _infer_output_dim_from_state_dict(state_dict: dict) -> int:
    candidate = None
    for key, value in state_dict.items():
        if not key.startswith("head.") or not key.endswith(".weight"):
            continue
        if not hasattr(value, "shape"):
            continue
        candidate = value.shape[0]
    if candidate is None:
        raise RuntimeError("Unable to infer output dimension from state dict.")
    return int(candidate)


def _infer_output_dim_from_module(model: nn.Module) -> int:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith("head.8"):
            return module.out_features
    # Fallback: inspect final linear layer
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module.out_features
    raise RuntimeError("Unable to infer output dimension from module.")
