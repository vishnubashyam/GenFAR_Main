from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

from .config import FeatureKey
from .pipeline import PipelineConfig, run_pipeline

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data"),
    show_default=True,
    help="Directory containing NIfTI volumes.",
)
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("models"),
    show_default=True,
    help="Directory containing model checkpoints (.pth).",
)
@click.option(
    "--config-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/model_features.yaml"),
    show_default=True,
    help="YAML configuration mapping models to feature taps.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("outputs"),
    show_default=True,
    help="Directory where CSV outputs will be written.",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    show_default=True,
    help="Torch device identifier (cpu, cuda, cuda:0, auto).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Base seed for deterministic patch sampling (optional).",
)
@click.option(
    "--feature-key",
    type=click.Choice([key.value for key in FeatureKey]),
    default=None,
    help="Override feature tap for all models (mlp_64 or mlp_512).",
)
@click.option(
    "--csv-path",
    type=click.Path(path_type=Path, exists=True, file_okay=True),
    default=None,
    help="Path to CSV file containing ID and STUDY columns for filtering.",
)
@click.option(
    "--study-names",
    type=str,
    default=None,
    help="Comma-separated list of study names to include (e.g., 'PNC,PING').",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def main(
    data_dir: Path,
    models_dir: Path,
    config_path: Path,
    output_dir: Path,
    device: str,
    seed: Optional[int],
    feature_key: Optional[str],
    csv_path: Optional[Path],
    study_names: Optional[str],
    verbose: bool,
) -> None:
    _configure_logging(verbose)
    key = FeatureKey(feature_key) if feature_key else None
    
    # Parse study names if provided
    study_list = None
    if study_names:
        study_list = [s.strip() for s in study_names.split(",")]
    
    cfg = PipelineConfig(
        data_dir=data_dir,
        models_dir=models_dir,
        output_dir=output_dir,
        config_path=config_path,
        device=device,
        seed=seed,
        override_feature_key=key,
        csv_path=csv_path,
        study_names=study_list,
    )
    run_pipeline(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
