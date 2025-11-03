from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

SCAN_EXTENSIONS = {".nii", ".nii.gz"}


@dataclass(frozen=True)
class ScanItem:
    scan_id: str
    path: Path


def discover_scans(root: Path, recursive: bool = True) -> List[ScanItem]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    candidates: Iterable[Path]
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.glob("*")

    scans: List[ScanItem] = []
    for path in candidates:
        if not path.is_file():
            continue
        if not _is_nifti(path):
            continue
        # Extract scan ID properly handling .nii.gz extensions
        scan_id = path.stem
        if scan_id.endswith('.nii'):
            scan_id = scan_id[:-4]  # Remove .nii part
        scans.append(ScanItem(scan_id=scan_id, path=path))
    scans.sort(key=lambda item: item.scan_id)
    if not scans:
        LOG.warning("No NIfTI scans were found in %s", root)
    return scans


def filter_scans_by_study(
    scans: List[ScanItem],
    csv_path: Path,
    study_names: Optional[List[str]] = None,
    id_column: str = "ID",
    study_column: str = "STUDY",
) -> List[ScanItem]:
    if study_names is None:
        return scans
    
    if not csv_path.exists():
        LOG.warning("CSV file not found: %s. Returning all scans.", csv_path)
        return scans
    
    df = pd.read_csv(csv_path)
    if id_column not in df.columns:
        raise ValueError(f"CSV file must contain '{id_column}' column")
    if study_column not in df.columns:
        raise ValueError(f"CSV file must contain '{study_column}' column")
    
    # Filter to only include specified studies
    filtered_df = df[df[study_column].isin(study_names)]
    valid_ids = set(filtered_df[id_column].unique())
    
    LOG.info("Found %d IDs in studies %s", len(valid_ids), study_names)
    
    # Filter scans to only include those with matching IDs
    # Note: scan_id is the stem (filename without extension), so we need to match
    # against the file pattern {scan_id}_T1_BrainAligned.nii.gz
    filtered_scans = []
    for scan in scans:
        # Extract the base scan ID from the filename
        # Files are named like: {scan_id}_T1_BrainAligned.nii.gz
        base_id = scan.scan_id.replace("_T1_BrainAligned", "")
        if base_id in valid_ids:
            filtered_scans.append(scan)
    
    LOG.info("Filtered to %d scans from %d total", len(filtered_scans), len(scans))
    return filtered_scans


def load_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {data.shape}")
    return data


def _is_nifti(path: Path) -> bool:
    suffix = "".join(path.suffixes[-2:]).lower() if len(path.suffixes) >= 2 else path.suffix.lower()
    if suffix in SCAN_EXTENSIONS:
        return True
    if path.suffix.lower() in SCAN_EXTENSIONS:
        return True
    return False
