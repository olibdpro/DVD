# dataset_UnrealEXR.py
"""
Unreal Engine EXR dataset loader
─────────────────────────────────────────────────────────────────────────────
Loads RGB, depth, and camera data from pre-extracted EXR passes produced by
`extract_exr_passes.py`. Scene manifests are CSV files (one per scene) with
columns: scene, original_path, depht_path, normal_path, rgb_path, camera_path, error.

A top-level dataset.csv indexes all scenes and points to their per-scene CSVs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any, Mapping, Sequence

from data.dataset_MultiDatasets import VideoDepthDataset
from utils.io import read_exr_with_openexr


def _apply_filter(ddf: pd.DataFrame, spec: Filter) -> pd.DataFrame:
    """
    Applies a series of transformations to a Dask DataFrame based on the specifications
    provided in the Filter object. This includes querying, mapping rows, selecting
    specific columns, and normalizing object columns as per the provided instructions.

    Args:
        ddf: The Dask DataFrame to which the transformations will be applied.
        spec: A Filter object containing the transformation instructions including query,
              row mapping function, metadata, columns to write, and query local variables.

    Returns:
        A transformed Dask DataFrame after applying all specified operations.

    Raises:
        ValueError: If `spec.meta` is not provided when `spec.row_fn` is specified.
    """
    if spec.query:
        ddf = ddf.query(spec.query, local_dict=spec.query_locals or {})

    if spec.row_fn is not None:
        ddf = ddf.apply(lambda r: dict(spec.row_fn(r.to_dict())), axis=1, result_type="expand")

    if spec.write_columns is not None:
        ddf = ddf[list(spec.write_columns)]

    return ddf


RowFn = Callable[[Dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class Filter:
    """
    Specification dataframe filter and transformation logic. The parameters are
    effectively the same arguments used by built-in pandas (and dask)
    functions.

    For example:
        - The query string is passed to DataFrame.query,
        - The row function is use in pandas apply
    The purpose of this class is purely structural: it bundles commonly-used pandas
    arguments into a single, typed object so callers can define filters in configs.

    Attributes:
        query (str): Query string used to filter or process the data.
        query_locals (Optional[Dict[str, Any]]): Optional local variables to be used in the
            query context.
        row_fn (Optional[RowFn]): Optional row function to apply custom transformations to
            the rows.
        write_columns (Optional[Sequence[str]]): List of columns to include when writing
            the data.
    """
    query: str = ""
    query_locals: Optional[Dict[str, Any]] = None
    row_fn: Optional[RowFn] = None
    write_columns: Optional[Sequence[str]] = None


class UnrealEXRDataset(VideoDepthDataset):
    """Clip-based loader for Unreal Engine EXR scenes with per-frame camera JSON."""

    def __init__(
            self,
            dataset_csv: str,
            num_samples_per_epoch: int = 10000,
            clip_len: int = 16,
            stride: tuple[int, int] = (1, 3),
            resize: tuple[int, int] | None = None,
            interpolation: str = "BILINEAR",
            transform=None,
            rgb_norm: bool = True,
            depth_canon: bool = True,
            depth_canon_focal_px: float = 1024.0,
            depth_norm: bool = True,
            depth_norm_mode: str = "percentile",
            depth_norm_far_plane: float = 655.0,
            depth_norm_k_compression: float = 1.0,
            depth_norm_p_lo: float = 0.1,
            depth_norm_p_hi: float = 99.0,
            depth_norm_inv_mode: str = "flip",
            curve_correction_mode: str = "none",
            dataset_filter: Filter = None,
            scene_filter: Filter = None,
    ):
        self.dataset_csv = Path(dataset_csv)
        # Cache for per-call state: the scene CSV and indices used by load_rgbs,
        # so that __getitem__ can load matching camera data without re-deriving
        # the parent's random stride logic.
        self._last_scene_csv: Path | None = None
        self._last_indices: List[int] | None = None
        self._dataset_filter = dataset_filter
        self._scene_filter = scene_filter

        super().__init__(
            num_samples_per_epoch=num_samples_per_epoch,
            clip_len=clip_len,
            stride=stride,
            resize=resize,
            interpolation=interpolation,
            transform=transform,
            rgb_norm=rgb_norm,
            depth_canon=depth_canon,
            depth_canon_focal_px=depth_canon_focal_px,
            depth_norm=depth_norm,
            depth_norm_mode=depth_norm_mode,
            depth_norm_far_plane=depth_norm_far_plane,
            depth_norm_k_compression=depth_norm_k_compression,
            depth_norm_p_lo=depth_norm_p_lo,
            depth_norm_p_hi=depth_norm_p_hi,
            depth_norm_inv_mode=depth_norm_inv_mode,
            curve_correction_mode=curve_correction_mode,
        )

    # ------------------------------------------------------------------ index
    def build_indices(self) -> List[Tuple[Path, Path, int, int, List[int] | None]]:
        df_dataset = pd.read_csv(self.dataset_csv)
        if self._dataset_filter:
            df_dataset = _apply_filter(df_dataset, self._dataset_filter)
        index: list[tuple] = []
        required = 1 + (self.clip_len - 1) * self.stride[1]

        for _, row in df_dataset.iterrows():
            if pd.notna(row.get("skipped")) and row["skipped"]:
                continue

            scene_csv_path = Path(row["scene_path"])
            if not scene_csv_path.exists():
                continue

            df_scene = pd.read_csv(scene_csv_path)
            if "error" in df_scene.columns:
                df_scene = df_scene[df_scene["error"].isna() | (df_scene["error"] == "")]

            if self._scene_filter is not None:
                df_scene = _apply_filter(df_scene, self._scene_filter)

            n_frames = len(df_scene)
            if n_frames < required:
                continue

            # Both rgb_scene and depth_scene point to the same scene CSV;
            # load_rgbs / load_depths read the relevant column from it.
            frame_indices = list(range(n_frames))
            index.append((scene_csv_path, scene_csv_path, n_frames, 0, frame_indices))

        if index:
            shortest = min(entry[2] for entry in index)
            max_ok = (shortest - 1) // (self.clip_len - 1)
            if max_ok < 1:
                raise ValueError("All Unreal scenes are shorter than clip_len")
            if self.stride[1] > max_ok:
                self.stride = (self.stride[0], max_ok)

        return index

    # --------------------------------------------------------- CSV helper
    def _read_scene_csv(self, scene_csv: Path) -> pd.DataFrame:
        df = pd.read_csv(scene_csv)
        if "error" in df.columns:
            df = df[df["error"].isna() | (df["error"] == "")]
        return df.reset_index(drop=True)

    # --------------------------------------------------------- EXR loaders
    def _load_exr_rgb(self, path: Path) -> torch.Tensor:
        rgb = read_exr_with_openexr(path)  # H, W, 3  float32
        if rgb is None:
            raise IOError(f"Failed to read EXR: {path}")
        rgb = np.where(np.isfinite(rgb), rgb, 0.0).astype(np.float32, copy=False)
        rgb = np.clip(rgb, 0.0, 1.0)
        if self.rgb_norm:
            from utils.data import normalize_rgb_images
            rgb = normalize_rgb_images(rgb * 255.0)
        return torch.from_numpy(rgb).permute(2, 0, 1)  # 3×H×W

    def _load_exr_depth(self, path: Path) -> torch.Tensor:
        img = read_exr_with_openexr(path)  # H, W, 3 (single-ch duplicated)
        if img is None:
            raise IOError(f"Failed to read depth EXR: {path}")
        depth = img[:, :, 0]
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32, copy=False)
        return torch.from_numpy(depth).unsqueeze(0)  # 1×H×W

    def _load_camera_json(self, path: Path) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    # ---------------------------------------- VideoDepthDataset interface
    def load_rgbs(self, scene_csv: Path, indices: List[int]) -> torch.Tensor:
        # Cache scene + indices so __getitem__ can load matching cameras
        self._last_scene_csv = scene_csv
        self._last_indices = list(indices)

        df = self._read_scene_csv(scene_csv)
        frames = [self._load_exr_rgb(Path(df.iloc[i]["rgb_path"])) for i in indices]
        return torch.stack(frames, dim=1)  # 3×F×H×W

    def load_depths(self, scene_csv: Path, indices: List[int]) -> torch.Tensor:
        df = self._read_scene_csv(scene_csv)
        frames = [self._load_exr_depth(Path(df.iloc[i]["depht_path"])) for i in indices]
        return torch.stack(frames, dim=1)  # 1×F×H×W

    def load_cameras(self, scene_csv: Path, indices: List[int]) -> List[dict]:
        df = self._read_scene_csv(scene_csv)
        return [self._load_camera_json(Path(df.iloc[i]["camera_path"])) for i in indices]

    def get_fx_and_width(self, rgb_scene: Path, depth_scene: Path, indices: List[int]) -> Tuple[List[float], int] | None:
        cameras = self.load_cameras(rgb_scene, indices)
        if not cameras:
            return None

        # Determine native image width from the first RGB frame
        df = self._read_scene_csv(rgb_scene)
        first_rgb = read_exr_with_openexr(Path(df.iloc[indices[0]]["rgb_path"]))
        if first_rgb is None:
            return None
        W_orig = first_rgb.shape[1]

        fx_list = []
        for cam in cameras:
            focal_length = cam.get("focal_length")
            sensor_width = cam.get("sensor_width")
            if focal_length is not None and sensor_width is not None and sensor_width > 0:
                fx_px = (focal_length / sensor_width) * W_orig
            else:
                fx_px = float(W_orig)
            fx_list.append(fx_px)

        return fx_list, W_orig

    # --------------------------------------------------------- __getitem__
    def __getitem__(self, idx):
        # Parent calls load_rgbs (which caches scene_csv + indices), then
        # load_depths, then get_fx_and_width — all with the same indices.
        out = super().__getitem__(idx)

        # Attach camera metadata using the indices cached by load_rgbs
        if self._last_scene_csv is not None and self._last_indices is not None:
            cameras = self.load_cameras(self._last_scene_csv, self._last_indices)
            out["cameras"] = cameras

        return out


class DepthCollapseDataset(UnrealEXRDataset):
    def __init__(
            self,
            dataset_csv: str,
            num_samples_per_epoch: int = 10000,
            clip_len: int = 16,
            stride: tuple[int, int] = (1, 3),
            resize: tuple[int, int] | None = None,
            interpolation: str = "BILINEAR",
            transform=None,
            rgb_norm: bool = True,
            depth_canon: bool = True,
            depth_canon_focal_px: float = 1024.0,
            depth_norm: bool = True,
            depth_norm_mode: str = "percentile",
            depth_norm_far_plane: float = 655.0,
            depth_norm_k_compression: float = 1.0,
            depth_norm_p_lo: float = 0.1,
            depth_norm_p_hi: float = 99.0,
            depth_norm_inv_mode: str = "flip",
            curve_correction_mode: str = "none",
    ):
        dataset_filter = Filter(
            query="scene in @scenes",
            query_locals={ "scenes": [
                "SD_065",# Pedestal longest
                "SD_105",# Roll longest/fastest/jittery
                "SD_107",# Pan longest/fastest
                "SD_111",# Deepest depth change uncorrelated with camera (camera static, something moves in/out at distance)
                "SD_114",# Pan most jittery
                "SD_124",# Tilt longest/fastest/jittery
                "SD_312",# Biggest depth change uncorrelated with camera (camera static, large overall depth shift from scene dynamics)
                "SD_318",# Smallest depth change (most static depth over time)
                "SD_338",# Narrowest depth (tightest depth range with camera motion)
                "SD_483",# Extreme depth and normal variation from pure rotation (camera pivots in place, sweeping near and far geometry)
                "SD_571",#Truck longest
                "SD_577",# Pedestal fastest/jittery + near-field depth changes (something moving/appearing close to camera)
                "SD_578",# Truck fastest/jittery
                "SD_585",# Deepest depth (farthest real geometry, not far-plane)
                "SD_604",# Biggest depth change correlated with camera motion (camera moves through varying environments)
                "SD_621",# Dolly longest/fastest/jittery + deepest depth change correlated with camera motion (camera explores, revealing new distant geometry)
            ]}
        )
        scene_filter = Filter()
        super().__init__(
            dataset_csv=dataset_csv,
            num_samples_per_epoch=num_samples_per_epoch,
            clip_len=clip_len,
            stride=stride,
            resize=resize,
            interpolation=interpolation,
            transform=transform,
            rgb_norm=rgb_norm,
            depth_canon=depth_canon,
            depth_canon_focal_px=depth_canon_focal_px,
            depth_norm=depth_norm,
            depth_norm_mode=depth_norm_mode,
            depth_norm_far_plane=depth_norm_far_plane,
            depth_norm_k_compression=depth_norm_k_compression,
            depth_norm_p_lo=depth_norm_p_lo,
            depth_norm_p_hi=depth_norm_p_hi,
            depth_norm_inv_mode=depth_norm_inv_mode,
            curve_correction_mode=curve_correction_mode,
            dataset_filter=dataset_filter,
            scene_filter=scene_filter
        )
