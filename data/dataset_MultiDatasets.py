import os
import re
from typing import List, Tuple
from abc import ABC, abstractmethod
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import cv2
from natsort import natsorted
import h5py

from utils.utils import EPS
from utils.images import resize_rgbs_clip
from utils.data import  INTERPOLATION_MAP_CV2, normalize_rgb_images, normalize_depth_percentile, normalize_depth_logarithmic, apply_fusion_color_curve

from data.Sintel.sintel_io import depth_read

class MixedVideoDataset(Dataset):
    def __init__(self, datasets: list):
        self.datasets = datasets

    def __len__(self):
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        # ponytail: NFS gives transient ENOENT/ESTALE on files that exist;
        # samples are random anyway, so resample instead of killing the rank
        # (one dead rank stalls the others in allgather until NCCL timeout).
        last_err = None
        for attempt in range(5):
            ds = random.choice(self.datasets)
            try:
                return ds[random.randint(0, len(ds) - 1)]
            except OSError as e:
                last_err = e
                print(f" *** WARNING - dataset read failed (attempt {attempt + 1}/5), "
                      f"resampling: {e}", flush=True)
                time.sleep(2 ** attempt)
        raise last_err


def collate_depth_batch(batch: list) -> dict:
    """default_collate, except rgb_latent/depth_latent are only stacked when
    EVERY sample cache-hit. Their real shape (2, C, T, h_lat, w_lat) comes
    from the VAE and isn't knowable from a miss placeholder, so a mixed
    hit/miss batch has no consistent shape to stack. Downstream code already
    gates all reads of these two keys behind the same all-hit check (see
    depth_transformer.py), so dropping them here is a no-op for training.
    """

    # When collating from multiple dataset, it is not
    # guaranteed that all of them will return all
    # the same keys, we can only collate when
    # thats the case
    all_common_keys = set.intersection(*(set(d) for d in batch))
    all_keys = set.union(*(set(d) for d in batch))
    missing_keys = all_keys - all_common_keys

    all_hit = all(bool(s.get("_cache_hit", torch.tensor(False))) for s in batch)
    for d in batch:
        for key in missing_keys:
            d.pop(key, None)
        if not all_hit:
            d.pop("rgb_latent", None)
            d.pop("depth_latent", None)

    return torch.utils.data.default_collate(batch)


class VideoDepthDataset(Dataset, ABC):
    @property
    def dataset_name(self) -> str:
        return type(self).__name__.removesuffix("Dataset")

    def __init__(self,
                 num_samples_per_epoch=10000,
                 clip_len=16,
                 stride=(1, 3),
                 resize=None,
                 interpolation="BILINEAR",
                 transform=None,
                 rgb_norm=True,
                 depth_canon: bool = True,
                 depth_canon_focal_px: float = 1024.0,
                 depth_norm=True,
                 depth_norm_mode="percentile",
                 depth_norm_far_plane=655.00,
                 depth_norm_k_compression=1.0,
                 depth_norm_p_lo=0.1,
                 depth_norm_p_hi=99.0,
                 depth_norm_inv_mode="flip",
                 curve_correction_mode: str = "none",
        ):
        self.clip_len = clip_len
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.resize = resize
        self.interp = INTERPOLATION_MAP_CV2[interpolation]
        self.transform = transform
        self.rgb_norm = rgb_norm
        self.depth_canon = depth_canon
        self.depth_canon_focal_px = float(depth_canon_focal_px)
        self.depth_norm = depth_norm
        self.depth_norm_mode = depth_norm_mode
        self.depth_norm_far_plane = depth_norm_far_plane
        self.depth_norm_k_compression = depth_norm_k_compression
        self.depth_norm_p_lo = depth_norm_p_lo
        self.depth_norm_p_hi = depth_norm_p_hi
        self.depth_norm_inv_mode = depth_norm_inv_mode
        self.curve_correction_mode = curve_correction_mode
        self.num_samples_per_epoch = num_samples_per_epoch
        self.latent_cache = None
        # Why: during a caching-only pass the trainer returns a zero loss
        # before reading raw pixel content; loading 32 frames + percentile
        # norm per sample dominates wall time. When set, __getitem__ probes
        # the cache and skips raw I/O on a full hit.
        # How to apply: only flip True when downstream will not consume
        # rgbs_clip/depths_clip_* content (e.g. _first_epoch_cache_latent_only).
        self.skip_raw_on_cache_hit: bool = False
        self.index = self.build_indices()

    def _scene_key(self, rgb_scene: Path) -> str:
        """Unique-within-dataset identifier for a scene, derived from the
        scene path relative to the dataset root."""
        base = getattr(self, "rgb_root", None) or getattr(self, "root", None)
        return rgb_scene.relative_to(base).as_posix() if base is not None else rgb_scene.name

    @abstractmethod
    def build_indices(self):
        """Return a list of (scene_path, frame_count) tuples"""
        pass

    @abstractmethod
    def load_rgbs(self, scene_path, indices):
        pass

    @abstractmethod
    def load_depths(self, scene_path, indices):
        pass

    def get_fx_and_width(self, rgb_scene: Path, depth_scene: Path, indices: List[int]) -> Tuple[List[float], int] | None:
        """
        Return (fx_list_per_frame, original_width_pixels) or None to skip.
        Base returns None; datasets that support canonicalization override.
        """
        return None

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, idx):
        idx_scene = idx % len(self.index)
        rgb_scene, depth_scene, frame_count, first_idx, frame_indices = self.index[idx_scene]

        max_stride = self.stride[1]

        # Choose sample range
        max_start_offset = frame_count - 1 - (self.clip_len - 1) * max_stride
        if max_start_offset < 0:          # not enough frames for this stride
            return self.__getitem__((idx + 1) % self.__len__())   # resample
        
        offset_start = torch.randint(0, max_start_offset + 1, (1,)).item()

        # Sample stride sequence
        stride_seq = [
            torch.randint(*self.stride, (1,)).item() if self.stride[0] != self.stride[1]
            else self.stride[0]
            for _ in range(self.clip_len - 1)
        ]
        # Compute frame indices (non-contiguous vs contiguous case)
        if frame_indices is not None:                     # non-contiguous case
            pos = offset_start                            # position inside list
            indices = [frame_indices[pos]]
            for s in stride_seq:                          # stride = steps *inside* list
                pos += s
                if pos >= len(frame_indices):             # ran past the list – resample
                    return self.__getitem__((idx + 1) % self.__len__())
                indices.append(frame_indices[pos])
        else:                                             # contiguous case
            base_index = first_idx + offset_start
            indices = [base_index]
            for s in stride_seq:
                indices.append(indices[-1] + s)

        # Cheap cache probe + fast path: when enabled and all 32 latents already
        # exist on disk, skip the 16 RGB + 16 depth reads, canonical scaling,
        # percentile norm, and resize. Returns placeholder zero tensors for the
        # pixel-content keys so default_collate sees a consistent schema.
        if (
            self.latent_cache is not None
            and self.skip_raw_on_cache_hit
            and self.resize is not None
        ):
            ds_name = self.dataset_name
            seq_key = self._scene_key(rgb_scene)
            rgb_paths = [self.latent_cache.path(ds_name, seq_key, i, "rgb") for i in indices]
            depth_paths = [self.latent_cache.path(ds_name, seq_key, i, "depth") for i in indices]
            if all(p.exists() for p in rgb_paths) and all(p.exists() for p in depth_paths):
                rgb_data = [torch.load(p, map_location="cpu").clone() for p in rgb_paths]
                depth_data = [torch.load(p, map_location="cpu").clone() for p in depth_paths]
                return self._cache_hit_fast_dict(
                    indices, ds_name, seq_key, rgb_data, depth_data,
                )

        # Load at native resolution (no resizing here anymore)
        rgbs_clip = self.load_rgbs(rgb_scene, indices)     # (3, F, H0, W0)
        depths_clip = self.load_depths(depth_scene, indices)  # (1, F, H0, W0) at native res

        # Resize RGBs exactly once here (canonical-normalization stage) ---
        rgbs_clip = resize_rgbs_clip(rgbs_clip, self.resize)      # (3, F, h, w) if self.resize set

        # Initialize output dict AFTER we set the resized RGBs
        out = {"rgbs_clip": rgbs_clip}

        # --- Canonical depth scaling (LabelScaleCanonical adapted to your pipeline) ---
        # We assume final width = self.resize[1] (e.g., 1024). Scale depth by:
        # s = (f_canon * W_orig) / (fx_orig * W_target)
        F_frames = depths_clip.shape[1]
        depth_canon_scaler = torch.ones((1, F_frames, 1, 1), dtype=depths_clip.dtype)

        if self.depth_canon:
            fxW = self.get_fx_and_width(rgb_scene, depth_scene, indices)
            if fxW is not None:
                fx_list, W_orig = fxW
                assert self.resize is not None and len(self.resize) == 2, "Canonical depth scaling requires a target resize (H,W)."
                W_target = int(self.resize[1])
                s_list = [
                    (self.depth_canon_focal_px * float(W_orig)) / (float(fx) * float(W_target) + 1e-8)
                    for fx in fx_list
                ]
                depth_canon_scaler = torch.tensor(s_list, dtype=depths_clip.dtype).view(1, -1, 1, 1)

        depths_clip = depths_clip * depth_canon_scaler

        # Save the scalar for the de‑canonicalization later
        out["depth_canon_scaler"] = depth_canon_scaler.clone()  # (1,F,1,1)

        # NORMALIZE FULL-RESOLUTION DEPTH
        d_proc = depths_clip
        if self.depth_norm:
            if self.depth_norm_mode == "percentile":
                d_proc, (d_lo, d_hi) = normalize_depth_percentile(d_proc, p_lo=self.depth_norm_p_lo, p_hi=self.depth_norm_p_hi, far_plane=self.depth_norm_far_plane, k_compression=self.depth_norm_k_compression, flip_output_sign=self.depth_norm_inv_mode == "flip")
            elif self.depth_norm_mode == "logarithmic":
                d_proc, = normalize_depth_logarithmic(d_proc, k=self.depth_norm_k_compression, far_plane=self.depth_norm_far_plane)
                if self.depth_norm_inv_mode == "flip":
                    d_proc = -d_proc
        
        # RESIZE THE (POSSIBLY FLIPPED) NORMALIZED DEPTH MAP
        if self.resize:
            h, w = self.resize

            C_d, num_frames_d, _, _ = d_proc.shape 
            d_proc_reshaped = d_proc.permute(1, 0, 2, 3) # -> (F, C_d, H, W)
            d_proc_resized = F.interpolate(d_proc_reshaped, size=(h, w), mode='bilinear', align_corners=False)
            d_proc = d_proc_resized.permute(1, 0, 2, 3) # -> (C_d, F, h, w)
        
        # At this point, d_proc is NORMALIZED, (POSSIBLY) FLIPPED, and RESIZED.
        if self.curve_correction_mode != "none":
            d_proc = apply_fusion_color_curve(d_proc, curve_name=self.curve_correction_mode)

        # POPULATE THE MODEL INPUT KEY WITH FINAL, CORRECTLY-SIZED TENSOR
        if self.depth_norm and self.depth_norm_inv_mode == "flip":
            out["depths_clip_norm_inv_rgb"] = d_proc.repeat(3, 1, 1, 1)
        else:
            out["depths_clip_norm_inv_rgb"] = d_proc.repeat(3, 1, 1, 1) if d_proc.shape[0] == 1 else d_proc

        if self.latent_cache is not None:
            ds_name = self.dataset_name
            seq_key = self._scene_key(rgb_scene)
            out["_cache_ds_name"] = ds_name
            out["_cache_seq_name"] = seq_key
            out["_cache_frame_indices"] = torch.tensor(indices, dtype=torch.long)
            # Each cache file is a stacked tensor [2, C, h, w] with mu at [0] and sigma at [1].
            rgb_data = [self.latent_cache.load(ds_name, seq_key, i, "rgb") for i in indices]
            depth_data = [self.latent_cache.load(ds_name, seq_key, i, "depth") for i in indices]
            if all(x is not None for x in rgb_data) and all(x is not None for x in depth_data):
                out["rgb_latent"] = torch.stack(rgb_data, dim=2).float()      # (2, C, T, h, w)
                out["depth_latent"] = torch.stack(depth_data, dim=2).float()
                out["_cache_hit"] = torch.tensor(True)
            else:
                out["_cache_hit"] = torch.tensor(False)
                # Placeholder only — collate_depth_batch drops this key whenever any
                # sample in the batch misses, so its shape is never read or stacked.
                T = len(indices)
                out["rgb_latent"] = torch.zeros(2, 1, T, 1, 1)
                out["depth_latent"] = torch.zeros(2, 1, T, 1, 1)

        return out
    
    def _cache_hit_fast_dict(self, indices, ds_name, seq_key, rgb_data, depth_data):
        # Mirrors the keys produced by the slow path so default_collate sees
        # the same schema; values for unread pixel content are zero tensors.
        h, w = self.resize
        T = self.clip_len
        zeros_3thw = torch.zeros(3, T, h, w)
        out = {
            "rgbs_clip": zeros_3thw,
            "depths_clip_norm_inv_rgb": zeros_3thw.clone(),
            "depth_canon_scaler": torch.ones(1, T, 1, 1),
            "rgb_latent": torch.stack(rgb_data, dim=2).float(),
            "depth_latent": torch.stack(depth_data, dim=2).float(),
            "_cache_hit": torch.tensor(True),
            "_cache_ds_name": ds_name,
            "_cache_seq_name": seq_key,
            "_cache_frame_indices": torch.tensor(indices, dtype=torch.long),
        }
        return out

    def _load_image(self, path: Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)  # cv2.imread fails silently; OSError so MixedVideoDataset retries
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32)
        arr = normalize_rgb_images(arr) if self.rgb_norm else np.clip(arr / 255.0, 0.0, 1.0)
        return torch.from_numpy(arr).permute(2, 0, 1)  # 3×H×W


class MPISintelDataset(VideoDepthDataset):
    def __init__(self,
                 rgbs_root: str,
                 depths_root: str,
                 num_samples_per_epoch=10000,
                 clip_len=16,
                 stride=(1, 3),
                 resize=None,
                 interpolation="BILINEAR",
                 transform=None,
                 depth_canon: bool = True,
                 depth_canon_focal_px: float = 1024.0,
                 depth_norm=True,
                 depth_norm_mode="percentile",
                 depth_norm_far_plane=655.00,
                 depth_norm_k_compression=1.0,
                 depth_norm_p_lo=0.1,
                 depth_norm_p_hi=99.0,
                 depth_norm_inv_mode="flip",
                 curve_correction_mode: str = "none",):
        self.rgb_root = Path(rgbs_root)
        self.depth_root = Path(depths_root)
        super().__init__(num_samples_per_epoch=num_samples_per_epoch,
                         clip_len=clip_len,
                         stride=stride,
                         resize=resize,
                         interpolation=interpolation,
                         transform=transform,
                         depth_canon=depth_canon,
                         depth_canon_focal_px=depth_canon_focal_px,
                         depth_norm=depth_norm,
                         depth_norm_mode=depth_norm_mode,
                         depth_norm_far_plane=depth_norm_far_plane,
                         depth_norm_k_compression=depth_norm_k_compression,
                         depth_norm_p_lo=depth_norm_p_lo,
                         depth_norm_p_hi=depth_norm_p_hi,
                         depth_norm_inv_mode=depth_norm_inv_mode,
                         curve_correction_mode=curve_correction_mode,)

    def build_indices(self) -> List[Tuple[Path, Path, int, int, List[int] | None]]:
        rgb_scenes = natsorted([p for p in self.rgb_root.iterdir() if p.is_dir()])
        depth_scenes = natsorted([p for p in self.depth_root.iterdir() if p.is_dir()])
        assert len(rgb_scenes) == len(depth_scenes), "Scene count mismatch between RGB and depth."

        # ── clamp stride.upper so the shortest scene can still yield a clip ──
        if rgb_scenes:
            shortest = min(len(list(s.glob("*.png"))) for s in rgb_scenes)
            max_ok = (shortest - 1) // (self.clip_len - 1)
            if max_ok < 1:
                raise ValueError("All Sintel scenes are shorter than clip_len")
            if self.stride[1] > max_ok:
                self.stride = (self.stride[0], max_ok)

        index: list[tuple] = []
        required = 1 + (self.clip_len - 1) * self.stride[1]

        for rgb_scene, depth_scene in zip(rgb_scenes, depth_scenes):
            rgb_frames = natsorted(rgb_scene.glob("*.png"))
            depth_frames = natsorted(depth_scene.glob("*.dpt"))
            assert len(rgb_frames) == len(depth_frames), f"Frame mismatch in {rgb_scene.name}"
            if len(rgb_frames) < required:
                continue
            first_index = int(rgb_frames[0].stem.split('_')[-1])
            index.append((rgb_scene, depth_scene, len(rgb_frames), first_index, None))

        return index

    def load_rgbs(self, rgb_scene: Path, indices: List[int]) -> torch.Tensor:
        rgbs_clip = torch.stack([self._load_image(rgb_scene / f"frame_{i:04d}.png") for i in indices], dim=1) # 3×F×H×W
        return rgbs_clip

    def load_depths(self, depth_scene: Path, indices: List[int]) -> torch.Tensor:
        depths_clip = torch.stack([self._load_dpt(depth_scene / f"frame_{i:04d}.dpt") for i in indices], dim=1) # 1×F×H×W
        return depths_clip

    def _load_dpt(self, path: Path) -> torch.Tensor:
        depth = depth_read(str(path))
        return torch.from_numpy(depth).unsqueeze(0).float()

    def get_fx_and_width(self, rgb_scene: Path, depth_scene: Path, indices: List[int]) -> Tuple[List[float], int] | None:
        # MPI-Sintel default: focal ≈ 1120 px at original width 1024.
        fx = 1120.0
        W_orig = 1024
        return [fx] * len(indices), W_orig


class TartanAirDataset(VideoDepthDataset):
    def __init__(self,
                 root: str,
                 num_samples_per_epoch=10000,
                 clip_len=16,
                 stride=(1, 3),
                 resize=None,
                 interpolation="BILINEAR",
                 transform=None,
                 depth_canon: bool = True,
                 depth_canon_focal_px: float = 1024.0,
                 depth_norm=True,
                 depth_norm_mode="percentile",
                 depth_norm_far_plane=655.00,
                 depth_norm_k_compression=1.0,
                 depth_norm_p_lo=0.1,
                 depth_norm_p_hi=99.0,
                 depth_norm_inv_mode="flip",
                 curve_correction_mode: str = "none",):
        self.root = Path(root)
        super().__init__(num_samples_per_epoch=num_samples_per_epoch,
                         clip_len=clip_len,
                         stride=stride,
                         resize=resize,
                         interpolation=interpolation,
                         transform=transform,
                         depth_canon=depth_canon,
                         depth_canon_focal_px=depth_canon_focal_px,
                         depth_norm=depth_norm,
                         depth_norm_mode=depth_norm_mode,
                         depth_norm_far_plane=depth_norm_far_plane,
                         depth_norm_k_compression=depth_norm_k_compression,
                         depth_norm_p_lo=depth_norm_p_lo,
                         depth_norm_p_hi=depth_norm_p_hi,
                         depth_norm_inv_mode=depth_norm_inv_mode,
                         curve_correction_mode=curve_correction_mode,)
        

    def build_indices(self) -> List[Tuple[Path, Path, int, int, List[int]]]:
        scenes = natsorted([p for p in self.root.iterdir() if p.is_dir()])

        # ── find shortest shot to clamp stride.upper ──
        shortest = float("inf")
        for scene in scenes:
            hard_dir = scene / "Hard"
            if not hard_dir.exists():
                continue
            for shot in hard_dir.iterdir():
                rgb_dir = shot / "image_left"
                if rgb_dir.is_dir():
                    n_frames = len(list(rgb_dir.glob("*_left.png")))
                    if n_frames:
                        shortest = min(shortest, n_frames)

        if shortest == float("inf"):
            return []

        max_ok = (shortest - 1) // (self.clip_len - 1)
        if max_ok < 1:
            raise ValueError("No TartanAir shot long enough for one clip")
        if self.stride[1] > max_ok:
            self.stride = (self.stride[0], max_ok)

        index: list[tuple] = []
        required = 1 + (self.clip_len - 1) * self.stride[1]

        for scene in scenes:
            hard_dir = scene / "Hard"
            if not hard_dir.exists():
                continue
            for shot in natsorted([s for s in hard_dir.iterdir() if s.is_dir()]):
                rgb_dir = shot / "image_left"
                depth_dir = shot / "depth_left"
                if not rgb_dir.exists() or not depth_dir.exists():
                    continue

                rgb_files = natsorted(rgb_dir.glob("*_left.png"))
                depth_files = natsorted(depth_dir.glob("*_left_depth.npy"))
                if len(rgb_files) < required or len(depth_files) < required:
                    continue

                def extract_index(f: Path) -> int:
                    return int(re.search(r"(\d+)_left", f.stem).group(1))

                common_keys = sorted(
                    set(map(extract_index, rgb_files)) & set(map(extract_index, depth_files))
                )
                if len(common_keys) < required:
                    continue

                first_idx = common_keys[0]
                index.append((rgb_dir, depth_dir, len(common_keys), first_idx, common_keys))

        return index

    def load_rgbs(self, rgb_dir: Path, indices: List[int]) -> torch.Tensor:
        rgbs_clip = torch.stack([self._load_image(rgb_dir / f"{i:06d}_left.png") for i in indices], dim=1) # 3×F×H×W
        return rgbs_clip

    def load_depths(self, depth_dir: Path, indices: List[int]) -> torch.Tensor:
        depths_clip = torch.stack([self._load_npy(depth_dir / f"{i:06d}_left_depth.npy") for i in indices], dim=1) # 1×F×H×W
        return depths_clip

    def _load_npy(self, path: Path) -> torch.Tensor:
        depth = np.load(str(path))
        return torch.from_numpy(depth).unsqueeze(0).float()

    def get_fx_and_width(self, rgb_scene: Path, depth_scene: Path, indices: List[int]) -> Tuple[List[float], int] | None:
        # TartanAir: fx = 320 px, original width = 640.
        return [320.0] * len(indices), 640


class SpringDataset(VideoDepthDataset):
    def __init__(self,
                 root: str,
                 num_samples_per_epoch=10000,
                 clip_len=16,
                 stride=(1, 3),
                 resize=None,
                 interpolation="BILINEAR",
                 transform=None,
                 depth_canon: bool = True,
                 depth_canon_focal_px: float = 1024.0,
                 depth_norm=True,
                 depth_norm_mode="percentile",
                 depth_norm_far_plane=655.00,
                 depth_norm_k_compression=1.0,
                 depth_norm_p_lo=0.1,
                 depth_norm_p_hi=99.0,
                 depth_norm_inv_mode="flip",
                 curve_correction_mode: str = "none",):
        self.root = Path(root)
        self.stereo_camera_baseline_distance = 0.065  # meters
        super().__init__(num_samples_per_epoch=num_samples_per_epoch,
                         clip_len=clip_len,
                         stride=stride,
                         resize=resize,
                         interpolation=interpolation,
                         transform=transform,
                         depth_canon=depth_canon,
                         depth_canon_focal_px=depth_canon_focal_px,
                         depth_norm=depth_norm,
                         depth_norm_mode=depth_norm_mode,
                         depth_norm_far_plane=depth_norm_far_plane,
                         depth_norm_k_compression=depth_norm_k_compression,
                         depth_norm_p_lo=depth_norm_p_lo,
                         depth_norm_p_hi=depth_norm_p_hi,
                         depth_norm_inv_mode=depth_norm_inv_mode,
                         curve_correction_mode=curve_correction_mode,)

    def build_indices(self) -> List[Tuple[Path, Path, int, int, List[int]]]:
        scene_dirs = natsorted([p for p in self.root.iterdir() if p.is_dir()])

        # ── clamp stride.upper based on shortest scene ──
        shortest = float("inf")
        for scene in scene_dirs:
            n_frames = len(list((scene / "frame_left").glob("frame_left_*.png")))
            if n_frames:
                shortest = min(shortest, n_frames)

        if shortest == float("inf"):
            return []

        max_ok = (shortest - 1) // (self.clip_len - 1)
        if max_ok < 1:
            raise ValueError("All Spring scenes too short for clip_len")
        if self.stride[1] > max_ok:
            self.stride = (self.stride[0], max_ok)

        index: list[tuple] = []
        required = 1 + (self.clip_len - 1) * self.stride[1]

        for scene in scene_dirs:
            rgb_dir = scene / "frame_left"
            disp_dir = scene / "disp1_left"
            cam_file = scene / "cam_data" / "intrinsics.txt"
            if not rgb_dir.exists() or not disp_dir.exists() or not cam_file.exists():
                continue

            rgb_files = natsorted(rgb_dir.glob("frame_left_*.png"))
            disp_files = natsorted(disp_dir.glob("disp1_left_*.dsp5"))
            if len(rgb_files) < required or len(disp_files) < required:
                continue

            def extract_index(f: Path) -> int:
                return int(re.search(r"(\d+)$", f.stem).group(1))

            common_keys = sorted(
                set(map(extract_index, rgb_files)) & set(map(extract_index, disp_files))
            )
            if len(common_keys) < required:
                continue

            with open(cam_file, "r") as f:
                intrinsics_lines = f.readlines()
            assert len(intrinsics_lines) == len(rgb_files), \
                f"Mismatch between intrinsics and frames in {scene.name}"

            index.append((rgb_dir, disp_dir, len(common_keys), common_keys[0], common_keys))

        return index

    def load_rgbs(self, rgb_dir: Path, indices: List[int]) -> torch.Tensor:
        rgbs_clip = torch.stack([self._load_image(rgb_dir / f"frame_left_{i:04d}.png") for i in indices], dim=1) # 3×F×H×W
        return rgbs_clip

    def load_depths(self, disp_dir: Path, indices: List[int]) -> torch.Tensor:
        cam_file = disp_dir.parent / "cam_data" / "intrinsics.txt"
        with open(cam_file, "r") as f:
            intrinsics_lines = f.readlines()

        depth_list = []

        for j, i in enumerate(indices):
            disp = self._load_dsp5(disp_dir / f"disp1_left_{i:04d}.dsp5")
            fx = float(intrinsics_lines[j].split()[0])  # j is 0,1,...,clip_len-1

            # How can I compute metric depth from disparity?
            # In general, depth Z is computed from disparity d through Z = fx * B / d, 
            # where fx is the focal length in pixels (given in intrinsics.txt) and 
            # B is the stereo camera baseline distance; 
            # for Spring this is always 0.065m. 
            # Please note that the Spring dataset encodes infinitely distant sky pixels as zero disparity, 
            # leading to infinite values when using the above formula.

            valid_mask = disp > 0
            depth = np.zeros_like(disp, dtype=np.float32)
            depth[valid_mask] = (fx * self.stereo_camera_baseline_distance) / disp[valid_mask]

            # Cap invalid or extreme depth values
            depth[~valid_mask] = self.depth_norm_far_plane
            depth = np.clip(depth, 0.0, self.depth_norm_far_plane)

            depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
            depth_list.append(depth_tensor)

        depths_clip = torch.stack(depth_list, dim=1)  # 1×F×H×W
        return depths_clip

    def _load_dsp5(self, path: Path) -> np.ndarray:
        with h5py.File(path, "r") as f:
            if "disparity" not in f.keys():
                raise IOError(f"File {str(path)} does not have a 'disparity' key. Is this a valid dsp5 file?")
            return f["disparity"][()]

    def get_fx_and_width(self, rgb_scene: Path, depth_scene: Path, indices: List[int]) -> Tuple[List[float], int] | None:
        # Spring intrinsics are per-frame in cam_data/intrinsics.txt
        cam_file = depth_scene.parent / "cam_data" / "intrinsics.txt"
        with open(cam_file, "r") as f:
            intrinsics_lines = f.readlines()
        fx_list = [float(intrinsics_lines[j].split()[0]) for j, _ in enumerate(indices)]
        W_orig = 1920  # Spring left frames are 1920×1080
        return fx_list, W_orig


class SceneNetDataset(VideoDepthDataset):
    def __init__(self,
                 root: str,
                 num_samples_per_epoch: int = 10000,
                 clip_len: int = 16,
                 stride: tuple[int, int] = (1, 3),
                 resize: tuple[int, int] | None = None,
                 interpolation="BILINEAR",
                 transform=None,
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
                 far_value: float = 1000.00):
        self.root = Path(root)
        self.far_value = far_value
        super().__init__(num_samples_per_epoch=num_samples_per_epoch,
                         clip_len=clip_len,
                         stride=stride,
                         resize=resize,
                         interpolation=interpolation,
                         transform=transform,
                         depth_canon=depth_canon,
                         depth_canon_focal_px=depth_canon_focal_px,
                         depth_norm=depth_norm,
                         depth_norm_mode=depth_norm_mode,
                         depth_norm_far_plane=depth_norm_far_plane,
                         depth_norm_k_compression=depth_norm_k_compression,
                         depth_norm_p_lo=depth_norm_p_lo,
                         depth_norm_p_hi=depth_norm_p_hi,
                         depth_norm_inv_mode=depth_norm_inv_mode,
                         curve_correction_mode=curve_correction_mode,)

    def build_indices(self) -> List[Tuple[Path, Path, int, int, List[int]]]:
        """
        SceneNet: every scene is kept as a single clip-source.
        frame_indices  = sorted list of all numeric IDs that actually exist;
        frame_count    = len(frame_indices);
        first_idx      = 0  (unused when frame_indices is provided).
        """
        scene_dirs = natsorted([p for p in self.root.iterdir() if p.is_dir()])

        # ── find the shortest scene length to clamp stride.upper ────────────
        lengths = []
        for sc in scene_dirs:
            ids = sorted({int(re.search(r"(\d+)$", f.stem).group(1))
                        for f in (sc / "photo").glob("*.*")})
            if ids:
                lengths.append(len(ids))
        if not lengths:
            return []

        shortest = min(lengths)
        max_ok   = (shortest - 1) // (self.clip_len - 1)
        if max_ok < 1:
            raise ValueError("All SceneNet scenes shorter than clip_len")
        if self.stride[1] > max_ok:
            self.stride = (self.stride[0], max_ok)

        required = 1 + (self.clip_len - 1) * self.stride[1]
        index: list[tuple] = []

        # ── store every full-length scene -----------------------------------
        for sc in scene_dirs:
            rgb_dir, depth_dir = sc / "photo", sc / "depth"
            if not rgb_dir.exists() or not depth_dir.exists():
                continue

            ids = sorted({int(re.search(r"(\d+)$", f.stem).group(1))
                        for f in rgb_dir.glob("*.*")})
            if len(ids) < required:
                continue

            index.append((rgb_dir, depth_dir, len(ids), 0, ids))

        return index

    # ------------------------------------------------------------------ #
    #  data loaders
    # ------------------------------------------------------------------ #
    def load_rgbs(self, rgb_dir: Path, indices: List[int]) -> torch.Tensor:
        def find_file(i: int) -> Path:
            for ext in (".jpg", ".jpeg", ".png"):
                p = rgb_dir / f"{i}{ext}"
                if p.exists():
                    return p
            raise FileNotFoundError(f"Missing RGB for index {i} in {rgb_dir}")
        return torch.stack([self._load_image(find_file(i)) for i in indices], dim=1)

    def load_depths(self, depth_dir: Path, indices: List[int]) -> torch.Tensor:                                  # fixed “sky / invalid” depth (metres)
        depth_list = []

        for i in indices:
            p = depth_dir / f"{i}.png"                      # 16-bit PNG, millimetres
            depth_u16 = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if depth_u16 is None:
                raise FileNotFoundError(p)

            depth = depth_u16.astype(np.float32) / 1000.0   # → metres
            depth[depth <= EPS] = self.far_value            # patch bad / sky pixels

            depth_list.append(torch.from_numpy(depth).unsqueeze(0))

        return torch.stack(depth_list, dim=1)               # 1×T×H×W
