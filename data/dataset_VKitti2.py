# dataset_vkitti2.py
"""
VirtualKITTI-2 helpers
──────────────────────────────────────────────────────────────────────────────
* VirtualKITTI2  – per-frame loader
* VKitti2Clips   – sliding-window clip loader

RGB  : optional scaling to [-1, 1]
Depth: percentile normalisation + inverse-depth (reciprocal or flip)

Returned keys
─────────────
rgbs_clip        C × F × H × W
rgbs_src_clip    C × F × H × W  (clean target)
depths_clip      1 × F × H × W  (optional)
depth_norm_params {d2,d98}      (only if depth_norm=True)
captions_clip    ""             (placeholder)
"""
from __future__ import annotations
import os, glob
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted

from utils.data import  normalize_rgb_images, normalize_depth

# ╭──────────────────────────────  Frame dataset  ───────────────────────────╮ #
class VirtualKITTI2(Dataset):
    _SCENES   = ("01", "02", "06", "18", "20")
    _VARIANTS = (
        "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right",
        "clone", "fog", "morning", "overcast", "rain", "sunset",
    )

    def __init__(
        self,
        root        : str,
        variants    : Tuple[str, ...] | None = None,
        cameras     : Tuple[int, ...] | None = (0, 1),
        resize      : Optional[Tuple[int,int]] = None,
        interpolation: int = Image.BILINEAR,
        frame_stride: int  = 1,
        rgb_norm    : bool = True,
    ):
        if frame_stride < 1:
            raise ValueError("frame_stride must be ≥1")

        self.root      = os.path.abspath(root)
        self.variants  = variants or self._VARIANTS
        self.cameras   = cameras
        self.resize    = resize
        self.interp    = interpolation
        self.stride    = frame_stride
        self.rgb_norm  = rgb_norm
        self.index: List[Dict] = []
        self._build_indices()

    # --------------------------------------------------------------------- #
    def _build_indices(self):
        rgb_pat, dep_pat = "frames/rgb/Camera_{:d}/rgb_*.jpg", "frames/depth/Camera_{:d}/depth_*.png"
        for scene in self._SCENES:
            s_dir = f"Scene{scene}"
            for var in self.variants:
                for cam in self.cameras:
                    rgbs = natsorted(glob.glob(os.path.join(self.root, s_dir, var, rgb_pat.format(cam))))
                    deps = natsorted(glob.glob(os.path.join(self.root, s_dir, var, dep_pat.format(cam))))
                    for r,d in zip(rgbs[::self.stride], deps[::self.stride]):
                        fid = int(os.path.basename(r)[4:9])
                        self.index.append({
                                            "rgb_path":  r,       # ← was "rgb": r
                                            "depth_path": d,      # ← was "depth": d
                                            "scene": s_dir, "variant": var, "camera": cam, "frame_id": fid,
                                    })

    # --------------------------------------------------------------------- #
    def _load_rgb(self, p: str) -> torch.Tensor:
        img = Image.open(p).convert("RGB")
        if self.resize:
            h,w = self.resize; img = img.resize((w,h), self.interp)
        arr = np.asarray(img, np.float32)
        if self.rgb_norm :
            arr = normalize_rgb_images(arr)
        else:
            arr = np.clip((arr / 255.0), 0.0, 1.0)

        return torch.from_numpy(arr).permute(2,0,1)           # 3×H×W

    def _load_depth(self, p: str) -> torch.Tensor:
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)               # uint16 cm
        if d is None: raise IOError(p)
        if self.resize:
            h,w = self.resize; d = cv2.resize(d,(w,h),cv2.INTER_NEAREST)
        return torch.from_numpy(d.astype(np.float32)/100.0).unsqueeze(0)  # m

    # --------------------------------------------------------------------- #
    def __len__(self): return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.index[idx]
        rgb  = self._load_rgb(meta["rgb_path"])     # key changed
        depth= self._load_depth(meta["depth_path"]) # key changed
        return {"rgb": rgb, "depth": depth, **meta}
# ╰──────────────────────────────────────────────────────────────────────────╯ #


# ╭──────────────────────────────  Clip dataset  ────────────────────────────╮ #
class VKitti2Clips(Dataset):
    def __init__(
        self,
        root         : str,
        clip_len     : int  = 16,
        variants     : Tuple[str,...] | None = None,
        cameras      : Tuple[int,...] = (0,1),
        resize       : Optional[Tuple[int,int]] = None,
        interpolation: int = Image.BILINEAR,
        frame_stride : int = 1,
        include_depth      : bool = False,
        rgb_norm           : bool = True,
        depth_norm         : bool = False,
        inverse_depth_mode : Optional[str] = None,   # None | "reciprocal" | "flip"
    ):
        if inverse_depth_mode not in (None,"reciprocal","flip"):
            raise ValueError("inverse_depth_mode must be None,'reciprocal','flip'")

        self.base = VirtualKITTI2(root, variants, cameras,
                                  resize, interpolation,
                                  frame_stride, rgb_norm)

        self.L            = clip_len
        self.include_depth= include_depth
        self.depth_norm   = depth_norm
        self.inv_mode     = inverse_depth_mode

        # bucket frames by (scene,variant,cam)
        buckets: Dict[Tuple,str,List[int]] = {}
        for fid,meta in enumerate(self.base.index):
            buckets.setdefault((meta["scene"],meta["variant"],meta["camera"]),[]).append(fid)

        self.clip_idxs: List[List[int]] = []
        for ids in buckets.values():
            for s in range(0, len(ids)-clip_len+1):
                self.clip_idxs.append(ids[s:s+clip_len])

    # --------------------------------------------------------------------- #
    def __len__(self): return len(self.clip_idxs)

    def __getitem__(self, idx:int) -> Dict:
        fids = self.clip_idxs[idx]

        # list_aaa = [self.base[i]["rgb"] for i in fids]
        rgbs = torch.stack([self.base[i]["rgb"] for i in fids], dim=1)   # 3×F×H×W
        out  = {"rgbs_clip": rgbs,
                "rgbs_src_clip": rgbs.clone(),
                "captions_clip": ""}

        if self.include_depth:
            d_raw = torch.stack([self.base[i]["depth"] for i in fids], dim=1)  # 1×F×H×W

            out["depths_clip_raw"] = d_raw
            d_proc = d_raw

            # (1) percentile normalisation
            if self.depth_norm:
                d_proc, (d2,d98) = normalize_depth(d_proc)
                out["depths_clip_norm_params"] = {"d2": d2, "d98": d98}
                out["depths_clip_norm"] = d_proc

            # (2) flip after normalisation
            if self.inv_mode == "flip":
                if not self.depth_norm:
                    raise ValueError("'flip' requires depth_norm=True")
                d_proc = -d_proc
                out["depths_clip_norm_inv"] = d_proc

                d_proc = d_proc.repeat(3, 1, 1, 1)
                out["depths_clip_norm_inv_rgb"] = d_proc

            out["depths_clip"] = d_proc

        return out
# ╰──────────────────────────────────────────────────────────────────────────╯ #
