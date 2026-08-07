"""
WebDataset-format tar shards for VideoDepthDiffusion.

Single shard set, sequentially streamed
---------------------------------------
Every sequence is one WebDataset sample.  Pixels and (after baking) the VAE
latents live in the **same** tar member group, so a single sequential stream
carries everything — no random access, no offset index for latents, no
background shard-append thread.

Two-phase workflow
------------------
1. **warm + bake** (one-time, offline-ish):
   - ``prepare_image_shards`` packs pixels into raw image shards.
   - A *full-sequence* cache pass (``_wds_full_sequence`` + ``first_epoch_cache_latent_only``)
     encodes every frame of every sequence into the per-file ``LatentCache``.
   - ``bake_latents`` copies each raw image shard verbatim and folds the cached
     latents in as extra members, writing a combined shard set to ``<dir>_baked``.
2. **stream** (training): the combined ``*_baked`` shards are streamed; a random
   clip is sliced from each sample and its latents come along for free
   (``_wds_clip_select`` slices both).  Sequences that were never cached are
   written pixels-only and fall back to online VAE encoding.

Raw image shard sample keys (webdataset dot-split convention):
  {seq_id}.meta.json   – scene metadata
  {seq_id}.rgbs.pth    – uint8  (N, 3, H, W) at target resolution
  {seq_id}.depths.pth  – float32 (N, H, W) resized raw metric depths (pre-norm)

Baked shard sample adds:
  {seq_id}.rgb_lats.pth – float16 (N, 2, C, h, w)
  {seq_id}.dep_lats.pth – float16 (N, 2, C, h, w)
"""

from __future__ import annotations

import csv
import io
import json
import random
import tarfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _t2b(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.contiguous(), buf)
    return buf.getvalue()


def _b2t(data: bytes) -> torch.Tensor:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _scan_offsets(path: str) -> Dict[str, Tuple[int, int]]:
    """Return {member_name: (offset_data, size)} by scanning a tar."""
    out: Dict[str, Tuple[int, int]] = {}
    with tarfile.open(path, "r") as tf:
        for m in tf.getmembers():
            out[m.name] = (m.offset_data, m.size)
    return out


def _read_raw(path: str, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


# ─────────────────────────────────────────────────────────────────────────────
# ShardIndex  (image / baked shard membership, used at *build* time only)
# ─────────────────────────────────────────────────────────────────────────────

class ShardIndex:
    """
    CSV-backed offset index: (dataset, seq_key) → (shard_path, {member: (offset, size)}).

    Used to make shard building idempotent and to copy member bytes verbatim
    during the bake.  Streaming never consults it (webdataset reads tars directly).
    """

    FIELDS = ["dataset", "seq_key", "shard_path", "member_name", "offset", "size"]

    def __init__(self, index_path: Path):
        self.path = index_path
        self._lock = threading.Lock()
        # (ds, seq_key) → (shard_path, {member_name: (offset, size)})
        self._data: Dict[Tuple[str, str], Tuple[str, Dict[str, Tuple[int, int]]]] = {}
        if index_path.exists():
            self._load()

    def _load(self):
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                k = (row["dataset"], row["seq_key"])
                shard = row["shard_path"]
                if k not in self._data:
                    self._data[k] = (shard, {})
                self._data[k][1][row["member_name"]] = (int(row["offset"]), int(row["size"]))

    def contains(self, dataset: str, seq_key: str) -> bool:
        return (dataset, seq_key) in self._data

    def get(self, dataset: str, seq_key: str) -> Optional[Tuple[str, Dict[str, Tuple[int, int]]]]:
        return self._data.get((dataset, seq_key))

    def items(self):
        return self._data.items()

    def add(self, dataset: str, seq_key: str, shard_path: str,
            members: Dict[str, Tuple[int, int]]):
        with self._lock:
            self._data[(dataset, seq_key)] = (shard_path, members)
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.FIELDS)
                if new_file:
                    w.writeheader()
                for name, (off, sz) in members.items():
                    w.writerow({"dataset": dataset, "seq_key": seq_key,
                                "shard_path": shard_path, "member_name": name,
                                "offset": off, "size": sz})

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    def shard_paths(self) -> List[str]:
        return sorted({v[0] for v in self._data.values()})


# ─────────────────────────────────────────────────────────────────────────────
# WebDatasetShardManager
# ─────────────────────────────────────────────────────────────────────────────

class WebDatasetShardManager:
    """
    Builds and serves WebDataset tar shards.

    Raw image shards live in ``image_dir``; baked (pixels + latents) shards in
    ``baked_dir``.  Both are read by sequential webdataset streaming — there is
    no random access.

    Safe to pickle to DataLoader workers: holds only paths and ShardIndex
    objects (which manage their own lock).
    """

    _SEQ_FMT = "{:08d}"

    def __init__(
        self,
        image_shard_dir: str,
        version_tag: str,
        max_shard_size_gb: float = 2.0,
    ):
        root = Path(image_shard_dir)
        self.image_dir = root / version_tag
        self.baked_dir = root / f"{version_tag}_baked"
        self.version_tag = version_tag
        self.max_shard_bytes = int(max_shard_size_gb * 1024 ** 3)

        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.baked_dir.mkdir(parents=True, exist_ok=True)

        self.image_index = ShardIndex(self.image_dir / "index.csv")
        self.baked_index = ShardIndex(self.baked_dir / "index.csv")

        self._img_seq_ctr = len(self.image_index)
        self._baked_seq_ctr = len(self.baked_index)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _next_img_shard(self) -> Path:
        idx = len(sorted(self.image_dir.glob("image_shard_*.tar")))
        return self.image_dir / f"image_shard_{idx:06d}.tar"

    def _next_baked_shard(self) -> Path:
        idx = len(sorted(self.baked_dir.glob("baked_shard_*.tar")))
        return self.baked_dir / f"baked_shard_{idx:06d}.tar"

    def image_shards_ready(self) -> bool:
        return len(self.image_index) > 0

    def baked_shards_ready(self) -> bool:
        return len(self.baked_index) > 0

    # ── Image sharding (pixels only) ───────────────────────────────────────────

    def prepare_image_shards(
        self,
        datasets,          # list[VideoDepthDataset]
        resize: Tuple[int, int],
        show_progress: bool = True,
    ):
        """
        Pack every sequence from *datasets* into image shard tars.
        Idempotent: sequences already in the index are skipped.
        Must be called on rank-0 before training.
        """
        from utils.images import resize_rgbs_clip  # local import to avoid circular

        H, W = resize
        total = sum(len(ds.index) for ds in datasets)
        pbar = tqdm(total=total, desc="Building image shards", unit="seq",
                    dynamic_ncols=True) if show_progress else None

        cur_path: Optional[Path] = None
        cur_tf: Optional[tarfile.TarFile] = None
        cur_size: int = 0
        pending: list = []   # (sid, ds_name, seq_key) for index update after shard close

        def _flush():
            nonlocal cur_tf, cur_path, cur_size
            if cur_tf is None:
                return
            cur_tf.close()
            offsets = _scan_offsets(str(cur_path))
            for p in pending:
                members = {k: v for k, v in offsets.items()
                           if k.startswith(p["sid"] + ".")}
                self.image_index.add(p["ds"], p["seq"], str(cur_path), members)
            pending.clear()
            cur_tf = None
            cur_path = None
            cur_size = 0

        def _open():
            nonlocal cur_path, cur_tf, cur_size
            cur_path = self._next_img_shard()
            cur_tf = tarfile.open(cur_path, "w")
            cur_size = 0

        _open()

        try:
            for ds in datasets:
                ds_name = ds.dataset_name
                for entry in ds.index:
                    rgb_scene, depth_scene, frame_count, first_idx, frame_indices = entry
                    seq_key = ds._scene_key(rgb_scene)
                    if pbar:
                        pbar.update(1)
                    if self.image_index.contains(ds_name, seq_key):
                        continue

                    all_idx = (list(frame_indices) if frame_indices is not None
                               else list(range(first_idx, first_idx + frame_count)))

                    try:
                        rgbs = ds.load_rgbs(rgb_scene, all_idx)       # (3, N, H0, W0)
                        depths = ds.load_depths(depth_scene, all_idx)  # (1, N, H0, W0)
                    except Exception as exc:
                        if pbar:
                            pbar.write(f"  skip {ds_name}/{seq_key}: {exc}")
                        continue

                    # Resize RGB → uint8 (N, 3, H, W)
                    rgbs_r = resize_rgbs_clip(rgbs, (H, W))  # (3, N, H, W)
                    rgbs_u8 = (rgbs_r.clamp(0, 1) * 255).byte().permute(1, 0, 2, 3).contiguous()

                    # Resize depths → float32 (N, H, W)
                    d_in = depths.squeeze(0).unsqueeze(1)  # (N, 1, H0, W0)
                    d_r = F.interpolate(d_in, size=(H, W),
                                        mode="bilinear", align_corners=False)
                    depths_r = d_r.squeeze(1).contiguous()  # (N, H, W)

                    # Canon scalers per frame
                    fxW = ds.get_fx_and_width(rgb_scene, depth_scene, all_idx)
                    if fxW is not None and ds.depth_canon:
                        fx_list, W0 = fxW
                        canon = [float(ds.depth_canon_focal_px * W0) / (float(fx) * W + 1e-8)
                                 for fx in fx_list]
                    else:
                        canon = [1.0] * len(all_idx)

                    meta = {
                        "ds": ds_name,
                        "seq_key": seq_key,
                        "n_frames": len(all_idx),
                        "first_idx": first_idx if frame_indices is None else None,
                        "contiguous": frame_indices is None,
                        "frame_indices": frame_indices,
                        "canon_scalers": canon,
                    }

                    sid = self._SEQ_FMT.format(self._img_seq_ctr)
                    self._img_seq_ctr += 1

                    entries = {
                        f"{sid}.meta.json": json.dumps(meta).encode(),
                        f"{sid}.rgbs.pth":  _t2b(rgbs_u8),
                        f"{sid}.depths.pth": _t2b(depths_r),
                    }
                    entry_bytes = sum(len(v) for v in entries.values())

                    if cur_size > 0 and cur_size + entry_bytes > self.max_shard_bytes:
                        _flush()
                        _open()

                    for name, data in entries.items():
                        ti = tarfile.TarInfo(name=name)
                        ti.size = len(data)
                        cur_tf.addfile(ti, io.BytesIO(data))
                    cur_size += entry_bytes
                    pending.append({"sid": sid, "ds": ds_name, "seq": seq_key})

        finally:
            _flush()
            if pbar:
                pbar.close()

    # ── Bake: fold per-file latents into combined shards ───────────────────────

    @staticmethod
    def _meta_sid(members: Dict[str, Tuple[int, int]]) -> Optional[str]:
        for name in members:
            if name.endswith(".meta.json"):
                return name[: -len(".meta.json")]
        return None

    def bake_latents(self, latent_cache, show_progress: bool = True,
                     require_complete: bool = True) -> Dict[str, int]:
        """
        Copy every raw image shard sequence into the baked shard set, folding in
        the per-file VAE latents as extra members.  Idempotent: sequences already
        baked are skipped.  Returns {"baked", "skipped"}.

        Coverage guard
        --------------
        The model decides per *batch* whether to use cached latents
        (``cache_hit = batch["_cache_hit"].all()``), and baked-hit samples carry
        zeroed pixels.  A baked set that mixed with-latents and pixels-only
        sequences would therefore train the baked samples on ``encode(zeros)``
        once shuffle put them in the same batch.  To keep the set uniformly
        all-hit, a partially-cached bake is by default refused (``require_complete``):
        re-run the full-sequence cache pass first.  With ``require_complete=False``
        the uncovered sequences are *skipped* (not written) rather than mixed in.

        Must be called on rank-0 after the full-sequence cache pass has been
        flushed.  The combined shards are written to a *new* directory
        (``baked_dir``), so no shard that streaming workers might be reading is
        ever mutated in place.
        """
        to_do = sorted(k for k, _ in self.image_index.items()
                       if not self.baked_index.contains(*k))

        # ── Pre-scan coverage *before* writing anything ───────────────────────
        # (cheap: one tiny meta read + per-frame existence stat per sequence).
        planned: list = []        # (ds, seq, shard_path, members, sid, fis)
        incomplete: list = []
        for ds_name, seq_key in to_do:
            shard_path, members = self.image_index.get(ds_name, seq_key)
            sid = self._meta_sid(members)
            if sid is None:
                continue
            meta = json.loads(_read_raw(shard_path, *members[f"{sid}.meta.json"]))
            fis = meta.get("frame_indices")
            if fis is None:
                fis = list(range(meta["first_idx"], meta["first_idx"] + meta["n_frames"]))
            covered = all(latent_cache.path(ds_name, seq_key, fi, k).exists()
                          for fi in fis for k in ("rgb", "depth"))
            (planned if covered else incomplete).append(
                (ds_name, seq_key, shard_path, members, sid, fis))

        if require_complete and incomplete:
            sample = [(d, s) for d, s, *_ in incomplete[:3]]
            raise ValueError(
                f"bake aborted: {len(incomplete)}/{len(to_do)} sequences are not "
                f"fully cached (e.g. {sample}). Re-run the full-sequence cache pass "
                f"(webdataset.mode=bake + first_epoch_cache_latent_only=true), or "
                f"call with require_complete=False to bake only the covered sequences."
            )

        stats = {"baked": 0, "skipped": len(incomplete)}
        if not planned:
            return stats

        pbar = tqdm(total=len(planned), desc="Baking latents into shards", unit="seq",
                    dynamic_ncols=True) if show_progress else None

        cur_path: Optional[Path] = None
        cur_tf: Optional[tarfile.TarFile] = None
        cur_size: int = 0
        pending: list = []

        def _flush():
            nonlocal cur_tf, cur_path, cur_size
            if cur_tf is None:
                return
            cur_tf.close()
            offsets = _scan_offsets(str(cur_path))
            for p in pending:
                members = {k: v for k, v in offsets.items()
                           if k.startswith(p["sid"] + ".")}
                self.baked_index.add(p["ds"], p["seq"], str(cur_path), members)
            pending.clear()
            cur_tf = None
            cur_path = None
            cur_size = 0

        def _open():
            nonlocal cur_path, cur_tf, cur_size
            cur_path = self._next_baked_shard()
            cur_tf = tarfile.open(cur_path, "w")
            cur_size = 0

        _open()

        try:
            for ds_name, seq_key, shard_path, members, sid, fis in planned:
                if pbar:
                    pbar.update(1)

                rgb_list = [latent_cache.load(ds_name, seq_key, fi, "rgb") for fi in fis]
                dep_list = [latent_cache.load(ds_name, seq_key, fi, "depth") for fi in fis]
                if None in rgb_list or None in dep_list:
                    # Existed at scan time but failed to load (corrupt/raced) —
                    # skip rather than write a partial sequence.
                    stats["skipped"] += 1
                    continue

                # Copy the original pixel members verbatim (no re-encode) + latents.
                entries: Dict[str, bytes] = {
                    name: _read_raw(shard_path, off, sz)
                    for name, (off, sz) in members.items()
                }
                entries[f"{sid}.rgb_lats.pth"] = _t2b(torch.stack(rgb_list, dim=0).to(torch.float16))
                entries[f"{sid}.dep_lats.pth"] = _t2b(torch.stack(dep_list, dim=0).to(torch.float16))

                entry_bytes = sum(len(v) for v in entries.values())
                if cur_size > 0 and cur_size + entry_bytes > self.max_shard_bytes:
                    _flush()
                    _open()

                for name, data in entries.items():
                    ti = tarfile.TarInfo(name=name)
                    ti.size = len(data)
                    cur_tf.addfile(ti, io.BytesIO(data))
                cur_size += entry_bytes
                pending.append({"sid": sid, "ds": ds_name, "seq": seq_key})
                stats["baked"] += 1
        finally:
            _flush()
            if pbar:
                pbar.close()

        return stats

    # ── WebDataset shard URL lists ─────────────────────────────────────────────

    @staticmethod
    def _split_urls(all_shards: List[str], rank: int, world_size: int) -> List[str]:
        # Interleaved split distributes shards evenly across ranks and preserves
        # approximate per-rank data diversity even without reshuffle between epochs.
        return all_shards[rank::world_size]

    def image_shard_urls(self, rank: int = 0, world_size: int = 1) -> List[str]:
        all_shards = sorted(str(p) for p in self.image_dir.glob("image_shard_*.tar"))
        return self._split_urls(all_shards, rank, world_size)

    def baked_shard_urls(self, rank: int = 0, world_size: int = 1) -> List[str]:
        all_shards = sorted(str(p) for p in self.baked_dir.glob("baked_shard_*.tar"))
        return self._split_urls(all_shards, rank, world_size)


# ─────────────────────────────────────────────────────────────────────────────
# WebDataset process functions
# ─────────────────────────────────────────────────────────────────────────────

def _wds_decode_pth(sample: dict) -> dict:
    """Decode bytes values whose key ends in .pth into torch tensors."""
    for k in list(sample.keys()):
        if k.endswith(".pth") and isinstance(sample[k], bytes):
            sample[k] = _b2t(sample[k])
    return sample


def _resolve_meta(sample: dict) -> dict:
    raw_meta = sample.get("meta.json")
    if isinstance(raw_meta, (bytes, bytearray)):
        raw_meta = json.loads(raw_meta.decode())
    return raw_meta


def _attach_clip(sample: dict, meta: dict, positions: List[int],
                 clip_orig_indices: List[int]) -> dict:
    """Slice pixels (and baked latents, if present) by ``positions``."""
    sample["_clip_rgbs_u8"] = sample["rgbs.pth"][positions]
    sample["_clip_depths"] = sample["depths.pth"][positions]
    sample["_clip_positions"] = positions
    sample["_clip_orig_indices"] = clip_orig_indices
    sample["_meta"] = meta

    # Baked latents ride in the same sample.  Present → slice and flag a hit.
    if "rgb_lats.pth" in sample and "dep_lats.pth" in sample:
        sample["_clip_rgb_lat"] = sample["rgb_lats.pth"][positions]   # (T,2,C,h,w)
        sample["_clip_dep_lat"] = sample["dep_lats.pth"][positions]
        sample["_baked_latent_present"] = True
    else:
        sample["_baked_latent_present"] = False
    return sample


def _wds_clip_select(sample: dict, clip_len: int, stride: Tuple[int, int]) -> Optional[dict]:
    """
    Select a random T-frame clip from a (streamed) shard sample and slice every
    per-frame tensor — pixels and baked latents — by the same positions.
    Returns None to skip sequences too short for the given stride.
    """
    meta = _resolve_meta(sample)
    N = meta["n_frames"]
    stride_lo, stride_hi = stride

    max_start = N - 1 - (clip_len - 1) * stride_hi
    if max_start < 0:
        return None

    offset = random.randint(0, max_start)
    # torch.randint(lo, hi, …) is exclusive of hi — replicate exactly
    stride_seq = [
        random.randint(stride_lo, stride_hi - 1) if stride_lo != stride_hi else stride_lo
        for _ in range(clip_len - 1)
    ]

    frame_indices_list = meta.get("frame_indices")
    first_idx = meta.get("first_idx") or 0

    pos = offset
    positions = [pos]
    for s in stride_seq:
        pos += s
        if pos >= N:
            return None
        positions.append(pos)

    if frame_indices_list is not None:
        clip_orig_indices = [frame_indices_list[p] for p in positions]
    else:
        clip_orig_indices = [first_idx + p for p in positions]

    return _attach_clip(sample, meta, positions, clip_orig_indices)


def _wds_full_sequence(sample: dict) -> Optional[dict]:
    """
    Emit the *entire* sequence (every frame) instead of a random clip.  Used by
    the full-sequence cache pass so ``_write_latent_cache`` covers every frame of
    every sequence → a guaranteed-complete bake.  Raw image shards carry no
    latents, so this always takes the pixel path.
    """
    meta = _resolve_meta(sample)
    N = meta["n_frames"]
    positions = list(range(N))

    frame_indices_list = meta.get("frame_indices")
    first_idx = meta.get("first_idx") or 0
    if frame_indices_list is not None:
        clip_orig_indices = list(frame_indices_list)
    else:
        clip_orig_indices = [first_idx + p for p in positions]

    return _attach_clip(sample, meta, positions, clip_orig_indices)


def _wds_depth_process(
    sample: dict,
    depth_norm: bool,
    depth_norm_mode: str,
    depth_norm_far_plane: float,
    depth_norm_k_compression: float,
    depth_norm_p_lo: float,
    depth_norm_p_hi: float,
    depth_norm_inv_mode: str,
    curve_correction_mode: str,
) -> dict:
    """
    Apply depth canonical scaling, normalisation, and colour-curve correction.
    On a baked-latent hit, skips expensive pixel processing (training uses the
    latent directly) and returns zero tensors for pixel fields.
    """
    meta = sample["_meta"]
    positions = sample["_clip_positions"]
    T = len(positions)

    # Canon scaler is cheap and always needed for batch consistency
    canon_scalers = meta["canon_scalers"]
    clip_scalers = [canon_scalers[p] for p in positions]
    scaler_t = torch.tensor(clip_scalers, dtype=torch.float32).view(1, -1, 1, 1)

    if sample.get("_baked_latent_present", False):
        rgbs_u8 = sample["_clip_rgbs_u8"]
        _, _, H, W = rgbs_u8.shape
        sample["rgbs_clip"] = torch.zeros(3, T, H, W)
        sample["depths_clip_norm_inv_rgb"] = torch.zeros(3, T, H, W)
        sample["depth_canon_scaler"] = scaler_t
        return sample

    from utils.data import (normalize_depth_percentile, normalize_depth_logarithmic,
                            apply_fusion_color_curve)

    # ── RGB: convert uint8 → float32 normalised [-1, 1] ──────────────────────
    rgbs_u8 = sample["_clip_rgbs_u8"].float()  # (T, 3, H, W)
    rgbs_norm = (rgbs_u8 / 127.5 - 1.0).clamp(-1.0, 1.0)
    rgbs_out = rgbs_norm.permute(1, 0, 2, 3).contiguous()  # (3, T, H, W)

    # ── Depth: canonical scaling ──────────────────────────────────────────────
    depths_raw = sample["_clip_depths"]          # (T, H, W)
    depths_t = depths_raw.unsqueeze(0)            # (1, T, H, W)  (batch of 1)
    d_proc = depths_t * scaler_t                  # (1, T, H, W)

    # ── Depth: normalisation ──────────────────────────────────────────────────
    if depth_norm:
        if depth_norm_mode == "percentile":
            d_proc, _ = normalize_depth_percentile(
                d_proc,
                p_lo=depth_norm_p_lo,
                p_hi=depth_norm_p_hi,
                far_plane=depth_norm_far_plane,
                k_compression=depth_norm_k_compression,
                flip_output_sign=(depth_norm_inv_mode == "flip"),
            )
        elif depth_norm_mode == "logarithmic":
            (d_proc,) = normalize_depth_logarithmic(
                d_proc,
                k=depth_norm_k_compression,
                far_plane=depth_norm_far_plane,
            )
            if depth_norm_inv_mode == "flip":
                d_proc = -d_proc

    if curve_correction_mode != "none":
        d_proc = apply_fusion_color_curve(d_proc, curve_name=curve_correction_mode)

    # (1, T, H, W) → (3, T, H, W) to match model expectation
    depths_out = d_proc.repeat(3, 1, 1, 1) if d_proc.shape[0] == 1 else d_proc

    sample["rgbs_clip"] = rgbs_out                   # (3, T, H, W)
    sample["depths_clip_norm_inv_rgb"] = depths_out  # (3, T, H, W)
    sample["depth_canon_scaler"] = scaler_t          # (1, T, 1, 1)
    return sample


def _wds_latent_lookup(
    sample: dict,
    latent_shape_fallback: Tuple[int, int, int] = (4, 36, 64),
) -> dict:
    """
    Attach latent tensors from the (already-sliced) baked members, or zero
    placeholders for a pixels-only sample (trainer encodes online and, on the
    cache pass, writes the per-file cache using the _cache_* metadata below).
    """
    meta = sample["_meta"]
    ds_name = meta["ds"]
    seq_key = meta["seq_key"]
    clip_orig_indices = sample["_clip_orig_indices"]
    T = len(clip_orig_indices)

    if sample.get("_baked_latent_present", False):
        rgb_lat = sample["_clip_rgb_lat"].permute(1, 2, 0, 3, 4)  # (T,2,C,h,w)→(2,C,T,h,w)
        dep_lat = sample["_clip_dep_lat"].permute(1, 2, 0, 3, 4)
        sample["rgb_latent"] = rgb_lat
        sample["depth_latent"] = dep_lat
        sample["_cache_hit"] = torch.tensor(True)
    else:
        # Pixels-only sample: trainer encodes online (and may cache during a pass).
        C, h, w = latent_shape_fallback
        sample["rgb_latent"] = torch.zeros(2, C, T, h, w)
        sample["depth_latent"] = torch.zeros(2, C, T, h, w)
        sample["_cache_hit"] = torch.tensor(False)

    sample["_cache_ds_name"] = ds_name
    sample["_cache_seq_name"] = seq_key
    sample["_cache_frame_indices"] = torch.tensor(clip_orig_indices, dtype=torch.long)
    return _wds_clean(sample)


def _wds_clean(sample: dict) -> dict:
    """Remove private working keys before batching.
    Note: __key__ / __url__ / __local_path__ are left intact — webdataset re-injects
    them (as None) after the map call if we pop them, breaking default_collate.
    """
    for k in ["_clip_rgbs_u8", "_clip_depths", "_clip_positions",
              "_clip_orig_indices", "_clip_rgb_lat", "_clip_dep_lat",
              "_baked_latent_present", "_meta", "meta.json",
              "rgbs.pth", "depths.pth", "rgb_lats.pth", "dep_lats.pth"]:
        sample.pop(k, None)
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

def _no_split(src):
    """No-op nodesplitter: shards are already split per-rank by *_shard_urls().

    Module-level (not a lambda) so it survives pickling to persistent/forkserver
    DataLoader workers.  Replaces webdataset's default ``single_node_only``, which
    raises under world_size>1.
    """
    yield from src


def build_wds_dataloader(
    shard_urls: List[str],
    clip_len: int,
    stride: Tuple[int, int],
    depth_norm: bool,
    depth_norm_mode: str,
    depth_norm_far_plane: float,
    depth_norm_k_compression: float,
    depth_norm_p_lo: float,
    depth_norm_p_hi: float,
    depth_norm_inv_mode: str,
    curve_correction_mode: str,
    batch_size: int,
    num_workers: int,
    shuffle_buffer: int = 200,
    num_samples_per_epoch: int = 0,
    full_sequence: bool = False,
    cache_dir: Optional[str] = None,
    cache_size_gb: float = -1.0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> "DataLoader":
    """
    Build a streaming dataloader.

    full_sequence=False (training): random T-frame clip per sample; loops shards
        and slices into fixed-size epochs (with_epoch) so Lightning's epoch
        boundary / LR scheduler / progress bar work.
    full_sequence=True (cache pass): emit every frame of every sequence exactly
        once (no shuffle, no repeat) for a guaranteed-complete latent cache.

    cache_dir: when set (and not full_sequence), shards are read through
        ``pipe:cat`` so webdataset's FileCache treats them as non-local and caches
        them to ``cache_dir`` with an LRU cap of ``cache_size_gb`` (≤0 = unbounded).
        This is the only way to get NFS→local-SSD shard caching, since webdataset
        reads plain/file: paths in place.  Skipped for the one-pass cache phase.
    """
    import shlex
    import webdataset as wds
    from functools import partial

    depth_fn = partial(
        _wds_depth_process,
        depth_norm=depth_norm,
        depth_norm_mode=depth_norm_mode,
        depth_norm_far_plane=depth_norm_far_plane,
        depth_norm_k_compression=depth_norm_k_compression,
        depth_norm_p_lo=depth_norm_p_lo,
        depth_norm_p_hi=depth_norm_p_hi,
        depth_norm_inv_mode=depth_norm_inv_mode,
        curve_correction_mode=curve_correction_mode,
    )

    do_shuffle = (shuffle_buffer > 0) and not full_sequence

    # Local-SSD LRU cache for NFS-resident shards.  webdataset only caches
    # non-local URLs, so route reads through `pipe:cat` and hand WebDataset a
    # cache dir + byte cap.  Disabled for the one-pass cache phase (no reuse).
    use_cache = bool(cache_dir) and not full_sequence
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)  # WebDataset errors if absent
        urls = [f"pipe:cat {shlex.quote(u)}" for u in shard_urls]
        cache_size = int(cache_size_gb * 1024 ** 3) if cache_size_gb and cache_size_gb > 0 else -1
        cache_kwargs = dict(cache_dir=cache_dir, cache_size=cache_size)
    else:
        urls = shard_urls
        cache_kwargs = {}

    # nodesplitter=_no_split: shard_urls already split per-rank; keep webdataset's
    # default split_by_worker so workers within a rank don't duplicate shards.
    pipeline = wds.WebDataset(urls, shardshuffle=do_shuffle, nodesplitter=_no_split,
                              **cache_kwargs)
    if do_shuffle:
        pipeline = pipeline.shuffle(shuffle_buffer)

    select_fn = (_wds_full_sequence if full_sequence
                 else partial(_wds_clip_select, clip_len=clip_len, stride=stride))

    pipeline = (
        pipeline
        .decode(wds.autodecode.basichandlers)  # handles .json → dict, .pth → tensor
        .map(select_fn)
        .select(lambda x: x is not None)
        .map(depth_fn)
        .map(_wds_latent_lookup)
    )
    if full_sequence:
        # Exactly one deterministic pass over the rank's shards.
        bs = 1   # sequences have variable N → cannot batch heterogeneous lengths
    else:
        bs = batch_size
        if num_samples_per_epoch > 0:
            pipeline = pipeline.repeat().with_epoch(num_samples_per_epoch)

    loader_kwargs: dict = dict(batch_size=bs, num_workers=num_workers,
                               pin_memory=pin_memory, collate_fn=_collate_fn)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers and not full_sequence
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(pipeline, **loader_kwargs)


def _collate_fn(samples: list) -> dict:
    """Collate a list of sample dicts into a batched dict.
    Drops any key whose value is None (webdataset can re-inject __key__=None
    after _wds_clean removes it from the pipeline-internal dict).
    """
    from torch.utils.data.dataloader import default_collate
    # Filter None-valued keys per sample, then drop keys absent in any sample
    filtered = [{k: v for k, v in s.items() if v is not None} for s in samples]
    common_keys = set(filtered[0].keys())
    for s in filtered[1:]:
        common_keys &= set(s.keys())
    filtered = [{k: s[k] for k in common_keys} for s in filtered]
    return default_collate(filtered)
