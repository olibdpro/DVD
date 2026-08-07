import os
import queue
import threading
from pathlib import Path

import torch


class LatentCache:
    """Disk-backed cache of per-frame VAE latents.

    Writes happen from the trainer (main) process via a background thread that
    drains an in-memory queue. Dataloader workers (forked) share the path
    convention and call ``load()`` only — they never enqueue writes, so the
    writer thread missing in their address space is harmless.

    Latents are stored as fp32 ``[C, h, w]`` tensors, one file per
    (dataset, sequence, frame_idx, kind) tuple. Kind is ``"rgb"`` or
    ``"depth"``. Writes are atomic (tmp → rename) and idempotent (skipped if
    the destination already exists).

    The latent stored on disk matches the output of ``encode_images`` exactly
    (i.e. it already includes the model's ``scale_factor`` multiplier), so a
    cache hit can be substituted for the encode result without further
    rescaling.
    """

    def __init__(self, cache_dir: str, version_tag: str, write_enabled: bool = True,
                 queue_size: int = 2048):
        self.root = Path(cache_dir) / version_tag
        self.write_enabled = write_enabled
        if write_enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._writer_q: queue.Queue = queue.Queue(maxsize=queue_size)
            self._writer = threading.Thread(target=self._drain, name="LatentCacheWriter", daemon=True)
            self._writer.start()

    def path(self, ds: str, seq: str, frame_idx: int, kind: str) -> Path:
        return self.root / ds / seq / f"{frame_idx:06d}_{kind}.pt"

    def load(self, ds: str, seq: str, frame_idx: int, kind: str):
        p = self.path(ds, seq, frame_idx, kind)
        if not p.exists():
            return None
        try:
            return torch.load(p, map_location="cpu", weights_only=True).clone()
        except Exception:
            return None

    def schedule_write(self, ds: str, seq: str, frame_idx: int, kind: str, latent_chw: torch.Tensor):
        if not self.write_enabled:
            return
        if self.path(ds, seq, frame_idx, kind).exists():
            return
        try:
            self._writer_q.put_nowait((ds, seq, frame_idx, kind, latent_chw))
        except queue.Full:
            # Drop silently — the next time we encode this frame we'll get another shot.
            pass

    def __getstate__(self):
        # DataLoader workers pickle the dataset (and its LatentCache).
        # queue.Queue and threading.Thread hold _thread.lock objects that
        # can't be pickled. Workers only call load()/path() — never
        # schedule_write() — so strip the writer state and mark read-only.
        state = self.__dict__.copy()
        state.pop("_writer_q", None)
        state.pop("_writer", None)
        state["write_enabled"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # write_enabled is False; no thread/queue needed in worker processes

    def _drain(self):
        while True:
            ds, seq, fi, kind, t = self._writer_q.get()
            p = self.path(ds, seq, fi, kind)
            if p.exists():
                continue
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".pt.tmp")
                torch.save(t.contiguous(), tmp)
                os.replace(tmp, p)
            except Exception:
                # Any I/O failure: drop this write; cache will refill on next encounter.
                pass
