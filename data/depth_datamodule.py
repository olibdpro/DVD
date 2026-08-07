import time

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from data.dataset_MultiDatasets import (MPISintelDataset,
                                        TartanAirDataset,
                                        SpringDataset,
                                        SceneNetDataset,
                                        MixedVideoDataset,
                                        collate_depth_batch)
from data.dataset_UnrealEXR import UnrealEXRDataset
from data.latent_cache import LatentCache
from utils.utils import print_execution_time


# Samples allowed in flight across all workers. `prefetch_factor` counts *batches*, so
# raising batch_size multiplies the shared memory the loader pins without anyone editing
# the prefetch config: on 2026-07-30 a `batch_size=32` CLI override against the config's
# `batch_size: 1` / `prefetch_factor: 16` put 64 batches (~77 GB) in /dev/shm and got the
# rank OOM-killed by its Tractor cgroup. Budgeting samples makes prefetch fall as the
# batch grows.
#
# 256 samples is ~10 GB of /dev/shm at 288x512x16f fp32. It leaves the tuned
# batch_size=1 / prefetch_factor=16 untouched, and still gives batch_size=32 a prefetch
# depth of 2 (torch's default) rather than starving the loader — EXR decode is slow
# enough that these workers need real lookahead. Headroom: the run that died was at
# 56.7 GB anon + 59 GB shm when the cgroup killed it (the cap itself was never read, so
# treat ~116 GB as a lower bound on it, not a measurement).
#
# This budget assumes ~38 MB/sample (16 frames, 288x512, rgb+depth fp32). It divides by
# num_workers * batch_size, so overriding either is safe. Raising resolution or clip_len
# is NOT — that changes bytes-per-sample, which this treats as constant.
PREFETCH_SAMPLE_BUDGET = 256


def prefetch_for(batch_size: int, num_workers: int, requested: int) -> int:
    """Prefetch batches per worker, capped so workers hold ~PREFETCH_SAMPLE_BUDGET samples.

    Never raises the configured value — only lowers it. Floor of 1 (torch's minimum);
    at large batch sizes one batch per worker is already plenty of lookahead.
    """
    if num_workers <= 0:
        return requested
    return max(1, min(requested, PREFETCH_SAMPLE_BUDGET // (num_workers * max(1, batch_size))))


def prepare_dataloader(list_dataset_names,
                       num_samples_per_epoch,
                       batch_size,
                       clip_len,
                       stride,
                       resize,
                       interpolation,
                       depth_canon,
                       depth_canon_focal_px,
                       depth_norm,
                       depth_norm_mode,
                       depth_norm_far_plane,
                       depth_norm_k_compression,
                       depth_norm_inv_mode,
                       curve_correction_mode,
                       num_workers=1,
                       persistent_workers: bool = True,
                       prefetch_factor: int = 4,
                       latent_cache=None,
                       skip_raw_on_cache_hit: bool = False,
                       rescale_by_world_size: bool = False):
    # Optionally rescale per‑dataset samples by world size so that each rank
    # still sees `num_samples_per_epoch` samples from each dataset.
    if rescale_by_world_size and torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    effective_num_samples_per_epoch = num_samples_per_epoch * world_size

    list_datasets = []
    if("Sintel" in list_dataset_names):
        dataset_sintel = MPISintelDataset(rgbs_root="/jobs/ADGRE/eguo/TEST/Stereo/Datasets/MPI-Sintel/MPI-Sintel/final",
                                          depths_root="/jobs/ADGRE/eguo/TEST/Stereo/Datasets/MPI-Sintel/MPI-Sintel/depth",
                                          num_samples_per_epoch=effective_num_samples_per_epoch,
                                          clip_len=clip_len,
                                          stride=stride,
                                          resize=resize,
                                          interpolation=interpolation,
                                          transform=None,
                                          depth_canon=depth_canon,
                                          depth_canon_focal_px=depth_canon_focal_px,
                                          depth_norm=depth_norm,
                                          depth_norm_mode=depth_norm_mode,
                                          depth_norm_far_plane=depth_norm_far_plane,
                                          depth_norm_k_compression=depth_norm_k_compression,
                                          depth_norm_inv_mode=depth_norm_inv_mode,
                                          curve_correction_mode=curve_correction_mode,)
        list_datasets.append(dataset_sintel)

    if("TartanAir" in list_dataset_names):
        dataset_tartanair = TartanAirDataset(root="/jobs/ADGRE/eguo/TEST/Stereo/Datasets/TartanAir/TartanAirFull",
                                             num_samples_per_epoch=effective_num_samples_per_epoch,
                                             clip_len=clip_len,
                                             stride=stride,
                                             resize=resize,
                                             interpolation=interpolation,
                                             transform=None,
                                             depth_canon=depth_canon,
                                             depth_canon_focal_px=depth_canon_focal_px,
                                             depth_norm=depth_norm,
                                             depth_norm_mode=depth_norm_mode,
                                             depth_norm_far_plane=depth_norm_far_plane,
                                             depth_norm_k_compression=depth_norm_k_compression,
                                             depth_norm_inv_mode=depth_norm_inv_mode,
                                             curve_correction_mode=curve_correction_mode,)
        list_datasets.append(dataset_tartanair)

    if("Spring" in list_dataset_names):
        dataset_spring = SpringDataset(root="/jobs/ADGRE/eguo/TEST/Stereo/Datasets/Spring/spring/train",
                                       num_samples_per_epoch=effective_num_samples_per_epoch,
                                       clip_len=clip_len,
                                       stride=stride,
                                       resize=resize,
                                       interpolation=interpolation,
                                       transform=None,
                                       depth_canon=depth_canon,
                                       depth_canon_focal_px=depth_canon_focal_px,
                                       depth_norm=depth_norm,
                                       depth_norm_mode=depth_norm_mode,
                                       depth_norm_far_plane=depth_norm_far_plane,
                                       depth_norm_k_compression=depth_norm_k_compression,
                                       depth_norm_inv_mode=depth_norm_inv_mode,
                                       curve_correction_mode=curve_correction_mode,)
        list_datasets.append(dataset_spring)

    if("SceneNet" in list_dataset_names):
        dataset_scenenet = SceneNetDataset(root="/jobs/ADGRE/eguo/TEST/Stereo/Datasets/SceneNet/Full",
                                           num_samples_per_epoch=effective_num_samples_per_epoch,
                                           clip_len=clip_len,
                                           stride=stride,
                                           resize=resize,
                                           interpolation=interpolation,
                                           transform=None,
                                           depth_norm=depth_norm,
                                           depth_norm_mode=depth_norm_mode,
                                           depth_norm_far_plane=depth_norm_far_plane,
                                           depth_norm_k_compression=depth_norm_k_compression,
                                           depth_norm_inv_mode=depth_norm_inv_mode,
                                           curve_correction_mode=curve_correction_mode,)
        list_datasets.append(dataset_scenenet)

    if("UnrealEXR" in list_dataset_names):
        dataset_unreal = UnrealEXRDataset(dataset_csv="/jobs/ADGRE/sd_001_500/unreal/dataset.csv",
                                          num_samples_per_epoch=effective_num_samples_per_epoch,
                                          clip_len=clip_len,
                                          stride=stride,
                                          resize=resize,
                                          interpolation=interpolation,
                                          transform=None,
                                          depth_canon=depth_canon,
                                          depth_canon_focal_px=depth_canon_focal_px,
                                          depth_norm=depth_norm,
                                          depth_norm_mode=depth_norm_mode,
                                          depth_norm_far_plane=depth_norm_far_plane,
                                          depth_norm_k_compression=depth_norm_k_compression,
                                          depth_norm_inv_mode=depth_norm_inv_mode,
                                          curve_correction_mode=curve_correction_mode,)
        list_datasets.append(dataset_unreal)

    if latent_cache is not None:
        for ds in list_datasets:
            ds.latent_cache = latent_cache
            ds.skip_raw_on_cache_hit = skip_raw_on_cache_hit

    # Wrap it into a MixedVideoDataset
    dataset_training_mixed = MixedVideoDataset(list_datasets)

    loader_kwargs = dict(batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers,
                         pin_memory=True,
                         collate_fn=collate_depth_batch)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        capped = prefetch_for(batch_size, num_workers, prefetch_factor)
        if capped != prefetch_factor:
            print(f" *** prefetch_factor {prefetch_factor} -> {capped} "
                  f"(batch_size={batch_size}, num_workers={num_workers}): keeps in-flight "
                  f"samples near {PREFETCH_SAMPLE_BUDGET} instead of "
                  f"{num_workers * prefetch_factor * batch_size}", flush=True)
        loader_kwargs["prefetch_factor"] = capped
    loader = DataLoader(dataset_training_mixed, **loader_kwargs)
    return loader


class DepthDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loader_train = None
        self.loader_val = None
        self.latent_cache = None

    def setup(self, stage=None):
        cfg = self.cfg
        cfg_c, cfg_d, cfg_v, cfg_t = cfg.common, cfg.dataset, cfg.validation, cfg.training

        lc_cfg = cfg_d.get("latent_cache", None)
        if lc_cfg is not None and lc_cfg.get("enabled", False):
            self.latent_cache = LatentCache(
                cache_dir=lc_cfg.cache_dir,
                version_tag=lc_cfg.version_tag,
                write_enabled=True,
            )

        time_start_dataloader = time.time()

        loader_train = prepare_dataloader(list_dataset_names=cfg_d.training,
                                          num_samples_per_epoch=cfg_d.num_samples_per_epoch,
                                          batch_size=cfg_t.batch_size,
                                          clip_len=cfg_c.clip_len,
                                          stride=tuple(cfg_d.stride),
                                          resize=(cfg_c.height, cfg_c.width),
                                          interpolation=cfg_d.interpolation,
                                          depth_canon=cfg_d.depth_canon,
                                          depth_canon_focal_px=cfg_d.depth_canon_focal_px,
                                          depth_norm=cfg_d.depth_norm,
                                          depth_norm_mode=cfg_d.depth_norm_mode,
                                          depth_norm_far_plane=cfg_d.depth_norm_far_plane,
                                          depth_norm_k_compression=cfg_d.depth_norm_k_compression,
                                          depth_norm_inv_mode=cfg_d.depth_norm_inv_mode,
                                          curve_correction_mode=cfg_d.get("curve_correction_mode", "none"),
                                          num_workers=cfg_d.num_workers,
                                          persistent_workers=cfg_d.get("persistent_workers", True),
                                          prefetch_factor=cfg_d.get("prefetch_factor", 4),
                                          latent_cache=self.latent_cache,
                                          skip_raw_on_cache_hit=cfg_d.get("skip_raw_on_cache_hit", False),
                                          rescale_by_world_size=False)

        loader_val = prepare_dataloader(list_dataset_names=cfg_d.validation,
                                        num_samples_per_epoch=cfg_d.num_samples_per_epoch,
                                        batch_size=cfg_v.batch_size,
                                        clip_len=cfg_c.clip_len,
                                        stride=tuple(cfg_d.stride),
                                        resize=(cfg_c.height, cfg_c.width),
                                        interpolation=cfg_d.interpolation,
                                        depth_canon=cfg_d.depth_canon,
                                        depth_canon_focal_px=cfg_d.depth_canon_focal_px,
                                        depth_norm=cfg_d.depth_norm,
                                        depth_norm_mode=cfg_d.depth_norm_mode,
                                        depth_norm_far_plane=cfg_d.depth_norm_far_plane,
                                        depth_norm_k_compression=cfg_d.depth_norm_k_compression,
                                        depth_norm_inv_mode=cfg_d.depth_norm_inv_mode,
                                        curve_correction_mode=cfg_d.get("curve_correction_mode", "none"),
                                        num_workers=cfg_d.num_workers,
                                        persistent_workers=cfg_d.get("persistent_workers", True),
                                        prefetch_factor=cfg_d.get("prefetch_factor", 4),
                                        latent_cache=self.latent_cache,
                                        rescale_by_world_size=False)

        time_end_dataloader = time.time()
        print_execution_time(time_start_dataloader, time_end_dataloader, prefix=">>>>>>>>>>>>>>>>>>>>Preparing Dataloader")

        self.loader_train = loader_train
        self.loader_val = loader_val

    def train_dataloader(self):
        return self.loader_train

    def val_dataloader(self):
        return self.loader_val