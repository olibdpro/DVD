import os
import io
import re
import yaml
import gc
import math
import logging
from pathlib import Path

import importlib
from collections import OrderedDict
from typing import Optional, Callable, List
from functools import reduce
from natsort import natsorted

from omegaconf import OmegaConf, open_dict
import cv2
import numpy as np
import torch
import torch.distributed as dist

EPS = 1e-8
FP32 = "fp32"
FP16 = "fp16"


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """
    name: full name of source para
    para_list: partial name of target para
    """
    istarget = False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k ** 0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )


def load_model_checkpoint(ckpt_path):
    ckpt_raw = torch.load(ckpt_path, map_location="cpu")
    model_state_dict = None
    optimizer_state_dict = None
    scheduler_state_dict = None
    config = None
    epoch = None
    step = None

    # if "state_dict" in list(state_dict_raw.keys()): # For directly loading the original dynamicrafter checkpoint
    if "state_dict" in ckpt_raw:  # For directly loading the original dynamicrafter checkpoint
        model_state_dict = ckpt_raw["state_dict"]
    # elif "model_state_dict" in list(state_dict_raw.keys()): # For loading the vdd checkpoint
    elif "model_state_dict" in ckpt_raw:  # For loading the vdd checkpoint
        model_state_dict = ckpt_raw["model_state_dict"]

        if "optimizer_state_dict" in ckpt_raw:
            optimizer_state_dict = ckpt_raw["optimizer_state_dict"]

        if "scheduler_state_dict" in ckpt_raw:
            scheduler_state_dict = ckpt_raw["scheduler_state_dict"]

        if "config" in ckpt_raw:
            config = OmegaConf.create(ckpt_raw["config"])

        if "epoch" in ckpt_raw:
            epoch = ckpt_raw["epoch"]

        if "step" in ckpt_raw:
            step = ckpt_raw["step"]

    print(f">>> [checkpoint] loaded from {ckpt_path}")

    ckpt_raw.clear()
    del ckpt_raw
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    return model_state_dict, optimizer_state_dict, scheduler_state_dict, config, epoch, step


def clean_up_state_dict(list_state_dicts):
    for state_dict in list_state_dicts:
        if state_dict is not None:
            state_dict.clear()
            del state_dict

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def load_vdd_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "model_state_dict" in list(state_dict.keys()):
        state_dict = state_dict["model_state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k, v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            # model.load_state_dict(new_pl_sd, strict=True)
            model.load_state_dict(new_pl_sd, strict=False)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def find_latest_checkpoints(ckpt_dir: str | Path, ):
    """
    Return (sorted_list, latest_path) where filenames are ordered
    “naturally” (1, 2, … 10, 11 … 100) instead of lexicographically.
    """
    CKPT_NAME_PATTERN = re.compile(r"ckpt_step_\d+\.pt$", re.IGNORECASE)
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"{ckpt_dir} is not a directory")

    # gather every matching file in the directory
    ckpt_files = [p for p in ckpt_dir.iterdir() if CKPT_NAME_PATTERN.match(p.name)]

    # natural-sort by filename (natsort handles embedded numbers)
    ckpt_sorted = natsorted(ckpt_files, key=lambda p: p.name)

    ckpt_name_latest = ckpt_sorted[-1] if ckpt_sorted else None
    return ckpt_name_latest


def load_checkpoint(model,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    ckpt_path: Path,
                    *,
                    ema: bool = False,
                    device="cuda") -> int:
    """
    Load weights (and optionally optimiser / scaler) from `ckpt_path`.
    Returns the step we resume from.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # ToDo: ─ EMA for **sampling** (skip if you only care about training) ────────
    # if ema and "model_ema" in ckpt and hasattr(model, "ema_model"):
    #     print("[checkpoint] loading EMA parameters")
    #     model.ema_model.load_state_dict(ckpt["model_ema"], strict=False)

    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step = ckpt["step"]
    epoch = ckpt["epoch"]
    config = OmegaConf.create(ckpt["config"])

    print(f"[checkpoint] loaded from {ckpt_path} at step {step}")

    ckpt.clear()
    del ckpt
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    return model, optimizer, scheduler, config, step, epoch


def save_checkpoint_and_config(model_state_dict,
                               optimizer_state_dict: torch.optim.Optimizer,
                               scheduler_state_dict: torch.optim.lr_scheduler._LRScheduler,
                               cfg_save,
                               step: int,
                               epoch: int,
                               save_folder_path,
                               *,
                               ema: bool = False,
                               ) -> Path:
    """
    Persist model/optim/&c. so you can resume **training** or do **inference**
    later.  Returns the file path that was written.
    """
    os.makedirs(save_folder_path, exist_ok=True)
    cfg_dynamicrafter_yaml_str = OmegaConf.to_yaml(OmegaConf.load(cfg_save["model"]["cfg_yaml"]), resolve=True)

    ckpt_path = f"{save_folder_path}/ckpt_step_{step:010d}.pt"
    cfg_path = f"{save_folder_path}/ckpt_step_{step:010d}.yaml"
    cfg_dynamicrafter_path = f"{save_folder_path}/ckpt_step_dynamicrafter_{step:010d}.yaml"

    cfg_save["model"]["ckpt_path"] = ckpt_path
    cfg_save["model"]["cfg_yaml"] = cfg_dynamicrafter_path

    cfg_yaml_str = OmegaConf.to_yaml(cfg_save, resolve=True)

    payload = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "config": OmegaConf.to_yaml(cfg_save, resolve=True),
    }
    payload["optimizer_state_dict"] = optimizer_state_dict if optimizer_state_dict is not None else None
    payload["scheduler_state_dict"] = scheduler_state_dict if scheduler_state_dict is not None else None
    torch.save(payload, ckpt_path)

    with open(cfg_path, 'w') as f:
        f.write(cfg_yaml_str)

    with open(cfg_dynamicrafter_path, 'w') as f:
        f.write(cfg_dynamicrafter_yaml_str)

    print(f"[checkpoint] saved → {ckpt_path}")
    print(f"[config] saved → {cfg_path}")
    print(f"[dynamicrafter_config] saved → {cfg_dynamicrafter_path}")
    return ckpt_path


def save_deepspeed_checkpoint_and_config(engine,
                                         scheduler_state_dict,
                                         cfg_save,
                                         step: int,
                                         epoch: int,
                                         save_folder_path,
                                         *,
                                         ema: bool = False,
                                         ) -> Path:
    os.makedirs(save_folder_path, exist_ok=True)
    ds_ckpt_dir = os.path.join(save_folder_path, "ds_checkpoints")
    os.makedirs(ds_ckpt_dir, exist_ok=True)

    tag = f"global_step{step}"

    client_state = {}

    engine.save_checkpoint(ds_ckpt_dir, tag=tag, client_state=client_state)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    model_state_dict = None
    need_gather = False

    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        try:
            model_state_dict = get_fp32_state_dict_from_zero_checkpoint(ds_ckpt_dir, tag=tag)
        except Exception as e:
            need_gather = True
            print(f"[checkpoint] zero_to_fp32 failed → {e}")

    if dist.is_available() and dist.is_initialized():
        flag_device = "cuda" if torch.cuda.is_available() else "cpu"
        flag = torch.tensor([1 if need_gather else 0], device=flag_device)
        dist.broadcast(flag, src=0)
        need_gather = bool(flag.item())

    if need_gather:
        try:
            if hasattr(engine, "_zero3_consolidated_16bit_state_dict"):
                model_state_dict = engine._zero3_consolidated_16bit_state_dict()
            elif hasattr(engine, "module"):
                model_state_dict = engine.module.state_dict()
            else:
                model_state_dict = engine.state_dict()
        except Exception:
            model_state_dict = None

    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        if isinstance(model_state_dict, dict) and len(model_state_dict) > 0:
            k0 = next(iter(model_state_dict.keys()))
            if isinstance(k0, str) and k0.startswith("module."):
                new_sd = OrderedDict()
                for k, v in model_state_dict.items():
                    new_sd[k[len("module."):]] = v
                model_state_dict = new_sd

            save_checkpoint_and_config(model_state_dict=model_state_dict,
                                       optimizer_state_dict=None,
                                       scheduler_state_dict=scheduler_state_dict,
                                       cfg_save=cfg_save,
                                       step=step,
                                       epoch=epoch,
                                       save_folder_path=save_folder_path,
                                       ema=ema)
        else:
            print(f"[checkpoint] failed to build a consolidated state_dict for step {step}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return f"{save_folder_path}/ckpt_step_{step:010d}.pt"


def load_deepspeed_checkpoint(engine,
                              ckpt_dir,
                              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                              *,
                              tag: Optional[str] = None):
    load_path, client_state = engine.load_checkpoint(str(ckpt_dir), tag=tag)
    if load_path is None:
        raise FileNotFoundError(f"[checkpoint] failed to load DeepSpeed checkpoint from {ckpt_dir}")

    step = None
    epoch = None
    config = None

    print(f"[checkpoint] loaded from {ckpt_dir}")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    return engine, scheduler, config, step, epoch


def convert_module_to_fp16(module: torch.nn.Module) -> None:
    """
    Cast everything to fp16 **except** layers that are numerically sensitive.
    """
    for subm in module.modules():
        # keep norms & embeddings in fp32 for stability
        if isinstance(subm, (torch.nn.LayerNorm,
                             torch.nn.GroupNorm,
                             torch.nn.BatchNorm2d,
                             torch.nn.Embedding)):
            subm.float()
        else:
            subm.half()


def print_execution_time(time_start, time_end, prefix):
    time_elapsed = int(time_end - time_start)
    hours = time_elapsed // 3600
    minutes = (time_elapsed % 3600) // 60
    seconds = time_elapsed % 60
    print(f"{prefix} Timed: {hours:02}:{minutes:02}:{seconds:02}")


class DummyAimRun:
    def track(self, *args, **kwargs):
        pass

    def __setitem__(self, key, value):
        pass


def clear_memory(*to_clear):
    for item in to_clear:
        if type(item) in [dict, list, set]:
            item.clear()
        elif item is None:
            pass
        try:
            del item
        except Exception as e:
            logging.error(f"Error raised when trying to clear memory: {e}")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def load_config_from_aim_hash(run_hash: str, aim_repo: str = "."):
    """Read the training config stored in a previous AIM run and locate its checkpoint.

    Reads the ``hparams`` logged at train time
    (``logger.experiment["hparams"] = OmegaConf.to_container(cfg, resolve=True)``),
    rebuilds the config from it, and points at ``<output_dir>/last.ckpt``. The
    checkpoint's existence is NOT verified here — callers that require it (e.g.
    :func:`resume_from_aim_hash`) should check, while callers that may override the
    checkpoint (e.g. evaluation) can ignore a missing ``last.ckpt``.

    Args:
        run_hash: AIM run hash to load.
        aim_repo: Path or URL accepted by ``aim.Run(repo=...)``. Use ``"."`` for a
            local repo (AIM appends ``.aim``) or ``aim://host:port`` for a server.

    Returns:
        (cfg, ckpt_path): the config rebuilt from the run's ``hparams`` (fully
        resolved, so no resolvers are needed) and the path to its ``last.ckpt``
        (which may be a single file or a DeepSpeed ZeRO ``.ckpt`` directory).
    """
    from aim import Run as AimRun, Repo

    run = AimRun(run_hash=run_hash, repo=aim_repo, read_only=True)

    if run.get("hparams", None) is None:

        all_runs = [run.hash for run in Repo(aim_repo).iter_runs()]
        if run_hash not in all_runs:
            from aim.sdk.errors import RepoIntegrityError

            all_run_name = []
            for run in Repo(aim_repo).iter_runs():
                try:
                    all_run_name.append(run.name)
                except RepoIntegrityError:
                    all_run_name.append(f"{run.hash} is corrupted")
                    continue

            raise ValueError(
                f"AIM run {run_hash} not found in repo '{aim_repo}'. "
                f"\n Runs found: "
                f"\n        {all_runs}, "
                f"\n        {all_run_name}"
            )

        raise ValueError(
            f"AIM run {run_hash} has no 'hparams' key — was it logged with "
            "logger.experiment['hparams'] = ...?"
        )

    hparams = run["hparams"]
    del run

    cfg = OmegaConf.create(dict(hparams))

    dirpath = hparams.get("checkpointer").get("dirpath")
    if not dirpath:
        raise ValueError(f"AIM run {run_hash}: hparams missing 'checkpointer.dirpath'")
    ckpt_path = os.path.join(dirpath, "last.ckpt")

    return cfg, ckpt_path


def _migrate_resumed_strategy(cfg):
    """Map a resumed run's deprecated FairScale FSDP strategy string to native FSDP.

    Runs logged before the FairScale -> PyTorch-native FSDP migration stored
    ``trainer.strategy='fsdp'`` (PL's ``DDPFullyShardedStrategy``). The current
    model's ``configure_sharded_model`` is native-only (``torch.distributed.fsdp``);
    under the FairScale strategy, torch's ``wrap()`` is a no-op (its ``enable_wrap``
    context is never activated), so ``self.dit`` is never sharded, the optimizer
    references no ``FlatParameter``, and PL raises "The optimizer does not seem to
    reference any FSDP parameters" at ``setup_optimizers``. Rewrite the bare string
    to the native strategy the code targets. No-op for any other strategy.
    """
    if OmegaConf.select(cfg, "trainer.strategy") == "fsdp":
        with open_dict(cfg):
            cfg.trainer.strategy = "fsdp_native"
        logging.warning(
            "Resumed config used deprecated FairScale strategy 'fsdp'; migrating to "
            "'fsdp_native' to match the native-FSDP model code."
        )
    return cfg


def _is_valid_checkpoint(path: str) -> tuple[bool, bool]:
    """Return ``(is_readable, has_optimizer_states)`` for the checkpoint at *path*.

    ``has_optimizer_states`` is False for a ``save_weights_only=True`` checkpoint;
    the caller uses it to downgrade a full resume to a weights-only warm-start
    instead of letting PL's checkpoint connector hard-raise on the missing key.
    """
    try:
        probe = torch.load(path, map_location="cpu", weights_only=False)
        has_optim = "optimizer_states" in probe
        del probe
        return True, has_optim
    except Exception as exc:
        logging.warning(f"Checkpoint {path} appears corrupt ({exc.__class__.__name__}: {exc}).")
        return False, False


def _should_downgrade_to_weights_only(weights_only: bool, has_optimizer_state: bool) -> bool:
    """A full resume of a checkpoint with no optimizer state can't succeed — PL 1.9.3's
    checkpoint connector hard-raises on the missing ``optimizer_states`` / ``lr_schedulers``.
    Downgrade to a weights-only warm-start (fresh optimizer + step) instead of crashing."""
    return not weights_only and not has_optimizer_state


def _validate_checkpoint_or_fallback(ckpt_path: str) -> tuple[str, bool]:
    """Validate *ckpt_path*; if corrupt, scan siblings for the most recent valid one.

    Sibling ``.ckpt`` files in the same directory are sorted by modification
    time (newest first) and tried in order.  If no valid checkpoint is found
    a :class:`RuntimeError` is raised with a clear diagnostic message.

    Returns ``(path, has_optimizer_states)`` of a verified readable checkpoint.
    """
    is_valid, has_optim = _is_valid_checkpoint(ckpt_path)
    if is_valid:
        return ckpt_path, has_optim

    raise RuntimeError(
        f"Checkpoint {ckpt_path} is corrupt and no valid fallback was found in {ckpt_dir}. "
        "Please inspect the directory and remove or repair the corrupt file, then retry. "
        "You can also pass ++ignore_ckpt=true to start from scratch with the same config."
    )


def resume_from_aim_hash(
        cfg,
        run_hash: str,
        aim_repo: str = ".",
        resume_config: bool = True,
        resume_tracking: bool = True,
        weights_only: bool = False,
        ignore_ckpt: bool = False,
):
    """Locate and configure a resume from a previous AIM run hash.

    Reads the ``hparams`` stored in the AIM run to find the old ``output_dir``,
    then sets ``cfg.ckpt_path`` to the ``last.ckpt`` saved there.  When
    ``resume_config=True`` the stored hparams replace the current config
    (preserving only the new ``output_dir`` / ``run_name``), mirroring the
    behaviour of :func:`resume_config`.

    Args:
        cfg: Current Hydra DictConfig.
        run_hash: AIM run hash to resume from.
        aim_repo: Path or URL accepted by ``aim.Run(repo=...)``.
            Use ``"."`` for a local repo in the current directory (AIM appends
            ``.aim`` automatically), or ``aim://host:port`` for a remote server.
        resume_config: If True, restore the old hyper-parameter config.
        resume_tracking: If True, continue logging into the same AIM run
            (requires ``resume_config=True``).
        ignore_ckpt: If True, restore code/config from the referenced run but
            start training with fresh weights (ckpt_path is NOT set).

    Returns:
        Updated DictConfig ready for training.
    """
    if resume_tracking and not resume_config:
        raise ValueError("resume_config must be True when resume_tracking=True")

    old_cfg, ckpt_path = load_config_from_aim_hash(run_hash, aim_repo)
    old_cfg = _migrate_resumed_strategy(old_cfg)
    output_dir = old_cfg.get("output_dir")

    # load_config_from_aim_hash returns the weights-only checkpoint (checkpointer.dirpath).
    # A full resume (weights_only=False) needs the optimizer state, which only the
    # checkpointer_full callback writes (e.g. <output_dir>/full/last.ckpt). Point at it
    # when available; otherwise fall back to weights-only and warn that the optimizer
    # state will not be restored.
    if not weights_only and not ignore_ckpt:
        full_dir = OmegaConf.select(old_cfg, "checkpointer_full.dirpath")
        if full_dir:
            full_ckpt = os.path.join(full_dir, "last.ckpt")
            if os.path.exists(full_ckpt):
                ckpt_path = full_ckpt
            else:
                logging.warning(
                    f"Full resume requested but {full_ckpt} not found; falling back to "
                    f"weights-only checkpoint {ckpt_path} (optimizer state NOT restored)."
                )

    if not ignore_ckpt and not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            "Make sure the cluster storage is accessible from this machine."
        )

    # Validate the checkpoint is readable; fall back to the latest valid sibling if corrupt.
    if not ignore_ckpt:
        ckpt_path, has_optim = _validate_checkpoint_or_fallback(ckpt_path)
        # If a full resume was requested but the chosen checkpoint carries no optimizer
        # state (save_weights_only=True, or no checkpointer_full/full-ckpt was found
        # above), a full resume would crash in PL's checkpoint connector. Flip to
        # weights-only so `resume_config` merges to False (below) and train_transformer's
        # manual weights-only load path runs instead — config is still restored, but the
        # optimizer + step counter reset. Flipping weights_only (not resume_config)
        # leaves the config-restore branch below intact.
        if _should_downgrade_to_weights_only(weights_only, has_optim):
            logging.warning(
                f"Full resume requested but {ckpt_path} has no optimizer state; "
                "downgrading to a weights-only warm-start (optimizer + step reset). "
                "Set checkpointer.save_weights_only=false to enable full optimizer resume."
            )
            weights_only = True

    cur_output_dir = cfg.output_dir
    cur_run_name = cfg.run_name

    if not resume_config:
        with open_dict(cfg):
            if not ignore_ckpt:
                cfg.ckpt_path = ckpt_path
    else:
        cfg = old_cfg
        with open_dict(cfg):
            if not ignore_ckpt:
                cfg.ckpt_path = ckpt_path
            cfg.output_dir = cur_output_dir
            cfg.run_name = cur_run_name
            if resume_tracking and OmegaConf.select(cfg, "logger") is not None:
                cfg.logger.run_hash = run_hash
                cfg.logger.run_name = None
            else:
                if OmegaConf.select(cfg, "logger.run_hash") is not None:
                    cfg.logger.run_hash = None
                if OmegaConf.select(cfg, "logger.run_name") is not None:
                    cfg.logger.run_name = cur_run_name

    with open_dict(cfg):
        cfg = OmegaConf.merge(cfg, {
            "resume_aim_hash": run_hash,
            "resume_config": resume_config and not weights_only,
            "resume_tracking": resume_tracking,
        })

    if ignore_ckpt:
        logging.info(
            f"Restoring code/config from AIM run {run_hash} (output_dir={output_dir}) — "
            "starting with FRESH weights (ignore_ckpt=True)."
        )
    else:
        logging.info(f"Resuming from AIM run {run_hash} (output_dir={output_dir}, ckpt={ckpt_path})")
    return cfg


def resume_config(cfg, resume_dir: str, resume_config=True, resume_tracking=False, ckpt_name="ckpt.pt", ignore_ckpt=False):
    """Update the config to resume from a previous hydra run's output directory

    Args:
        cfg (omegaconf.DictConfig): Current config
        resume_dir (str): Path to the to-be-resumed run's output directory.
                            This dir should contain the hydra config files and checkpoints.
        resume_config (bool, optional): If True will overwrite the current config with old config values.
                                        If False will only change the ckpt_path to point to the resumed run's ckpt.
        resume_tracking (bool, optional): If True will include the old aim run's run hash in the config.
        ignore_ckpt (bool, optional): If True, restore code/config from the referenced run but
                                      start training with fresh weights (ckpt_path is NOT set).

    Returns:
        cfg (DictConfig): Reconfigured configs based on the resume setting

    """
    if not os.path.isdir(resume_dir):
        raise FileNotFoundError
    if resume_tracking and not resume_config:
        raise ValueError("resume_config has to be True to resume_tracking")
    ckpt_path = os.path.join(resume_dir, ckpt_name)
    if not ignore_ckpt and not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} does not exist.")
    if not resume_config:
        if not ignore_ckpt:
            cfg.ckpt_path = ckpt_path
    else:
        # FUTURE: Still allow override configs when resume_config=True
        # FIXME: This leaves the unresovled config.yaml incorrect.
        config_path = os.path.join(resume_dir, "hydra/resolved_config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(resume_dir, "hydra/config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config file found in {resume_dir}/hydra/")
        cur_output_dir, cur_run_name = cfg.output_dir, cfg.run_name
        cfg = OmegaConf.load(config_path)
        cfg = _migrate_resumed_strategy(cfg)
        if not ignore_ckpt:
            cfg.ckpt_path = ckpt_path
        cfg.output_dir = cur_output_dir
        cfg.run_name = cur_run_name
        # FUTURE: Also copy and continue the old log file if resume_tracking?
        if not resume_tracking:
            cfg.aim_run.run_hash = None
            if OmegaConf.select(cfg, "logger.run_hash") is not None:
                cfg.logger.run_hash = None
            if OmegaConf.select(cfg, "logger.run_name") is not None:
                cfg.logger.run_name = cur_run_name
    with open_dict(cfg):
        resume_config = {
            "resume": resume_dir,
            "resume_config": resume_config,
            "resume_tracking": resume_tracking
        }
        cfg = OmegaConf.merge(cfg, resume_config)
    if ignore_ckpt:
        logging.info(
            f"Restored code/config from {resume_dir} — starting with FRESH weights (ignore_ckpt=True)."
        )
    else:
        logging.info(f"Resumed training from {resume_dir}.")
    return cfg


def apply_functions(funcs: Callable | List[Callable], *inputs):
    """Sequentially apply each callable in list on each of the inputs."""
    if not isinstance(funcs, list):
        funcs = [funcs]
    outputs = map(lambda x: reduce(lambda acc, func: func(acc), funcs, x), inputs)
    return outputs