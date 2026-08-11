"""Dump grayscale depth PNGs. No metrics, no GT, no videos.

Runs the DVD checkpoint over the old research datasets (data/dataset_MultiDatasets.py,
data/dataset_UnrealEXR.py), or over a plain folder of frames / a video file.

Usage
─────
python test_script/eval_depth_dump.py --ckpt ckpt --dataset Sintel \
    --data_root /path/to/MPI-Sintel/training -o ./depth_dump --max_clips 10

python test_script/eval_depth_dump.py --ckpt ckpt --dataset DepthCollapse \
    --dataset_csv /path/to/shots.csv -o ./depth_dump

python test_script/eval_depth_dump.py --ckpt ckpt --dataset /path/to/frames -o ./depth_dump
python test_script/eval_depth_dump.py --ckpt ckpt --dataset demo/robot_navi.mp4 -o ./depth_dump

--data_root falls back to root_dir in configs/dataset/<name>.yaml when omitted.

Each clip is predicted at five input sizes -- the training size (--height/--width),
width 512, width 1024, width 3840 (4K, upscale-only) and the source's own size. A
source already at one of them is not resized: that pass is shared by both bins.

Output: <outputs_path>/{train,512,1024,4k,original}/{depth,rgb}/<clip name>/frame_0000.png ...
The rgb folder holds the exact (resized) frames that were fed to the model.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # data.* and utils.* aren't in the editable install

import numpy as np
import torch
from natsort import natsorted
from omegaconf import OmegaConf
from PIL import Image

# Sibling script: sys.path[0] is test_script/ when launched as
# `python test_script/eval_depth_dump.py` (how infer_bash/*.sh call it).
from test_single_video import (generate_depth_sliced, get_window_index,
                               load_model, read_video,
                               resize_for_training_scale)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")

UHD_WIDTH = 3840  # the "4k" bin; 16-aligned already, DCI 4096 would be too

# name -> (module, class). Root kwarg differs per family, see build_dataset.
DATASETS = {
    "Sintel":        ("data.dataset_MultiDatasets", "MPISintelDataset"),
    "TartanAir":     ("data.dataset_MultiDatasets", "TartanAirDataset"),
    "Spring":        ("data.dataset_MultiDatasets", "SpringDataset"),
    "SceneNet":      ("data.dataset_MultiDatasets", "SceneNetDataset"),
    "UnrealEXR":     ("data.dataset_UnrealEXR", "UnrealEXRDataset"),
    "DepthCollapse": ("data.dataset_UnrealEXR", "DepthCollapseDataset"),
}


def yaml_root(name):
    """root_dir from configs/dataset/<name>.yaml, as the old eval script did."""
    p = REPO / "configs" / "dataset" / f"{name.lower()}.yaml"
    return str(OmegaConf.load(p).root_dir) if p.exists() else None


def build_dataset(args):
    from importlib import import_module
    module, cls_name = DATASETS[args.dataset]
    cls = getattr(import_module(module), cls_name)

    common = dict(
        num_samples_per_epoch=args.max_clips,
        clip_len=args.clip_len,
        stride=(1, 1),
        resize=None,  # native resolution; size_variants() does every resize

        # Depth is loaded but never used here; keep it off the focal-length path.
        depth_canon=False,
    )
    if args.dataset in ("UnrealEXR", "DepthCollapse"):
        if args.dataset_csv is None:
            raise ValueError(f"--dataset_csv is required for {args.dataset}")
        return cls(dataset_csv=args.dataset_csv, **common)

    root = args.data_root or args.rgbs_root or yaml_root(args.dataset)
    if root is None:
        raise ValueError(f"--data_root is required for {args.dataset} "
                         f"(no configs/dataset/{args.dataset.lower()}.yaml)")
    if args.dataset == "Sintel":
        return cls(rgbs_root=root, depths_root=args.depths_root or root, **common)
    return cls(root=root, **common)


def iter_clips(args):
    """Yields (clip_name, rgb) with rgb (1, T, C, H, W) float in [0, 1]."""
    if args.dataset in DATASETS:
        dataset = build_dataset(args)
        for i in range(min(args.max_clips, len(dataset))):
            # These datasets sample a random start offset per __getitem__;
            # reseed so a rerun dumps the same clips (as the old eval did).
            torch.manual_seed(42 + i)
            rgbs_clip = dataset[i]["rgbs_clip"]              # (3, T, H, W) in [-1, 1]
            yield f"clip_{i:04d}", (rgbs_clip.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        return

    # ponytail: folder / video = one clip, sliced by the window logic below.
    # Per-scene subfolders aren't handled; point --dataset at one scene.
    path = Path(args.dataset)
    if path.is_dir():
        files = natsorted(p for p in path.iterdir()
                          if p.suffix.lower() in IMG_EXTS)
        if not files:
            raise ValueError(f"No images in {path} (extensions: {IMG_EXTS})")
        imgs = np.stack([np.array(Image.open(p).convert("RGB")) for p in files])
        rgb = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255.0).unsqueeze(0)
    elif path.is_file():
        rgb, _ = read_video(str(path))
    else:
        raise ValueError(f"--dataset '{args.dataset}' is neither a registered dataset "
                         f"({sorted(DATASETS)}), a folder, nor a file.")

    yield path.stem, rgb[:, :args.max_frames]


def size_variants(rgb, args):
    """Resulting size -> (resize target, bin names), for the five sizes we predict at.

    Keyed by resulting pixel size, so a source already at one of them (or two
    targets landing on the same size) runs a single inference pass shared by
    every bin. 512/1024 are widths with the aspect ratio kept, matching the old
    288x512 / 576x1024 training configs.

    The size is probed on a 1-frame slice and only the target is returned, so
    main() resizes one clip at a time -- a 300-frame 4K variant is 30 GB.
    4K is probed last: if it OOMs, the cheaper bins are already on disk.
    """
    H, W = rgb.shape[-2:]
    targets = {
        "train": (args.height, args.width),        # DVD's covering resize
        "512": (round(H * 512 / W), 512),
        "1024": (round(H * 1024 / W), 1024),
        "original": (H, W),                        # /16 alignment only, no scaling
        # 4K upscales a smaller source but never shrinks one that is already >= 4K
        # wide -- such a source falls through to "original" and shares that pass.
        "4k": (round(H * max(1.0, UHD_WIDTH / W)), max(W, UHD_WIDTH)),
    }
    variants = {}
    for name, target in targets.items():
        # The target itself is carried, not the probed size: resize_for_training_scale
        # is not idempotent (1080p -> 1088x1920 -> 1088x1936).
        probe, _ = resize_for_training_scale(rgb[:, :1], *target)
        variants.setdefault(tuple(probe.shape[-2:]), (target, []))[1].append(name)
    return variants


def save_grayscale(depth, out_dir):
    """depth (T, H, W, 3) -> grayscale PNGs, normalized over the whole clip."""
    d = depth.mean(axis=-1)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)  # per-clip: per-frame flickers
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(d):
        u8 = (frame * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(u8, mode="L").save(out_dir / f"frame_{i:04d}.png")


def save_rgb(rgb_in, out_dir):
    """rgb_in (1, T, C, H, W) in [0, 1] -> the PNGs the model actually saw."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = rgb_in[0].permute(0, 2, 3, 1).cpu().numpy()
    for i, frame in enumerate(frames):
        u8 = (frame * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(u8, mode="RGB").save(out_dir / f"frame_{i:04d}.png")


def parse_args():
    p = argparse.ArgumentParser("Dump grayscale depth PNGs")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model_config", default="ckpt/model_config.yaml")
    p.add_argument("--dataset", type=str, required=True,
                   help=f"one of {sorted(DATASETS)}, a folder of frames, or a video file")
    p.add_argument("--data_root", type=str, default=None,
                   help="dataset root; defaults to configs/dataset/<name>.yaml root_dir")
    p.add_argument("--rgbs_root", type=str, default=None, help="alias for --data_root")
    p.add_argument("--depths_root", type=str, default=None, help="Sintel only")
    p.add_argument("--dataset_csv", type=str, default=None,
                   help="required for UnrealEXR / DepthCollapse")
    p.add_argument("-o", "--outputs_path", type=str, default="./depth_dump")
    p.add_argument("--max_clips", type=int, default=10)
    p.add_argument("--clip_len", type=int, default=16,
                   help="frames per clip, registered datasets only. Must be <= the "
                        "shortest scene or every sample resamples (RecursionError).")
    p.add_argument("--max_frames", type=int, default=None,
                   help="truncate a folder/video to this many frames")
    p.add_argument("--window_size", type=int, default=81)
    p.add_argument("--overlap", type=int, default=21)
    # The "train" bin only; the other three sizes are 512, 1024 and the source's own.
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    return p.parse_args()


def check_windows():
    """get_window_index must tile [0, T) with chained, full-length windows."""
    for T in (1, 10, 81, 82, 100, 300):
        w = get_window_index(T, 81, 21)
        assert w[0][0] == 0 and w[-1][1] == T, w
        assert all(b > a for a, b in w), w
        assert all(w[i + 1][0] <= w[i][1] for i in range(len(w) - 1)), w
        if T > 81:
            assert all(b - a == 81 for a, b in w), w


def check_size_variants():
    """A source already at a target size must reuse that pass, not re-run it."""
    a = argparse.Namespace(height=480, width=640)

    v = size_variants(torch.zeros(1, 2, 3, 576, 1024), a)   # already the 1024 bin
    at = {b: s for s, (_, bins) in v.items() for b in bins}
    assert at == {"train": (480, 864), "512": (288, 512), "1024": (576, 1024),
                  "4k": (2160, 3840), "original": (576, 1024)}, at
    assert len(v) == 4, v.keys()          # 1024 + original share one inference
    assert all(h % 16 == 0 and w % 16 == 0 for h, w in v), v.keys()

    # A source past 4K is never shrunk to it: "4k" rides along with "original".
    v = size_variants(torch.zeros(1, 2, 3, 2160, 4096), a)
    at = {b: s for s, (_, bins) in v.items() for b in bins}
    assert at["4k"] == at["original"] == (2160, 4096), at
    assert all(h % 16 == 0 and w % 16 == 0 for h, w in v), v.keys()

    # main() applies the target to the full clip, so it must land on the key
    # size the 1-frame probe reported -- 1080p is where a non-idempotent
    # re-resize would drift (1088x1920 -> 1088x1936).
    clip = torch.zeros(1, 2, 3, 1080, 1920)
    for size, (target, _) in size_variants(clip, a).items():
        assert resize_for_training_scale(clip, *target)[0].shape[-2:] == size, target


def main():
    args = parse_args()
    model = load_model(args.ckpt, OmegaConf.load(args.model_config))
    out_root = Path(args.outputs_path)

    for name, rgb in iter_clips(args):
        for (h, w), (target, bins) in size_variants(rgb, args).items():
            print(f"\n=== {name} @ {h}x{w} -> {bins}")
            rgb_in, _ = resize_for_training_scale(rgb, *target)
            with torch.no_grad():
                depth = generate_depth_sliced(
                    model, rgb_in, args.window_size, args.overlap)[0]
            for b in bins:
                save_grayscale(depth, out_root / b / "depth" / name)
                save_rgb(rgb_in, out_root / b / "rgb" / name)
            print(f"Wrote {depth.shape[0]} frames to "
                  + ", ".join(str(out_root / b / "{depth,rgb}" / name) for b in bins))


if __name__ == "__main__":
    check_windows()
    check_size_variants()
    main()
