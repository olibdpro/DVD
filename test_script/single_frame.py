"""Single frame in, full-resolution depth out.

test_single_video.py resizes the depth back to the source resolution and only
writes an mp4 -- both wrong when the point is a 4K depth map from a 1K source.
This keeps the depth at the inference resolution and writes raw float32 + 16-bit
PNG, and reports the CUDA peak so tile geometry can be tuned against a number.

Every tiling flag has the same name and meaning as in test_single_video.py.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsynth.models.wan_video_dit import WanModel  # noqa: E402
from diffsynth.models.wan_video_vae import WanVideoVAE  # noqa: E402
from diffsynth.pipelines.wan_video_new_determine import \
    WanVideoPipeline  # noqa: E402
from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule  # noqa: E402
from test_script.test_single_video import (  # noqa: E402
    check_latent_ref, check_latent_tile, check_vae_tile, generate_depth_sliced,
    resize_for_training_scale)

# Wan2.1-T2V-1.3B, the config wan_video_dit.py:789-802 hash-detects from the
# base weights. Hardcoded because we never load them: the DVD checkpoint carries
# all 1425 DiT tensors (LoRA included) and all 194 VAE tensors, so the 6.2 GB
# Wan-AI download the normal path insists on is dead weight -- verified key by
# key against the checkpoint header, zero missing/extra/mismatched.
WAN_T2V_1_3B = dict(has_image_input=False, patch_size=[1, 2, 2], in_dim=16, dim=1536,
                    ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
                    num_heads=12, num_layers=30, eps=1e-6)


def load_model_offline(ckpt_path, yaml_args, device="cuda"):
    """load_model() without the base-model download: build the modules from
    config, let the strict load fill them from the DVD checkpoint.

    device="cpu" runs in float32 -- slow, but it exercises the whole tiling path
    when the GPU is unavailable."""
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def from_pretrained(torch_dtype=torch.bfloat16, device=device, **_):
        pipe = WanVideoPipeline(device=device, torch_dtype=dtype)
        pipe.dit = WanModel(**WAN_T2V_1_3B).to(dtype=dtype)
        pipe.vae = WanVideoVAE(z_dim=16).to(dtype=dtype)
        pipe.image_encoder = pipe.motion_controller = pipe.vace = None
        return pipe

    original, WanVideoPipeline.from_pretrained = WanVideoPipeline.from_pretrained, \
        staticmethod(from_pretrained)
    try:
        model = WanTrainingModule(
            accelerator=Accelerator(), args=yaml_args, trainable_models=None,
            model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
            use_gradient_checkpointing=False, lora_rank=yaml_args.lora_rank,
            lora_base_model=yaml_args.lora_base_model)
    finally:
        WanVideoPipeline.from_pretrained = original

    state_dict = load_file(ckpt_path, device="cpu")
    for name in ("dit", "vae"):
        prefix = f"pipe.{name}."
        getattr(model.pipe, name).load_state_dict(
            {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)},
            strict=True)
    del state_dict
    model.merge_lora_layer()
    model.pipe.device = device
    model.pipe.torch_dtype = dtype
    return model.to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True,
                   help="one PNG, or a directory of frames (see --frames)")
    p.add_argument("--frames", type=int, default=1,
                   help="how many frames to take when --image is a directory. >1 "
                        "exercises the temporal windowing, where the per-window "
                        "anchor has to agree with its neighbour's.")
    p.add_argument("--window_size", type=int, default=None,
                   help="temporal window; defaults to --frames (one window)")
    p.add_argument("--overlap", type=int, default=0)
    p.add_argument("--output_dir", default="./inference_results")
    p.add_argument("--model_config", default="ckpt/model_config.yaml")
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--tag", default="", help="suffix on the output names, to keep runs apart")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--tile_size", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--tile_stride", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--spatial_tile", type=int, nargs=2, default=None, metavar=("H", "W"))
    p.add_argument("--spatial_tile_overlap", type=int, default=64)
    p.add_argument("--spatial_ref_width", type=int, default=0)
    p.add_argument("--latent_tile", "--latent-tile", type=int, nargs=2, default=None,
                   metavar=("H", "W"))
    p.add_argument("--latent_tile_overlap", "--latent-tile-overlap", type=int, default=8)
    p.add_argument("--latent_ref", "--latent-ref", type=int, nargs=2, default=None,
                   metavar=("H", "W"))
    p.add_argument("--latent_band_merge", "--latent-band-merge", action="store_true",
                   help="anchor supplies the low band, tiles only the detail above it")
    args = p.parse_args()
    check_vae_tile(args.tile_size, args.tile_stride, args.tiled)
    check_latent_tile(args.latent_tile, args.latent_tile_overlap)
    check_latent_ref(args.latent_ref, args.latent_band_merge,
                     args.spatial_ref_width)
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.image):
        paths = sorted(p for p in os.listdir(args.image)
                       if p.lower().endswith((".png", ".jpg", ".jpeg")))[:args.frames]
        if not paths:
            raise FileNotFoundError(f"no frames in {args.image}")
        paths = [os.path.join(args.image, p) for p in paths]
    else:
        paths = [args.image]
    stack = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        stack.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rgb = stack[0]
    frame = torch.from_numpy(np.stack(stack)).permute(0, 3, 1, 2).float().div_(255.0)[None]
    frame, _ = resize_for_training_scale(frame, args.height, args.width)
    _, _, _, H, W = frame.shape
    print(f"input {rgb.shape[1]}x{rgb.shape[0]} -> inference {W}x{H} "
          f"(latent {H // 8}x{W // 8}, {(H // 16) * (W // 16)} DiT tokens)")

    model = load_model_offline(args.ckpt, OmegaConf.load(args.model_config), args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    depth = generate_depth_sliced(
        model, frame, window_size=args.window_size or frame.shape[1],
        overlap=args.overlap, tiled=args.tiled,
        spatial_tile=args.spatial_tile, spatial_overlap=args.spatial_tile_overlap,
        tile_size=args.tile_size, tile_stride=args.tile_stride,
        latent_tile=args.latent_tile, latent_tile_overlap=args.latent_tile_overlap,
        spatial_ref_width=args.spatial_ref_width, latent_ref=args.latent_ref,
        latent_band_merge=args.latent_band_merge)[0]
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0.0
    print(f"done in {time.time() - t0:.1f}s, CUDA peak {peak:.2f} GiB")

    # Pipeline returns [T, H, W, 3] in [0, 1]; the three channels are a replicated
    # disparity, so one of them is the depth.
    all_d = np.asarray(depth, dtype=np.float32)
    all_d = all_d[..., 0] if all_d.ndim == 4 else all_d
    print(f"depth {all_d.shape} range {all_d.min():.4f}-{all_d.max():.4f}")

    base = os.path.splitext(os.path.basename(paths[0]))[0]
    for i, d in enumerate(all_d):
        stem = base + (f"_{i:04d}" if len(all_d) > 1 else "") + (f"_{args.tag}" if args.tag else "")
        np.save(os.path.join(args.output_dir, f"{stem}.npy"), d)
        # Normalized per frame ONLY for the preview; the .npy keeps raw values,
        # which is what any cross-frame comparison has to use.
        norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_depth16.png"),
                    (norm * 65535).astype(np.uint16))
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_vis.png"),
                    cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
    print(f"wrote {len(all_d)} frame(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
