import argparse
import gc
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new_determine import SpatialTiler_BCTHW
from examples.wanvideo.model_training.WanTrainingModule import \
    WanTrainingModule


# The CUDA and host allocators word exhaustion differently:
# "CUDA out of memory" vs "DefaultCPUAllocator: can't allocate memory".
OOM_MARKERS = ("out of memory", "can't allocate memory",
               "cannot allocate memory", "not enough memory")


def is_oom(err):
    """True if err is an allocator running dry rather than a real bug.

    torch.cuda.OutOfMemoryError subclasses RuntimeError, so callers catch
    RuntimeError and use this to decide whether to re-raise.
    """
    return any(m in str(err).lower() for m in OOM_MARKERS)


# =============================
# Helper: Math & Alignment
# =============================
def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    """Computes scale and shift for overlap alignment."""
    if mask is None:
        mask = np.ones_like(ref_frames)

    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0

    return scale, shift


# =============================
# Helper: Video Processing
# =============================
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(
        video_np).permute(0, 3, 1, 2).float() / 255.0

    return video_tensor.unsqueeze(0), fps   # [1, T, C, H, W], fps


def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = int(np.ceil(H * ratio))
    new_W = int(np.ceil(W * ratio))

    # Align to 16
    new_H = (new_H + 15) // 16 * 16
    new_W = (new_W + 15) // 16 * 16

    if new_H == H and new_W == W:
        return video_tensor, (H, W)

    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(
        new_H, new_W), mode="bilinear", align_corners=False)
    resized = resized.view(B, T, C, new_H, new_W)
    return resized, (H, W)


def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=(
        orig_H, orig_W), mode='bilinear', align_corners=False)
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()


def pad_time_mod4(video_tensor):
    """Pads the temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            # Last window ensures full window_size length if possible
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res


def get_spatial_tile_index(length, tile, overlap):
    """[start, end) pairs covering [0, length), mirroring get_window_index:
    every tile is full-size (except length <= tile) and consecutive tiles
    overlap by at least `overlap`; the last tile is snapped to end at `length`.
    """
    if length <= tile:
        return [(0, length)]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(
            f"spatial overlap ({overlap}) must be smaller than the tile ({tile})")
    index = []
    start = 0
    while start + tile < length:
        index.append((start, start + tile))
        start += stride
    index.append((length - tile, length))
    return index


def spatial_blend_mask(h, h_, w, w_, H, W, overlap_h, overlap_w):
    """Feather weight for one tile: 1 inside, linear ramps on edges that are
    not the frame border, so neighbouring tiles cross-fade. Same scheme as the
    VAE's build_mask, but applied to the predicted depth instead of latents."""
    def ramp(length, at_start_border, at_end_border, overlap):
        overlap = min(overlap, length)
        x = np.ones(length, dtype=np.float32)
        if not at_start_border and overlap > 0:
            x[:overlap] = np.arange(1, overlap + 1, dtype=np.float32) / overlap
        if not at_end_border and overlap > 0:
            x[-overlap:] = np.arange(overlap, 0, -1, dtype=np.float32) / overlap
        return x
    mask_h = ramp(h_ - h, h == 0, h_ == H, overlap_h)
    mask_w = ramp(w_ - w, w == 0, w_ == W, overlap_w)
    return np.minimum(mask_h[:, None], mask_w[None, :])


def _run_pipe(model, rgb_slice, tiled, tile_size=None, tile_stride=None,
              latent_tile=None, latent_tile_overlap=8):
    """One pipeline call on a (B, T, C, H, W) slice. Spatial crops go through
    the exact same path as full frames, so conditioning stays in-distribution."""
    B, T, _, H, W = rgb_slice.shape
    # Omitted when unset so the pipeline's own tile defaults stay authoritative.
    tile_kwargs: dict = {k: tuple(v) for k, v in (("tile_size", tile_size),
                                                  ("tile_stride", tile_stride)) if v}
    if latent_tile is not None:
        tile_kwargs["latent_tile_size"] = tuple(latent_tile)
        tile_kwargs["latent_tile_overlap"] = latent_tile_overlap
    return model.pipe(
        prompt=[""] * B,
        negative_prompt=[""] * B,
        mode=model.args.mode,
        height=H,
        width=W,
        num_frames=T,
        batch_size=B,
        input_image=rgb_slice[:, 0],
        extra_images=rgb_slice,
        extra_image_frame_index=torch.ones([B, T]).to(model.pipe.device),
        input_video=rgb_slice,
        cfg_scale=1,
        seed=0,
        tiled=tiled,
        **tile_kwargs,
        denoise_step=model.args.denoise_step,
    )


def _run_pipe_spatial_tiled(model, rgb_window, origin_T, tiled,
                            spatial_tile, spatial_overlap,
                            tile_size=None, tile_stride=None,
                            latent_tile=None, latent_tile_overlap=8,
                            ref_width=0):
    """Depth for one temporal window, inferred on overlapping spatial crops and
    feather-blended back to full frame. Each crop runs the whole pipeline
    (VAE + DiT) on its own, so the VAE's global spatial attention stays intact
    and the conditioning is exactly what the model saw in training -- unlike
    `tiled`, which shards the VAE's attention across latent tiles.

    ref_width: if > 0, first run one extra full-window pass at this width
    (aspect kept -- small enough to fit untiled), upsample its depth to the
    window size, and least-squares align every crop to that reference over the
    crop's FULL extent before blending. Without the anchor, each crop picks
    its own global depth range (crops mostly agree on textured content but
    drift apart on featureless regions, quilting the blend); seam-band-only
    alignment can't fix global drift."""
    tile_h, tile_w = spatial_tile
    if tile_h % 16 or tile_w % 16:
        raise ValueError(
            "spatial_tile must be 16-aligned (VAE 8x downsample + DiT 2x "
            f"patchify), got {spatial_tile}")
    _, _, _, H, W = rgb_window.shape
    h_index = get_spatial_tile_index(H, tile_h, spatial_overlap)
    w_index = get_spatial_tile_index(W, tile_w, spatial_overlap)

    ref = None
    if ref_width:
        # The anchor carries only global structure; it is intentionally NOT
        # latent-tiled (one coherent pass) and is cheap at this size.
        ref_rgb, _ = resize_for_training_scale(
            rgb_window, round(H * ref_width / W), ref_width)
        ref_depth = _run_pipe(model, ref_rgb, tiled,
                              tile_size, tile_stride)['depth'][:, :origin_T]
        B, Tr = ref_depth.shape[0], ref_depth.shape[1]
        ref_t = torch.from_numpy(ref_depth).permute(0, 1, 4, 2, 3).float()
        ref_t = F.interpolate(ref_t.reshape(B * Tr, *ref_t.shape[2:]),
                              size=(H, W), mode="bilinear", align_corners=False)
        ref = ref_t.view(B, Tr, *ref_t.shape[2:]).permute(0, 1, 3, 4, 2).numpy()
        del ref_rgb, ref_depth, ref_t

    B = rgb_window.shape[0]
    values, weight = None, np.zeros((H, W), dtype=np.float32)
    for h, h_ in h_index:
        for w, w_ in w_index:
            outputs = _run_pipe(model, rgb_window[:, :, :, h:h_, w:w_], tiled,
                                tile_size, tile_stride,
                                latent_tile, latent_tile_overlap)
            tile_depth = outputs['depth'][:, :origin_T]
            if ref is not None:
                scale, shift = compute_scale_and_shift(
                    tile_depth, ref[:, :, h:h_, w:w_])
                scale = np.clip(scale, 0.7, 1.5)
                print(f"  crop ({h}:{h_}, {w}:{w_}): anchor scale={scale:.4f} "
                      f"shift={shift:.4f}")
                tile_depth = tile_depth * scale + shift
            mask = spatial_blend_mask(h, h_, w, w_, H, W,
                                      spatial_overlap, spatial_overlap)
            if values is None:
                values = np.zeros(
                    (B, origin_T, H, W, tile_depth.shape[-1]), dtype=np.float32)
            values[:, :, h:h_, w:w_] += tile_depth * mask[None, None, :, :, None]
            weight[h:h_, w:w_] += mask
    return values / weight[None, None, :, :, None]


def check_vae_tile(tile_size, tile_stride, tiled=True):
    """Reject VAE tile geometry the VAE would silently turn into NaN."""
    if (tile_size is None) != (tile_stride is None):
        raise ValueError("tile_size and tile_stride must be given together -- "
                         "one alone can end up larger than the other's default")
    if tile_size is None:
        return
    if min(tile_size) < 1 or min(tile_stride) < 1:
        raise ValueError(f"tile_size {tuple(tile_size)} and tile_stride "
                         f"{tuple(tile_stride)} must be positive")
    if tile_stride[0] > tile_size[0] or tile_stride[1] > tile_size[1]:
        # tiled_encode/decode step by stride and cover size, so a stride past
        # the tile leaves latents no tile reaches; their blend weight stays 0
        # and the divide at the end turns them into NaN.
        raise ValueError(f"tile_stride {tuple(tile_stride)} must not exceed "
                         f"tile_size {tuple(tile_size)}")
    if not tiled:
        print("Warning: --tile_size/--tile_stride only apply to --tiled, which "
              "is off; ignoring them. (--spatial_tile has its own geometry.)")


def check_latent_tile(latent_tile, latent_tile_overlap):
    """Reject latent-tile geometry the DiT cannot patchify or the tiler cannot cover."""
    if latent_tile is None:
        return
    if min(latent_tile) < 2 or any(t % 2 for t in latent_tile):
        raise ValueError(f"--latent_tile {tuple(latent_tile)} must be positive and "
                         "even (the DiT patchifies 2x2; odd tiles lose a row/column)")
    if latent_tile_overlap < 0 or latent_tile_overlap % 2:
        raise ValueError(f"--latent_tile_overlap {latent_tile_overlap} must be even "
                         "and non-negative (an odd overlap gives odd tile starts)")
    if latent_tile_overlap >= min(latent_tile):
        # stride = size - overlap would be <= 0: the tile loop would never advance.
        raise ValueError(f"--latent_tile_overlap {latent_tile_overlap} must be "
                         f"smaller than --latent_tile {tuple(latent_tile)}")


# =============================
# Core Inference
# =============================
def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, scale_only=False,
                          partial_on_oom=False, tiled=False,
                          spatial_tile=None, spatial_overlap=64,
                          tile_size=None, tile_stride=None,
                          latent_tile=None, latent_tile_overlap=8,
                          spatial_ref_width=0):
    """partial_on_oom: stop at the window that ran out of memory and return the ones
    already computed (None if the first one failed), instead of propagating. Off by
    default -- silently short output is the wrong answer for callers writing a video.

    tiled: run the VAE encode/decode in tiles. Trades speed for a lower memory peak
    without touching the window, which is the part that carries temporal context.
    Approximate: the VAE's global spatial attention is sharded per tile, which
    visibly degrades depth output. Prefer spatial_tile.

    spatial_tile: None, or (tile_h, tile_w) to infer every window as overlapping
    spatial crops of that size, feather-blended in pixel space. The pipeline-level
    alternative to tiled: every crop runs the full pipeline on its own, so the
    VAE's attention and the training distribution are preserved. Crops are blended
    raw (the model is deterministic and overlapping crops mostly agree); widen
    spatial_overlap if seams show. Tile sizes must be 16-aligned.

    tile_size/tile_stride: the VAE tile geometry `tiled` uses, in latent units
    (1 unit = 8 px). Left None, the pipeline's own defaults apply. Smaller tiles
    lower the peak further; a shorter stride widens the blend seam.

    latent_tile: None, or (tile_h, tile_w) in latent units (1 unit = 8 px) to run
    the DiT on overlapping spatial tiles of the latent and feather-blend the
    predicted depth latents, followed by one global VAE decode. This shrinks the
    DiT memory peak (tokens scale with tile area) -- the part that OOMs at large
    inputs -- while temporal context and the VAE's attention both stay whole,
    unlike `tiled` (per-tile VAE attention) or a smaller window (less temporal
    context). Tiles must be even (DiT patchify is 2x2). Each tile builds its own
    0-based RoPE, which under independent per-tile passes is identical to sliced
    global coordinates, so no positional handling is needed.
    latent_tile_overlap: feather width between latent tiles, in latent units;
    even and smaller than the tile (default 8 = 64 px).

    spatial_ref_width: if > 0 and spatial_tile is set, anchor every crop to a
    coherent global reference: one extra full-window pipeline pass at this
    width (aspect kept, small enough to run untiled), upscaled back and used
    as the full-extent scale/shift LSQ target per crop. Removes the per-crop
    depth-calibration drift that quilts featureless content; 0 = off.
    """
    check_vae_tile(tile_size, tile_stride, tiled)
    check_latent_tile(latent_tile, latent_tile_overlap)
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")
    if latent_tile is not None:
        lh, lw = latent_tile
        n_h = len(SpatialTiler_BCTHW.tile_index(H // 8, lh, lh - latent_tile_overlap))
        n_w = len(SpatialTiler_BCTHW.tile_index(W // 8, lw, lw - latent_tile_overlap))
        print(f"latent tiling: {n_h}x{n_w} DiT tiles of {lh}x{lw} latents "
              f"(overlap {latent_tile_overlap}) per window")
    if spatial_tile is not None:
        n_h = len(get_spatial_tile_index(H, spatial_tile[0], spatial_overlap))
        n_w = len(get_spatial_tile_index(W, spatial_tile[1], spatial_overlap))
        print(f"spatial tiling: {n_h}x{n_w} crops of "
              f"{spatial_tile[0]}x{spatial_tile[1]} (overlap {spatial_overlap}) "
              f"per window")
        if spatial_ref_width:
            print(f"  anchored: one extra full-window pass at width "
                  f"{spatial_ref_width}, crops LSQ-aligned to it")
    elif spatial_ref_width:
        print("Warning: --spatial_ref_width only applies to --spatial_tile; "
              "ignoring it.")

    depth_res_list = []

    # 1. Inference per window
    for start, end in tqdm(depth_windows, desc="Inferencing Slices"):
        _input_rgb_slice = input_rgb[:, start:end]

        # Ensure 4n+1 padding
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)

        try:
            if spatial_tile is None:
                outputs = _run_pipe(model, _input_rgb_slice, tiled,
                                    tile_size, tile_stride,
                                    latent_tile, latent_tile_overlap)
                depth_window = outputs['depth'][:, :origin_T]
            else:
                depth_window = _run_pipe_spatial_tiled(
                    model, _input_rgb_slice, origin_T, tiled,
                    spatial_tile, spatial_overlap, tile_size, tile_stride,
                    latent_tile, latent_tile_overlap,
                    ref_width=spatial_ref_width)
        except RuntimeError as err:
            if not (partial_on_oom and is_oom(err)):
                raise
            print(f"\nOut of memory on window {start}-{end} at "
                  f"{H}x{W}. Keeping the "
                  f"{len(depth_res_list)} window(s) already computed.")
            del _input_rgb_slice
            # A caught OOM holds the failed graph alive through frame locals;
            # without this the caller's save path can OOM too.
            gc.collect()
            torch.cuda.empty_cache()
            break
        # Drop the padded frames
        depth_res_list.append(depth_window)

    if not depth_res_list:
        return None

    # 2. Overlap Alignment
    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        print(f"Handling window {i} start: {start}, end: {end}")

        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / \
                    (np.sum(curr_frames * curr_frames) + 1e-6)
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)

            scale = np.clip(scale, 0.7, 1.5)

            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            # Debugging Output
            curr_overlap_aligned = aligned_t[:, :real_overlap]
            diff = np.abs(curr_overlap_aligned - ref_frames)
            mae_scalar = float(
                diff.mean(axis=tuple(range(1, diff.ndim))).mean())

            print(f"\n[Overlap {i}]")
            print(f"real_overlap = {real_overlap}")
            print(f"scale = {scale:.8f}, shift = {shift:.8f}")
            print(
                f"aligned curr range = {aligned_t.min():.6f} ~ {aligned_t.max():.6f}")
            print(f"overlap MAE(after align) = {mae_scalar:.6f}")

            # Smooth blending
            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + \
                alpha * aligned_t[:, :real_overlap]

            depth_list_aligned = np.concatenate(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap,
                 aligned_t[:, real_overlap:]], axis=1
            )
        else:
            # Fallback if no overlap exists
            depth_list_aligned = np.concatenate(
                [depth_list_aligned, t], axis=1)

        print(
            f"Total depth range after concat = {depth_list_aligned.min():.6f} ~ {depth_list_aligned.max():.6f}")
        prev_end = end

    # Crop to original length
    return depth_list_aligned[:, :T]


# =============================
# Pipeline Components
# =============================
def load_model(ckpt_dir, yaml_args):
    """Initializes and loads the model checkpoint."""
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
    )

    # Accept the .safetensors itself (e.g. ckpt/dvd_1.1.safetensors) or a dir holding
    # model.safetensors, so v1.0/v1.1 can sit side by side without renaming.
    ckpt_path = ckpt_dir if os.path.isfile(ckpt_dir) else os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (--ckpt {ckpt_dir})")
    state_dict = load_file(ckpt_path, device="cpu")
    dit_state_dict = {k.replace("pipe.dit.", ""): v for k,
                      v in state_dict.items() if "pipe.dit." in k}
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()
    model = model.to("cuda")
    
    return model


def load_video_data(args):
    """Loads and resizes the input video."""
    input_tensor, origin_fps = read_video(args.input_video)
    print("Original shape:", input_tensor.shape)

    input_tensor, orig_size = resize_for_training_scale(
        input_tensor, args.height, args.width)
    print("Resized shape:", input_tensor.shape)
    print(f"input range {input_tensor.min()} - {input_tensor.max()}")

    return input_tensor, orig_size, origin_fps


def predict_depth(model, input_tensor, orig_size, args):
    """Runs depth prediction and post-processes the output to original size."""
    depth = generate_depth_sliced(
        model, input_tensor, args.window_size, args.overlap, tiled=args.tiled,
        spatial_tile=args.spatial_tile,
        spatial_overlap=args.spatial_tile_overlap,
        tile_size=args.tile_size, tile_stride=args.tile_stride,
        latent_tile=args.latent_tile,
        latent_tile_overlap=args.latent_tile_overlap,
        spatial_ref_width=args.spatial_ref_width)[0]
    print(f"depth range shape {depth.min()} - {depth.max()}, shape {depth.shape}")

    # Post Process: resize back to original
    depth = resize_depth_back(depth, orig_size)
    print(f"after resizing {depth.min()} - {depth.max()}, {depth.shape}")

    return depth


def save_results(depth, origin_fps, args):
    """Normalizes and saves the depth video to disk."""
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.input_video).split('.')[0]
    gray_scale = 'gray' if args.grayscale else 'color'
    out_prefix = os.path.join(
        args.output_dir, f"{base_name}_{gray_scale}")

    output_path = f"{out_prefix}_depth_vis.mp4"
    print(f"Saving to {output_path}")
    d_min, d_max = depth.min(), depth.max()
    vis_depth = (depth - d_min) / (d_max - d_min + 1e-8)
    
    save_video(vis_depth, output_path,
               fps=origin_fps, quality=6, grayscale=args.grayscale)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./inference_results")
    parser.add_argument('--model_config', default='ckpt/model_config.yaml')
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--tiled', action='store_true',
                        help="tile the VAE encode/decode: slower, lower memory peak, "
                             "but degrades output (per-tile VAE attention); "
                             "--spatial_tile is the quality-preserving alternative")
    parser.add_argument('--tile_size', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help="VAE tile geometry for --tiled, in latent units "
                             "(1 unit = 8 px). Default: the pipeline's own (30 52). "
                             "Smaller tiles lower the memory peak further.")
    parser.add_argument('--tile_stride', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help="step between VAE tiles, in latent units; must be <= "
                             "--tile_size, and is required alongside it. "
                             "Default: the pipeline's own (15 26).")
    parser.add_argument('--spatial_tile', type=int, nargs=2, default=None,
                        metavar=('H', 'W'),
                        help="pipeline-level spatial tiling: run each temporal window "
                             "as overlapping HxW crops and feather-blend the depth. "
                             "Unlike --tiled, the VAE stays whole per crop, so output "
                             "quality is preserved. H W must be 16-aligned.")
    parser.add_argument('--spatial_tile_overlap', type=int, default=64,
                        help="feather width between spatial crops in pixels")
    parser.add_argument('--latent_tile', '--latent-tile', type=int, nargs=2,
                        default=None, metavar=('H', 'W'),
                        help="run the DiT on overlapping HxW tiles of the latent "
                             "(1 unit = 8 px) and feather-blend the depth latents "
                             "before one global VAE decode. Shrinks the DiT peak -- "
                             "what OOMs at large sizes -- while temporal context and "
                             "the VAE both stay whole (unlike --tiled). H W must be "
                             "even. Combine with --tiled to bound the VAE too.")
    parser.add_argument('--latent_tile_overlap', '--latent-tile-overlap', type=int,
                        default=8, metavar='N',
                        help="feather width between latent tiles, in latent units; "
                             "even and < --latent_tile. Default 8 (= 64 px).")
    parser.add_argument('--spatial_ref_width', type=int, default=0, metavar='W',
                        help="with --spatial_tile: anchor crops to a coherent "
                             "global reference -- one extra full-window pass at "
                             "this width, crops LSQ-aligned to it. Removes the "
                             "calibration drift that quilts featureless content. "
                             "0 = off.")
    return parser.parse_args()


# =============================
# Main Script
# =============================
def main():
    args = parse_args()
    yaml_args = OmegaConf.load(args.model_config)

    # 1. Load Model
    model = load_model(args.ckpt, yaml_args)

    # 2. Load Video
    input_tensor, orig_size, origin_fps = load_video_data(args)

    # 3. Predict Depth
    depth = predict_depth(model, input_tensor, orig_size, args)

    # 4. Save Results
    save_results(depth, origin_fps, args)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()