import os
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageChops

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms.functional as VF

import torch.nn.functional as F

import OpenEXR, Imath
import scipy.sparse
from scipy.sparse.linalg import spsolve


from utils.utils import EPS

def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x.detach().cpu().clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()

def resize_rgbs_clip(rgbs_clip: torch.Tensor, new_size=None) -> torch.Tensor:
    """
    rgbs_clip: (3, F, H, W) float tensor in [0,1] (or normalized by normalize_rgb_images).
    Returns resized (3, F, h, w) if new_size is set; else returns as-is.
    """
    if not new_size:
        return rgbs_clip
    h, w = new_size
    x = rgbs_clip.permute(1, 0, 2, 3)                # (F,3,H,W)
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
    return x.permute(1, 0, 2, 3)                     # (3,F,h,w)

def depth_to_colormap(x: torch.Tensor) -> np.ndarray:
    d = x.squeeze(0).detach().cpu().numpy()
    d = (d - d.min()) / (d.ptp() + 1e-8)
    cm = plt.get_cmap("turbo")((d * 255).astype(np.uint8))[:, :, :3]
    return (cm * 255).astype(np.uint8)

def depth_to_rgb_like(depth: torch.Tensor) -> torch.Tensor:
    depth.clamp_(-1.0, 1.0)
    depth.repeat(3, 1, 1, 1)

    depths_rgb = depth
    return depths_rgb

def normalise_depth(depth: torch.Tensor, *, p_low=2.0, p_hi=98.0):
    """Eq.(1) from Ke et al. 2024 → depth_n  ∈ [-1,1].  Works on clips or frames."""
    B = depth.size(0)
    flat = depth.view(B, -1)
    d2 = torch.quantile(flat, p_low, dim=1, keepdim=True).view(B, 1, 1, 1, 1)
    d98 = torch.quantile(flat, p_hi, dim=1, keepdim=True).view(B, 1, 1, 1, 1)

    depth_n = ((depth - d2) / (d98 - d2 + EPS) - 0.5) * 2.0
    return depth_n, (d2, d98)

def denormalise_depth(depth_n: torch.Tensor, d2: torch.Tensor, d98: torch.Tensor):
    """Invert `normalise_depth` back to raw depth (cm or m)."""
    return ((depth_n / 2.0) + 0.5) * (d98 - d2) + d2

def denormalise(image, mean, std):
    """
    Reverse normalization on an image tensor
    Args:
        image (*, C, H, W): normalized image tensor
        mean (list or tuple, size=C): mean used for normalization
        std (list or tuple, size=C): std used for normalization
    """
    C = image.shape[-3]
    mean = torch.tensor(mean, device=image.device).view(C, 1, 1)
    std = torch.tensor(std, device=image.device).to(image.device).view(C, 1, 1)
    return image * std + mean
    
def inverse_depth(depth_raw: torch.Tensor, eps: float = 1e-4):
    """Reciprocal depth (disparity): closer → larger value, units 1/cm or 1/m."""
    return 1.0 / torch.clamp(depth_raw, min=eps)

def clip_to_strip(clip: torch.Tensor) -> torch.Tensor:  # unchanged
    strip = vutils.make_grid(clip.permute(1, 0, 2, 3), nrow=clip.shape[1], padding=1)
    return (((strip+1.0)/2.0).clamp(0, 1) * 255).byte()

def make_strip(clip):
    return vutils.make_grid(clip.permute(1, 0, 2, 3), nrow=clip.shape[1], padding=1)

def pad_images_sequence(images_sequence: torch.Tensor,      # [T,C,H,W]
                        window_size: int,
                        stride: int,):
    """
    1. Pad *back* so len % stride == 0      (stride alignment).
    2. Add symmetric context so each window of length `window_size`
       can be centred on the `stride` kept frames.
    3. Slice sliding windows and, for each, report the index (in the ORIGINAL
       sequence) of its first and last *real* frame.  Indices are clamped to
       [0, T-1] when the window extends into padding.

    window_size >= stride is required.
    """
    if window_size < stride:
        raise ValueError("`window_size` must be ≥ `stride`.")

    T_orig   = images_sequence.size(0)       # original sequence length

    # ─────────────── 1)  back-pad to multiple of stride ─────────────────── #
    T_last = T_orig % stride
    pad_last_stride = stride - T_last
    if pad_last_stride:
        back_pad = images_sequence[-1:].repeat(pad_last_stride, 1, 1, 1)
        images_sequence_aligned = torch.cat([images_sequence, back_pad], dim=0)
    else:
        images_sequence_aligned = images_sequence

    # ─────────────── 2)  symmetric context padding for window_size ──────── #
    ctx_total = window_size - stride        # total context per window
    pad_ctx_front = ctx_total // 2
    pad_ctx_back = ctx_total - pad_ctx_front

    images_sequence_padded = images_sequence_aligned

    if pad_ctx_front:
        images_sequence_ctx_front = images_sequence_padded[0:1].repeat(pad_ctx_front, 1, 1, 1)
        images_sequence_padded = torch.cat([images_sequence_ctx_front, images_sequence_padded], dim=0)

    if pad_ctx_back:
        images_sequence_ctx_back = images_sequence_padded[-1:].repeat(pad_ctx_back, 1, 1, 1)
        images_sequence_padded = torch.cat([images_sequence_padded, images_sequence_ctx_back], dim=0)

    T_pad = images_sequence_padded.size(0)

    # ─────────────── 3)  build sliding windows & collect indices ────────── #
    windows = []
    indices_windows = []

    for start_window in range(0, T_pad - window_size + 1, stride):
        end_window = start_window + window_size            # index in padded space
        windows.append(images_sequence_padded[start_window:end_window].permute(1, 0, 2, 3))

        # map padded indices → original indices
        index_first = max(0, pad_ctx_front)
        index_last  = min(window_size, pad_ctx_front + stride)
        if start_window + stride >= T_pad - window_size + 1:
            index_last  = min(window_size, pad_ctx_front + T_last)
        indices_windows.append((index_first, index_last))

    windows = torch.stack(windows)                # (N,C,window_size,H,W)
    return images_sequence_padded, windows, indices_windows

def get_image_size(path: Path):
    """
    Returns (width, height) for any supported image.
    For EXR: try OpenEXR header (fast) else imageio.
    """
    if path.suffix.lower() == ".exr":
        exr = OpenEXR.InputFile(str(path))
        dw = exr.header()["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        exr.close()
        return (width, height)
    else:
        im = Image.open(path)
        return im.size
    
def rgb_chw_normed_to_uint8_hwc(rgb_chw_normed: np.ndarray) -> np.ndarray:
    """
    Convert your loader's [-1,1] float CHW into uint8 HWC for imshow.
    """
    arr = ((rgb_chw_normed + 1.0) * 0.5).clip(0.0, 1.0)  # [0,1]
    arr = np.transpose(arr, (1, 2, 0))                   # HWC
    return (arr * 255.0 + 0.5).astype(np.uint8)

def connected_component_filter(mask, min_size=8):
    """Given a binary mask, remove connected components smaller than min_size.
    i.e., filter out "floating pixels"
    """
    mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) if isinstance(mask, torch.Tensor)  else mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8, ltype=cv2.CV_32S)
    filtered_mask_cca = np.zeros_like(mask_np)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask_cca[labels == i] = 1.0
    return  torch.from_numpy(filtered_mask_cca).unsqueeze(0).to(mask.dtype) if isinstance(mask, torch.Tensor) else filtered_mask_cca

def dilate(mask, kernel_size=3, iterations=1, kernel=None):
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(mask.cpu().numpy().squeeze() if isinstance(mask, torch.Tensor) else mask, kernel, iterations=iterations)
    return torch.from_numpy(dilated_mask).unsqueeze(0) if isinstance(mask, torch.Tensor) else dilated_mask

def shift_image_horizontal(image, shift_pixels):
    """
    Shifts an image horizontally by a specified number of pixels (positive values shift right, negative values shift left).
    """
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    shifted_img = ImageChops.offset(img, shift_pixels, 0)
    if isinstance(image, np.ndarray):
        return np.array(shifted_img)
    return shifted_img

def combine_images(source: torch.Tensor, target: torch.Tensor, source_mask: torch.Tensor):
    """ Combine two images according to a mask. 
    Args:
        source (torch.Tensor): image
        target (torch.Tensor): image
        source_mask (torch.Tensor): A mask where non-0 indicates the region to take from the source and copy to target. (No blending).
    """
    return torch.where(source_mask != 0, source, target)

def seamless_clone(source, target, source_mask):
    h, w = target.shape[-2:]
    if len(source.shape) < 4:
        source = source[None]
        target = target[None]
        source_mask = source_mask[None]

    def tensor2np(x):
        return np.array(VF.to_pil_image(x))

    output = []
    for s,t,m in zip(source, target, source_mask):
        try:
            source_np, target_np, source_mask_np = map(tensor2np, [s, t, m])
            if source_mask_np.shape[-1] == 1:
                source_mask_np =  np.expand_dims(source_mask_np, -1).repeat(3, axis=2)
            source_mask_np = np.clip(source_mask_np, 0, 1)
            img = cv2.seamlessClone(source_np, target_np, source_mask_np, (w//2, h//2), cv2.NORMAL_CLONE)
            output.append(VF.to_tensor(img))
        except Exception as e:
            print(source.shape, target.shape, source_mask.shape)
            print(source_np.shape, target_np.shape, source_mask_np.shape)
            raise e
    return torch.stack(output)
    
def poisson_blend(source, target, mask):
    """
    Solves the Poisson equation for arbitrary non-continuous masks.
    Handles boundaries correctly without IndexErrors.
    """

    def poisson_blend_single_frame(source, target, mask):
        device = source.device
        source_np = source.cpu().numpy()
        target_np = target.cpu().numpy()
        mask_np = mask.cpu().numpy().squeeze()
        if len(mask_np.shape) == 3: 
            mask_np = mask_np[0]
        
        H, W = mask_np.shape
        y_idx, x_idx = np.where(mask_np > 0.5)
        N = len(y_idx)
        
        # Mapping: (y, x) -> row_index (0 to N-1)
        coord_to_id = np.full((H, W), -1, dtype=np.int32)
        coord_to_id[y_idx, x_idx] = np.arange(N)

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        rows = []
        cols = []
        data = []
        
        # Diagonal: 4
        rows.append(np.arange(N))
        cols.append(np.arange(N))
        data.append(np.full(N, 4, dtype=np.float32))
        
        # Store boundary info for later: (is_boundary_mask, neighbor_y, neighbor_x)
        boundary_terms = []

        for dy, dx in neighbors:
            ny, nx = y_idx + dy, x_idx + dx
            
            # 1. Filter for valid spatial bounds
            valid_mask = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            
            # 2. Determine if valid neighbors are inside the mask
            inside_mask = np.zeros(N, dtype=bool)
            # SAFELY access coord_to_id using only valid indices
            inside_mask[valid_mask] = (coord_to_id[ny[valid_mask], nx[valid_mask]] != -1)

            # --- Fill Matrix A (Internal Neighbors) ---
            # Current pixel index (row) -> Neighbor pixel index (col)
            # Only where neighbor is ALSO in mask
            curr_nodes = coord_to_id[y_idx[inside_mask], x_idx[inside_mask]]
            neigh_nodes = coord_to_id[ny[inside_mask], nx[inside_mask]]
            
            rows.append(curr_nodes)
            cols.append(neigh_nodes)
            data.append(np.full(len(curr_nodes), -1, dtype=np.float32))
            
            # --- Prepare Boundary Terms (External Neighbors) ---
            # Neighbor is valid spatially, but NOT in mask
            is_boundary = valid_mask & (~inside_mask)
            boundary_terms.append((is_boundary, ny, nx))

        # Construct Sparse Matrix
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        A = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, N))

        # Solve per channel
        result_np = target_np.copy()
        
        for c in range(source.shape[0]):
            s_chan = source_np[c]
            t_chan = target_np[c]
            
            # Laplacian of Source (4*center - neighbors)
            laplacian_s = 4 * s_chan[y_idx, x_idx]
            for dy, dx in neighbors:
                ny, nx = y_idx + dy, x_idx + dx
                # Clamp for safety when computing source gradient
                ny_c, nx_c = np.clip(ny, 0, H-1), np.clip(nx, 0, W-1)
                laplacian_s -= s_chan[ny_c, nx_c]
                
            b = laplacian_s

            # Add Boundary Conditions (Target image values)
            for is_boundary, ny, nx in boundary_terms:
                # For pixels that have a boundary neighbor, add that neighbor's target intensity to b
                # Filter ny/nx by is_boundary to get only the relevant boundary pixels
                b[is_boundary] += t_chan[ny[is_boundary], nx[is_boundary]]

            x = spsolve(A, b)
            x = np.clip(x, 0, 1)
            result_np[c, y_idx, x_idx] = x

        return torch.from_numpy(result_np).to(device)
    
    if len(source.shape) < 4:
        return poisson_blend_single_frame(source, target, mask)
    else:
        return torch.stack([poisson_blend_single_frame(s,t,m) for s,t,m in zip(source, target, mask)])

def recompose_tiles(tiles: torch.Tensor, n_tiles=(2, 2), overlap=16):
    """Given image tiles, recompose into the original images.

    Args:
        tiles (torch.Tensor [B, NH*NW, C, TH, TW]): _description_
        n_tiles (tuple, optional): _description_. Defaults to (2, 2).
        overlap (int, optional): _description_. Defaults to 20.

    Returns:
        recomposed (Tensor [B, C, H, W]): 
    """
    B, _, C, TH, TW = tiles.shape
    NH, NW = n_tiles
    H, W = NH * (TH - overlap // 2), NW * (TW - overlap // 2)
    SH, SW = TH - overlap, TW - overlap

    tiles = tiles.transpose(0,1).reshape(NH, NW, B, C, TH, TW)
    w = torch.linspace(0, 1, overlap, device=tiles.device)
    tiles[:-1, ..., -overlap:, :] = torch.einsum("hwbcoW,o->hwbcoW", tiles[:-1, ..., -overlap:, :], w.flip(0))
    tiles[1:, ..., :overlap, :] = torch.einsum("hwbcoW,o->hwbcoW", tiles[1:, ..., :overlap, :], w)
    tiles[:, :-1, ..., -overlap:] = torch.einsum("hwbcHo,o->hwbcHo", tiles[:, :-1, ..., -overlap:], w.flip(0))
    tiles[:, 1:, ..., :overlap] = torch.einsum("hwbcHo,o->hwbcHo", tiles[:, 1:, ..., :overlap], w)
    tiles = tiles.permute(2, 3, 4, 5, 0, 1).reshape(B, C * TH * TW, NH * NW)
    recomposed = F.fold(tiles, (H, W), (TH, TW), stride=(SH, SW))
    return recomposed


def create_tiles(images, n_tiles=(2, 2), overlap=16):
    """Spatially divide images in to tiles

    Args:
        images (Tensor [B, C, H, W]): Input image tensor
        n_tiles (tuple [n_height, n_width], optional): Number of tiles per side. Defaults to (2, 2).
        overlap (int, optional): Number of pixels overlap between tiles. Defaults to 20.

    Returns:
        tiles (Tensor, [B, N_tiles_per_image, C, tile_height, tile_width]):
    """
    B, C, H, W = images.shape
    NH, NW = n_tiles
    TH, TW = (H // NH) + overlap // 2, (W // NW) + overlap // 2
    SH, SW = TH - overlap, TW - overlap

    tiles = F.unfold(images, (TH, TW), stride=(SH, SW))
    tiles = tiles.reshape(B, C, TH, TW, -1).permute(0, -1, 1, 2, 3)  # B, NH*NW, C, TH, TW
    return tiles