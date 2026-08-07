import os
from typing import Tuple, List
import torch
import numpy as np
import cv2
from PIL import Image

INTERPOLATION_MAP_CV2 = {
    "NEAREST":  cv2.INTER_NEAREST,
    "BILINEAR": cv2.INTER_LINEAR,
    "BICUBIC":  cv2.INTER_CUBIC,
    "LANCZOS":  cv2.INTER_LANCZOS4,
}

INTERPOLATION_MAP_PIL = {
    # "NEAREST":  Image.NEAREST,
    "NEAREST":  Image.Resampling.NEAREST,
    # "BILINEAR": Image.BILINEAR,
    "BILINEAR":  Image.Resampling.BILINEAR,
    # "BICUBIC":  Image.BICUBIC,
    "BICUBIC":  Image.Resampling.BICUBIC,
    # "LANCZOS":  Image.LANCZOS, 
    "LANCZOS":  Image.Resampling.LANCZOS,
}

def normalize_rgb_images(arr_rgb_clips):
    arr_rgb_clips_norm = np.clip((arr_rgb_clips / 127.5 - 1.0), -1.0, 1.0)
    return arr_rgb_clips_norm

MAX_ELEMENTS_FOR_QUANTILE = 5_000_000 
def _quantile_helper(tensor: torch.Tensor, p_val: float) -> torch.Tensor:
    """
    Calculates the quantile of a tensor, with a FAST fallback to subsampling for very large tensors.
    """
    numel = tensor.numel()
    
    # Check if the tensor is too large
    if numel > MAX_ELEMENTS_FOR_QUANTILE:
        # --- FASTER SAMPLING LOGIC ---
        # Calculate the stride to get approximately the desired number of samples
        stride = numel // MAX_ELEMENTS_FOR_QUANTILE
        
        # Pick a random starting offset
        offset = torch.randint(0, stride, (1,)).item()
        
        # Subsample using a strided slice
        tensor_subset = tensor[offset::stride]
        
        return torch.quantile(tensor_subset.float(), p_val / 100.0, interpolation='linear')
    else:
        # If the tensor is small enough, calculate quantile directly
        return torch.quantile(tensor.float(), p_val / 100.0, interpolation='linear')

def _interp_manual(x, xp, fp):
    """
    A manual implementation of 1D linear interpolation for PyTorch,
    similar to numpy.interp. Works for older PyTorch versions.
    x: The input tensor to be interpolated.
    xp: The 1D tensor of data points (x-coordinates). Must be sorted.
    fp: The 1D tensor of data values (y-coordinates).
    """
    # Find the indices of the intervals that contain the x values
    # 'right' means that if x[i] == xp[j], the index j is returned.
    indices = torch.searchsorted(xp, x, right=True)

    # Handle out-of-bounds cases by clamping indices
    # Values < xp[0] will have index 0, map to fp[0]
    # Values > xp[-1] will have index len(xp), map to fp[-1]
    
    # Get the left and right bounds for interpolation
    left_indices = (indices - 1).clamp(min=0)
    right_indices = indices.clamp(max=len(xp) - 1)

    # Get the corresponding x and y values from the lookup table
    xp_left = xp[left_indices]
    xp_right = xp[right_indices]
    fp_left = fp[left_indices]
    fp_right = fp[right_indices]

    # Calculate the interpolation weight
    # Avoid division by zero if xp_left == xp_right
    denom = xp_right - xp_left
    weight = (x - xp_left) / torch.where(denom == 0, torch.ones_like(denom), denom)
    
    # Perform the linear interpolation
    return fp_left + weight * (fp_right - fp_left)

def _cubic_bezier(t: torch.Tensor, p0: tuple, p1: tuple, p2: tuple, p3: tuple) -> torch.Tensor:
    """
    Calculates a point on a cubic Bézier curve.

    Args:
        t: A tensor of parameter values in the range [0, 1].
        p0, p1, p2, p3: The four control points of the Bézier curve.

    Returns:
        A tensor of (x, y) coordinates on the curve.
    """
    # Reshape t from [N] to [N, 1] to enable broadcasting with control points.
    t = t.unsqueeze(-1)

    p0 = torch.tensor(p0, dtype=t.dtype, device=t.device)
    p1 = torch.tensor(p1, dtype=t.dtype, device=t.device)
    p2 = torch.tensor(p2, dtype=t.dtype, device=t.device)
    p3 = torch.tensor(p3, dtype=t.dtype, device=t.device)
    
    u = 1.0 - t
    u_squared = u * u
    u_cubed = u_squared * u
    t_squared = t * t
    t_cubed = t_squared * t
    
    return u_cubed * p0 + 3.0 * u_squared * t * p1 + 3.0 * u * t_squared * p2 + t_cubed * p3

# Global variable to cache the pre-computed Look-Up Tables (LUTs).
# This avoids regenerating the curves on every function call.
_FUSION_LUTS = {}

# Define the control points for the different curves
_CURVE_DEFINITIONS = {
    "high": {
        0.0: {'y': 0.0, 'rh': (0.664884135472371, 0.0518518518518519)},
        1.0: {'y': 1.0, 'lh': (0.923351158645276, 0.355555555555556)}
    },
    "medium": {
        0.0: {'y': 0.0, 'rh': (0.643831503893424, 0.128774928774929)},
        1.0: {'y': 1.0, 'lh': (0.867210807768083, 0.365170940170941)}
    },
    "low": {
        0.0: {'y': 0.0, 'rh': (0.577164837226757, 0.152813390313391)},
        1.0: {'y': 1.0, 'lh': (0.839140632329486, 0.437286324786326)}
    }
}

def _generate_fusion_lut(curve_name: str, samples: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the color curve LUT from pre-defined Bézier points for a given curve name.
    The LUT is intentionally created on the CPU to be device-agnostic.
    """
    if curve_name not in _CURVE_DEFINITIONS:
        raise ValueError(f"Unknown curve name: {curve_name}. Available options are: {list(_CURVE_DEFINITIONS.keys())}")

    curve_points = _CURVE_DEFINITIONS[curve_name]
    
    sorted_points = sorted(curve_points.items())
    
    lut_x = []
    lut_y = []
    
    # Generate tensors on the CPU. They will be moved to the correct device later.
    t_vals = torch.linspace(0, 1, samples // (len(sorted_points) - 1), device='cpu')

    for i in range(len(sorted_points) - 1):
        x0, p0_data = sorted_points[i]
        x1, p1_data = sorted_points[i+1]
        
        p0 = (x0, p0_data['y'])
        p3 = (x1, p1_data['y'])
        p1 = p0_data['rh']
        p2 = p1_data['lh']
        
        segment_points = _cubic_bezier(t_vals, p0, p1, p2, p3)
        
        lut_x.append(segment_points[:-1, 0])
        lut_y.append(segment_points[:-1, 1])

    p3 = (sorted_points[-1][0], sorted_points[-1][1]['y'])
    lut_x.append(torch.tensor([p3[0]], device='cpu'))
    lut_y.append(torch.tensor([p3[1]], device='cpu'))

    final_lut_x = torch.cat(lut_x)
    final_lut_y = torch.cat(lut_y)

    return final_lut_x, final_lut_y

def apply_fusion_color_curve(
    depth_tensor: torch.Tensor,
    curve_name: str = "high",
    samples: int = 1024
) -> torch.Tensor:
    """
    Applies a pre-computed color curve from Fusion Studio to a depth tensor.

    The function remaps the input from [-1, 1] to [0, 1], applies the curve
    using a cached LUT, and returns the result in the [-1, 1] range.
    """
    global _FUSION_LUTS

    # Lazy initialization: Generate the LUT only on the first call for this curve.
    if curve_name not in _FUSION_LUTS:
        # This is not thread-safe, but usually fine for typical PyTorch data loaders.
        _FUSION_LUTS[curve_name] = _generate_fusion_lut(curve_name, samples)

    # Move the cached LUT to the same device as the input tensor.
    # This is a fast, non-blocking operation if it's already on the correct device.
    lut_x, lut_y = _FUSION_LUTS[curve_name]
    device = depth_tensor.device
    lut_x = lut_x.to(device)
    lut_y = lut_y.to(device)

    # Remap input tensor from [-1, 1] to [0, 1] to match the curve's domain
    depth_remapped = (depth_tensor.float() + 1.0) / 2.0
    
    # Apply the LUT using interpolation
    curved_depth = _interp_manual(depth_remapped, lut_x, lut_y)

    # Remap the output from [0, 1] back to [-1, 1]
    return curved_depth * 2.0 - 1.0

def normalize_depth_percentile(depth: torch.Tensor,
                               p_lo: float = 0.0,               # Percentile for the lower bound of normalization
                               p_hi: float = 98.0,              # Percentile for the upper bound of normalization
                               *,
                               near_plane: float = 0.0,
                               far_plane: float = 655.0,
                               near_value: float = -1.0,        # Value for near-plane pixels
                               far_value: float = 1.0,          # Value for far-plane pixels
                               k_compression: float = 3.0,      # Log compression factor
                               eps: float = 1e-6,               # Small epsilon for float comparisons and denominator
                               flip_output_sign: bool = False,  # Controls the final depth_n = -depth_n
                               apply_curve: bool = False):

    """
    Normalise a depth map to [-1, 1] clip-wise, with improved foreground handling.

    To avoid "foreground compression" (where distinct close values map to the
    same minimum normalized value because p_lo is too high), consider passing
    a very small `p_lo` (e.g., 0.0 or 0.1). If p_lo=0, d_lo becomes the
    minimum valid log-transformed depth, ensuring no valid depth maps below -1
    (before potential sign flip).

    Normalization Steps:
    1. Pixels with depth <= near_plane or depth >= far_plane (original units) are
       conditionally excluded from percentile statistics using a small epsilon buffer.
       Specifically, pixels with depth in (near_plane + eps, far_plane - eps) are
       considered "valid" for statistics.
    2. "Valid" pixels are log-transformed: log(1 + k_compression * depth).
    3. Percentiles (p_lo, p_hi) are computed on these valid, transformed depths (d_lo, d_hi).
    4. All log-transformed depths are linearly mapped such that d_lo -> -1 and d_hi -> +1.
       Special handling for cases where d_lo is very close to d_hi.
    5. Values outside [-1, 1] after this mapping ARE CLAMPED to [-1, 1].
    6. Near-plane pixels (depth <= near_plane + eps) are set to `near_value`.
       Far-plane pixels (depth >= far_plane - eps) are set to `far_value`.
    7. If `flip_output_sign` is True, the entire output is negated (depth_n = -depth_n).

    Returns
    -------
    depth_n  tensor with the same shape as `depth`, normalized float32 values.
    (d_lo, d_hi) the p_lo / p_hi percentiles of the *valid log-transformed pixels*.
                   These are 0-dim float32 tensors in log-space.
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError("Input depth must be a torch.Tensor.")
    if depth.ndim < 2:
        raise ValueError("Input depth should at least be 2D (H, W).")

    original_depth_values = depth # Keep reference to original values for masking

    # 1. Build masks on ORIGINAL depth values
    near_plane_thresh = near_plane + eps if near_plane > -float('inf') else -float('inf')
    far_plane_thresh = far_plane - eps if far_plane < float('inf') else float('inf')

    assign_near_mask = original_depth_values <= near_plane_thresh
    assign_far_mask  = original_depth_values >= far_plane_thresh

    valid_stats_mask = ~(assign_near_mask | assign_far_mask)

    # # 2. Power-Law transform all depth values
    # # k_compression is the 'k' in d**(1/k). It must be > 0.
    # # A k > 1 expands the foreground. k=1 is linear. 0 < k < 1 compresses the foreground.
    # if k_compression <= 0:
    #     # Or raise an error, but returning untransformed might be safer for the script
    #     depth_transformed = original_depth_values.float()
    # else:
    #     depth_transformed = original_depth_values.float().clamp_min(0) ** (1.0 / k_compression)

    # 2. Power-Law transform using log(k) for better sensitivity at high k.
    # Formula is d ** (1 / log(k)). k must be > 1.
    if k_compression <= 1:
        # k=1 would cause division by zero (log(1)=0). k<1 is not meaningful here.
        # Return untransformed as a safe fallback.
        depth_transformed = original_depth_values.float()
    else:
        exponent = 1.0 / torch.log(torch.tensor(k_compression, device=original_depth_values.device))
        depth_transformed = original_depth_values.float().clamp_min(0) ** exponent

    valid_log_depths = depth_transformed[valid_stats_mask]

    # 2. Log transform all depth values
    # Ensure input to log1p is non-negative if k_compression * depth could be < -1.
    # Assuming depth values are typically >= 0.
    # Convert depth to float for log transform if it isn't already.
    # depth_log = torch.log1p(k_compression * original_depth_values.float().clamp_min(0.0))
    # valid_log_depths = depth_log[valid_stats_mask]


    # Handle degenerate case: no valid pixels for statistics
    if valid_log_depths.numel() == 0:
        depth_n = torch.full_like(original_depth_values, 0.0, dtype=torch.float32)
        depth_n[assign_near_mask] = near_value
        depth_n[assign_far_mask]  = far_value
        if flip_output_sign:
            depth_n = -depth_n
        
        nan_val = torch.tensor(float("nan"), dtype=torch.float32, device=depth_n.device)
        return depth_n, (nan_val, nan_val)

    # 3. Robust percentiles on the valid log-transformed subset
    d_lo = _quantile_helper(valid_log_depths, p_lo)
    d_hi = _quantile_helper(valid_log_depths, p_hi)

    if torch.isnan(d_lo) or torch.isnan(d_hi): # Should be caught by numel=0 or if valid_log_depths has NaNs
        depth_n = torch.full_like(original_depth_values, 0.0, dtype=torch.float32)
        depth_n[assign_near_mask] = near_value
        depth_n[assign_far_mask]  = far_value
        if flip_output_sign:
            depth_n = -depth_n
        return depth_n, (d_lo, d_hi) # Return NaNs for d_lo/d_hi

    # 4. Linear map: d_lo -> -1, d_hi -> +1
    denominator = d_hi - d_lo

    if torch.abs(denominator).item() < eps:
        # d_lo is very close to d_hi. Valid log depths have negligible dynamic range.
        # Map values at or near d_lo to -1.0.
        # Map values noticeably greater than d_lo (and thus d_hi) to 1.0.
        depth_n_scaled = torch.full_like(depth_transformed, -1.0)
        depth_n_scaled[depth_transformed > d_lo + eps] = 1.0

    else:
        depth_n_scaled = ((depth_transformed - d_lo) / denominator - 0.5) * 2.0

    # 5. Clip/Clamp values to [-1, 1]
    # This ensures that pixels whose log_depths were outside [d_lo, d_hi]
    # (or affected by the small denominator case) are brought into the [-1, 1] range.
    # depth_n_clamped = depth_n_scaled.clamp(-1.0, 1.0)

    # Create the output tensor, ensuring it's float32
    # depth_n_clamped is already based on depth_log (float), so this should be fine.
    depth_n = depth_n_scaled.clone() 

    # 6. Overwrite the masked regions with user-specified constants
    depth_n[assign_near_mask] = near_value
    depth_n[assign_far_mask]  = far_value


    # 7. Optional final flip
    if flip_output_sign:
        depth_n = -depth_n
        # we can only apply the curve (if needed) after inverting the depth values.
        if apply_curve:
            depth_n = apply_fusion_color_curve(depth_n)


    return depth_n, (d_lo.float(), d_hi.float()) # Ensure returned d_lo/d_hi are float


def denormalize_depth_percentile(depth_n: torch.Tensor, d2: torch.Tensor, d98: torch.Tensor):
    return ((depth_n / 2.0) + 0.5) * (d98 - d2) + d2

def normalize_depth_logarithmic(depth: torch.Tensor, # 1, 16, 320, 512
                                k: float = 1.0,
                                far_plane: float = 1000.0,) -> torch.Tensor:
    """
    Log-space depth normalisation (near plane = 0 → 0.0, ref_far_distance → 1.0).

    Args:
        depth (torch.Tensor): Depth map [C, F, H, W], assumed ≥ 0.
        k (float): Compression factor (> 0).
        ref_far_distance (float): Distance that should normalise to 1.0 (> 0).

    Returns:
        torch.Tensor: Normalised depth, same shape as input.
    """
    if far_plane <= 0:
        raise ValueError("ref_far_distance must be positive.")
    if k <= 0:
        raise ValueError("k (compression factor) must be positive.")

    # Ensure non-negative depths.
    depth_clamped = depth.clamp_min(0.0)

    # Numerator: log(1 + k * d)  (use log1p for better precision)
    numerator = torch.log1p(k * depth_clamped)

    # Denominator: log(1 + k * F_ref)
    denominator = torch.log1p(torch.tensor(k * far_plane, dtype=depth.dtype, device=depth.device))

    # Prevent division by zero (shouldn’t happen unless far_plane == 0).
    if torch.isclose(denominator, torch.tensor(0., dtype=denominator.dtype, device=denominator.device)):
        # Depths ≤ 0 → 0, otherwise → 1 
        return torch.where(depth_clamped <= 0, torch.zeros_like(depth), torch.ones_like(depth))

    depth_n = (numerator / denominator - 0.5) * 2.0

    return depth_n

# ──────────────────────────────────────────────────────────────────────────── #