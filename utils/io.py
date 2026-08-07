import os
from typing import Tuple, List
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple, Union

import numpy as np
from natsort import natsorted
import cv2
from PIL import Image

import OpenEXR, Imath

from utils.utils import EPS
from utils.data import INTERPOLATION_MAP_CV2, INTERPOLATION_MAP_PIL, normalize_rgb_images
# Import openexr_numpy for EXR file handling
try:
    from openexr_numpy import imwrite as exr_imwrite
    EXR_AVAILABLE = True
except ImportError:
    EXR_AVAILABLE = False

DEFAULT_IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr")


# =============================================================================
#                  Reusable Path-pattern Module (Begin)
# =============================================================================

class PathPatternError(ValueError):
    """Raised when input/output path patterns are malformed or incompatible."""

@dataclass(frozen=True)
class PathMatch:
    """One matched input directory and the concrete names for each pattern segment."""
    input_dir: Path
    segments: Tuple[str, ...]


def _split_segments(spec: str) -> List[str]:
    """Split a relative path spec into segments; supports '/' or '\\' as separators."""
    s = spec.strip().strip("/\\")
    if not s:
        return []
    return re.split(r"[\\/]+", s)

def _validate_specs(in_segs: Sequence[str], out_segs: Sequence[str]) -> None:
    if len(in_segs) != len(out_segs):
        raise PathPatternError(
            f"Segment count mismatch: inputs={len(in_segs)} vs outputs={len(out_segs)}"
        )
    for i, (ins, outs) in enumerate(zip(in_segs, out_segs)):
        if ins == "*" and outs != "^":
            raise PathPatternError(
                f"At segment {i}: input is '*' so output must be '^' (got '{outs}')."
            )

class RelPathMapper:
    """
    Expand wildcard directory patterns under a root and map matches to output dirs.
    """

    def __init__(
        self,
        inputs_root: Union[str, Path],
        inputs_rel_path: str,
        outputs_root: Union[str, Path],
        outputs_rel_path: str,
    ) -> None:
        self.inputs_root = Path(inputs_root)
        self.outputs_root = Path(outputs_root)
        self._in_segs: Tuple[str, ...] = tuple(_split_segments(inputs_rel_path))
        self._out_segs: Tuple[str, ...] = tuple(_split_segments(outputs_rel_path))
        _validate_specs(self._in_segs, self._out_segs)

    def iter_matches(self) -> Iterator[PathMatch]:
        """Yield PathMatch entries for all directories that satisfy inputs_rel_path."""
        def walk(level: int, cur: Path, seg_names: List[str]) -> Iterator[PathMatch]:
            if level == len(self._in_segs):
                yield PathMatch(cur, tuple(seg_names))
                return
            seg = self._in_segs[level]
            if seg == "*":
                if not cur.exists():
                    return
                for child in sorted((p for p in cur.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
                    yield from walk(level + 1, child, seg_names + [child.name])
            else:
                child = cur / seg
                if child.is_dir():
                    yield from walk(level + 1, child, seg_names + [seg])

        yield from walk(0, self.inputs_root, [])

    def output_dir_for(self, match: PathMatch) -> Path:
        """Map one input match to the corresponding output directory."""
        if len(match.segments) != len(self._in_segs):
            raise PathPatternError("Internal error: segments length mismatch.")
        parts: List[str] = []
        for i, outseg in enumerate(self._out_segs):
            parts.append(match.segments[i] if outseg == "^" else outseg)
        return self.outputs_root.joinpath(*parts)

def list_images(dir_path: Path, exts: Sequence[str] = DEFAULT_IMAGE_EXTS) -> List[Path]:
    """
    List image files in a directory, naturally sorted by stem using `natsort`.
    (Leaves the UNTOUCHED module as-is; this function is used by our batch code.)
    """
    norm_exts = tuple(e if e.startswith('.') else f'.{e}' for e in (x.lower() for x in exts))
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in norm_exts]
    # Sort naturally by filename stem (e.g., 2 < 10)
    return natsorted(files, key=lambda p: p.stem)

def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist (parents included)."""
    p.mkdir(parents=True, exist_ok=True)
    
# =============================================================================
#                  Reusable Path-pattern Module (End)
# =============================================================================

def load_images_sequence(inputs_folder: str, 
                         resize: tuple[int, int]=None,
                         interpolation: str = "BILINEAR",
                         start_frame: int = None,
                         end_frame: int = None) -> np.ndarray:
    """
    Load all RGB images in a folder, resize, normalize, and return a
    NumPy array of shape [T, C, H, W] (T = number of frames).
    """
    
    all_paths = natsorted([p for p in inputs_folder.iterdir()
                           if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    if not all_paths:
        raise ValueError(f"No images found in {inputs_folder}")

    paths_to_process = []
    # If start/end frames are provided, filter the list
    if start_frame is not None and end_frame is not None:
        for p in all_paths:
            try:
                frame_str = p.name.split('.')[-2]
                frame_num = int(frame_str)
                
                if start_frame <= frame_num <= end_frame:
                    paths_to_process.append(p)
            except (ValueError, IndexError):
                # This catches files that don't match the naming convention
                # e.g. "reference_image.png" (no frame number)
                continue
                
        if not paths_to_process:
            raise ValueError(f"No frames found in range {start_frame}-{end_frame} in {inputs_folder}")
            
    else:
        # If no flags provided, load the full sequence
        paths_to_process = all_paths

    frames = []
    for p in paths_to_process:
        img = Image.open(p).convert("RGB")
        if(resize):
            h_resize, w_resize = resize
            img = img.resize((w_resize, h_resize), resample=INTERPOLATION_MAP_PIL[interpolation])
        arr = np.asarray(img, dtype=np.float32)
        arr = normalize_rgb_images(arr)       # [-1, 1]
        frames.append(arr.transpose(2, 0, 1))  # [C, H, W]

    return np.stack(frames, axis=0), paths_to_process            # [T, C, H, W]

def read_exr_with_openexr(path: Path):
    """Read EXR using OpenEXR/Imath into float32 RGB [H,W,3]."""
    try:
        import OpenEXR, Imath
        exr = OpenEXR.InputFile(str(path))
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chs = header.get("channels", {})

        def _chan(name):
            if name in chs:
                return np.frombuffer(exr.channel(name, pt), dtype=np.float32).reshape(height, width)
            return None

        r = _chan("R");  r = _chan("r") if r is None else r
        r = _chan("Y") if r is None else r;  r = _chan("y") if r is None else r

        g = _chan("G");  g = _chan("g") if g is None else g;  g = r if g is None else g
        b = _chan("B");  b = _chan("b") if b is None else b;  b = r if b is None else b

        if r is None:
            names = list(chs.keys())
            planes = []
            for i in range(min(3, len(names))):
                planes.append(np.frombuffer(exr.channel(names[i], pt), dtype=np.float32).reshape(height, width))
            if not planes:
                exr.close()
                return None
            while len(planes) < 3:
                planes.append(planes[-1])
            rgb = np.stack(planes[:3], axis=-1)
        else:
            rgb = np.stack([r, g, b], axis=-1)

        exr.close()
        return rgb.astype(np.float32, copy=False)
    except Exception:
        return None
    
def read_image_any(path: Path) -> np.ndarray:
    """
    Returns RGB float32 image in [0,1], shape (H, W, 3).
    Supports standard formats and EXR without requiring OpenCV EXR codec.
    """
    suffix = path.suffix.lower()
    if suffix == ".exr":
        img = read_exr_with_openexr(path)
        if img is None:
            raise RuntimeError(
                f"Could not read EXR file: {path}\n"
                f"Please install either 'imageio' (pip install imageio) or 'OpenEXR Imath'."
            )
        img = np.where(np.isfinite(img), img, 0.0).astype(np.float32, copy=False)
        img = np.clip(img, 0.0, 1.0)
        return img
    else:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr

def load_images_sequence_mixed(input_paths, resize, interpolation_name):
    """
    Load a sequence of images (supports EXR + standard images).
    Returns numpy array [T, C, H, W], float32 in [0,1].
    """
    
    H, W = resize
    inter = INTERPOLATION_MAP_CV2[interpolation_name] or cv2.INTER_LINEAR

    frames = []
    for p in input_paths:
        img = read_image_any(p)  # H, W, 3 in [0,1], float32
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=inter)
        frames.append(np.transpose(img, (2, 0, 1)))  # to C, H, W

    return np.stack(frames, axis=0).astype(np.float32, copy=False)

def save_images_sequence(images_sequence: np.ndarray,
                         outputs_folder: str,
                         prefix: str,
                         starting_index: int,
                         num_digits: int,
                         format: str = "png",
                         resize: tuple[int, int]=None) -> None:
    """
    Save a sequence of images of shape [T, C, H, W] to `outputs_folder`.
    Filenames follow `<prefix><index>.<format>`, where `index` starts at
    `starting_index` and is padded with zeros to `num_digits` length.
    """
    os.makedirs(outputs_folder, exist_ok=True)

    for idx, frame in enumerate(images_sequence):
        img = frame.transpose(1, 2, 0)  # [H, W, C]
        # img = ((img + 1.0) / 2.0).clip(0, 1)   # back to [0, 1]
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        if(resize):
            h_resize, w_resize = resize
            img = img.resize((w_resize, h_resize), Image.BILINEAR)
        filename = f"{prefix}{starting_index + idx:0{num_digits}d}.{format}"
        pil_img.save(outputs_folder / filename)

def save_images_sequence_high_precision(images_sequence: np.ndarray,
                                        outputs_folder: str,
                                        input_filenames: List[Path] = None,
                                        prefix: str = "image_",
                                        starting_index: int = 0,
                                        num_digits: int = 5,
                                        format: str = "png",
                                        resize: tuple[int, int] = None,
                                        interpolation: str = "BILINEAR",
                                        apply_normalization: bool = True,
                                        save_metadata: bool = True) -> None:
    """
    Save a sequence of images with high precision (16-bit PNG or 32-bit float EXR) to preserve raw depth values.
    
    Args:
        images_sequence: [T, C, H, W] array of images
        outputs_folder: Output directory
        input_filenames: List of original input file paths (if provided, uses original names with "_depth" suffix)
        prefix: Filename prefix (used only if input_filenames is None)
        starting_index: Starting index for filenames (used only if input_filenames is None)
        num_digits: Number of digits for zero-padding (used only if input_filenames is None)
        format: Image format ('png' for 16-bit, 'exr' for 32-bit float).
        resize: Optional resize tuple (H, W) to match input image size
        apply_normalization: Whether to apply global normalization across all frames
        save_metadata: Whether to save metadata files with scaling information
    """
    os.makedirs(outputs_folder, exist_ok=True)

    if apply_normalization:
        # Global normalization across all frames to prevent flickering
        global_min = images_sequence.min()
        global_max = images_sequence.max()
        
        print(f"Global value range: [{global_min:.6f}, {global_max:.6f}]")
        
        if global_max > global_min:
            # Scale entire sequence to utilize full 16-bit range while preserving relative values
            images_scaled = (images_sequence - global_min) / (global_max - global_min)
        else:
            # Handle constant images
            images_scaled = images_sequence
    else:
        # Use raw values without global normalization
        images_scaled = images_sequence
        global_min = images_sequence.min()
        global_max = images_sequence.max()

    for idx, frame in enumerate(images_scaled):
        img = frame.transpose(1, 2, 0)  # [H, W, C]

       # Generate filename
        if prefix:
            # If a custom prefix is provided, use it but keep the original frame number
            if input_filenames and idx < len(input_filenames):
                original_path = input_filenames[idx]
                original_name = original_path.name
                
                # Try to extract the frame number from the original filename
                match = re.search(r'(\d+)(?=\.\w+$)', original_name)
                if match:
                    frame_number_str = match.group(1)
                    # Use the same number of digits as the original
                    num_digits_original = len(frame_number_str)
                    frame_number = int(frame_number_str)
                    filename = f"{prefix}.{frame_number:0{num_digits_original}d}.{format}"
                else:
                    # Fallback if no frame number is found, use index
                    filename = f"{prefix}.{starting_index + idx:0{num_digits}d}.{format}"
            else:
                # Fallback for generic naming if no input files are available
                filename = f"{prefix}.{starting_index + idx:0{num_digits}d}.{format}"
        
        elif input_filenames and idx < len(input_filenames):
            # Parse original filename to handle patterns like "name.00000.png" → "name_depth.00000.png"
            original_path = input_filenames[idx]
            original_name = original_path.name
            
            # Check if filename has frame number pattern (digits before extension)
            # Pattern: name.digits.extension or name_digits.extension
            match = re.match(r'^(.+?)([._])(\d+)(\.\w+)$', original_name)
            if match:
                base_name = match.group(1)
                separator = match.group(2)
                frame_number = match.group(3)
                extension = match.group(4)
                filename = f"{base_name}_depth{separator}{frame_number}.{format}"
            else:
                # Fallback: just add _depth before extension
                stem = original_path.stem
                filename = f"{stem}_depth.{format}"
        else:
            # Fallback to generic naming
            filename = f"{prefix}{starting_index + idx:0{num_digits}d}.{format}"
        
        filepath = os.path.join(outputs_folder, filename)

        if format == 'png':
            # Handle single channel vs multi-channel
            if img.shape[2] == 1:
                img_single = img.squeeze(-1)
            else:
                # For depth maps, typically we only use the first channel
                img_single = img[:, :, 0]
            
            if resize:
                h_resize, w_resize = resize
                
                # Use BILINEAR interpolation for smooth resizing
                # Work with float32 then convert to 16-bit to avoid PIL mode compatibility issues
                img_float = img_single.astype(np.float32)
                pil_img_float = Image.fromarray(img_float, mode='F')
                
                # Resize in float mode using BILINEAR interpolation
                pil_img_float_resized = pil_img_float.resize((w_resize, h_resize), resample=INTERPOLATION_MAP_PIL[interpolation])
                
                # Convert back to numpy, then to 16-bit
                img_resized = np.array(pil_img_float_resized)
                img_16bit = (img_resized * 65535).astype(np.uint16)
                pil_img = Image.fromarray(img_16bit, mode='I;16')
            else:
                # No resize needed, convert directly to 16-bit
                img_16bit = (img_single * 65535).astype(np.uint16)
                pil_img = Image.fromarray(img_16bit, mode='I;16')
            
            pil_img.save(filepath)

        elif format == 'exr':
            # Use 32-bit float for EXR
            if not EXR_AVAILABLE:
                raise ImportError("openexr-numpy package is required for EXR support. Install with: pip install openexr-numpy")

            # We import scipy here to handle the large kernel median on float32 data
            try:
                from scipy.ndimage import median_filter
            except ImportError:
                raise ImportError("SciPy is required for median filters > 5 on EXR files. Install with: pip install scipy")
            
            if img.shape[2] == 1:
                img_32bit = img.squeeze(-1).astype(np.float32)
            else:
                img_32bit = img[:, :, 0].astype(np.float32)

            # 1. True Size 7 Median Filter (via SciPy)
            img_32bit = median_filter(img_32bit, size=5)

            if resize:
                h_resize, w_resize = resize
                # Use INTER_NEAREST to preserve depth values during resizing
                img_32bit = cv2.resize(img_32bit, (w_resize, h_resize), interpolation=INTERPOLATION_MAP_CV2[interpolation])

            # Save as single channel EXR using openexr_numpy, now saving directly to the Red channel.
            exr_imwrite(filepath, img_32bit, channel_names=['R'])
        
        else:
            raise ValueError(f"Unsupported format for high-precision saving: {format}. Use 'png' or 'exr'.")
        
        # Save metadata only if requested
        if save_metadata:
            if input_filenames and idx < len(input_filenames):
                # Use same naming pattern for metadata
                original_path = input_filenames[idx]
                original_name = original_path.name
                
                match = re.match(r'^(.+?)([._])(\d+)(\.\w+)$', original_name)
                if match:
                    base_name = match.group(1)
                    separator = match.group(2)
                    frame_number = match.group(3)
                    metadata_filename = f"{base_name}_depth{separator}{frame_number}_metadata.txt"
                else:
                    metadata_filename = f"{original_path.stem}_depth_metadata.txt"
            else:
                metadata_filename = f"{prefix}{starting_index + idx:0{num_digits}d}_metadata.txt"
                
            metadata_filepath = os.path.join(outputs_folder, metadata_filename)
            with open(metadata_filepath, 'w') as f:
                if apply_normalization:
                    f.write(f"global_min_value: {global_min}\n")
                    f.write(f"global_max_value: {global_max}\n")
                    f.write(f"global_range: [{global_min}, {global_max}]\n")
                    f.write(f"scaling_applied: (value - {global_min}) / ({global_max} - {global_min})\n")
                    f.write(f"reconstruction: scaled_value * ({global_max} - {global_min}) + {global_min}\n")
                    f.write(f"normalization: GLOBAL across all frames\n")
                else:
                    f.write(f"min_value: {frame.min()}\n")
                    f.write(f"max_value: {frame.max()}\n")
                    f.write(f"value_range: [{frame.min()}, {frame.max()}]\n")
                    if format == 'png':
                        f.write(f"scaling_applied: value * 65535 (direct 16-bit conversion)\n")
                        f.write(f"reconstruction: pixel_value / 65535\n")
                    elif format == 'exr':
                        f.write(f"scaling_applied: NONE (direct 32-bit float save)\n")
                        f.write(f"reconstruction: value is stored directly\n")
                    f.write(f"normalization: NONE - raw values preserved\n")
                    
                # Add resize information if applied
                if resize:
                    f.write(f"resized_to: {resize[1]}x{resize[0]} (WxH)\n")
                    f.write(f"resize_method: NEAREST (preserves depth values)\n")


def read_pfm(file_path, apply_scale=True):
    with open(file_path, 'rb') as f:
        # Read the header and determine dimensions and scale
        header = f.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise ValueError('Not a PFM file.')

        dimensions = f.readline().decode('ascii').rstrip()
        width, height = map(int, dimensions.split())
        scale = float(f.readline().rstrip())
        
        # Determine endianness
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        
        # Read the image data
        num_channels = 3 if color else 1
        data = np.fromfile(f, endian + 'f')
        data = np.reshape(data, (height, width, num_channels))
    if apply_scale:
        return np.flipud(data)*scale
    else:
        return np.flipud(data), scale
    
def read_exr(file_path, channels=None):
    """

    Args:
        file_path (str): path to the exr file to be read
        channels (list[str], optional): List of target channels to read from the exr. If None, read all channels in the exr.

    Returns:
        data (np.array): shape [H, W, C]
    """
    
    ifile = OpenEXR.InputFile(str(file_path))
    dw = ifile.header()['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    avalibale_channels = list(ifile.header()['channels'].keys())
    if channels is None:
        channels = avalibale_channels
    else:
        for target_c in channels:
            if target_c not in avalibale_channels:
                raise ValueError(f"Target channel {target_c} does not exist in {file_path}. Avalibale channels: {avalibale_channels}.")
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    return np.stack([np.frombuffer(ifile.channel(c, pt), dtype=np.float32).reshape(H, W) for c in channels], -1)