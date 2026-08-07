"""Extract depth, RGB, and normal passes from a multi-channel EXR to separate EXR files."""

import argparse
import copy
import json
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import Imath
import numpy as np
import OpenEXR
import pandas as pd
from tqdm import tqdm


def read_channel(exr_file, channel_name, height, width):
    raw = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
    return np.frombuffer(raw, np.float32).reshape(height, width)


def write_exr(path: Path, channels_dict, height, width):
    """Write an EXR file from a dict of {channel_name: numpy_array}."""
    header = OpenEXR.Header(width, height)
    header["channels"] = {
        name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        for name in channels_dict
    }

    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    out = OpenEXR.OutputFile(str(path), header)
    out.writePixels(
        {name: arr.astype(np.float32).tobytes() for name, arr in channels_dict.items()}
    )
    out.close()


def extract_passes(input_path) -> tuple[str, dict, np.array, tuple, dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")
    output_dir = input_path.parent
    name = input_path.name
    scene = input_path.parent.name
    stem = input_path.stem

    depth_path = input_path / 'error'
    normal_path = depth_path
    rgb_path = depth_path
    camera_path = depth_path
    camera = {}
    depth = np.array([])
    normal_r = copy.deepcopy(depth)
    normal_g = copy.deepcopy(depth)
    normal_b = copy.deepcopy(depth)

    try:
        exr = OpenEXR.InputFile(str(input_path))
        header = exr.header()
        dw = exr.header()["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        # RGB from base HALF channels (R, G, B)
        rgb_path = output_dir / "rbg_exr" / name
        write_exr(rgb_path, {
            "R": read_channel(exr, "R", height, width),
            "G": read_channel(exr, "G", height, width),
            "B": read_channel(exr, "B", height, width),
        }, height, width)

        # Depth from FinalImageMQR_WorldDepth
        depth_path = output_dir / "depth_exr" / name
        depth = read_channel(exr, "FinalImageMQR_WorldDepth.R", height, width)
        write_exr(depth_path, {"Y": depth}, height, width)
        # Normal from FinalImageMRQ_WorldNormal
        normal_path = output_dir / "normals_exr" / name

        normal_r = read_channel(exr, "FinalImageMRQ_WorldNormal.R", height, width)
        normal_g = read_channel(exr, "FinalImageMRQ_WorldNormal.G", height, width)
        normal_b = read_channel(exr, "FinalImageMRQ_WorldNormal.B", height, width)

        write_exr(normal_path, {
            "R": normal_r, "G": normal_g, "B": normal_b,
        }, height, width)

        camera = extract_camera_metadata(header)
        camera_path = output_dir / "camera" / f"{stem}.json"
        write_camera_json(path=camera_path, camera=camera)

        exr.close()
        error = ''
    except Exception as e:
        error = str(e)

    frame_metadata = {
        "original_path": str(input_path),
        "depht_path": str(depth_path),
        "normal_path": str(normal_path),
        "rgb_path": str(rgb_path),
        "camera_path": str(camera_path),
        "error": error
    }

    return scene, camera, depth, (normal_r, normal_g, normal_b), frame_metadata


def get_header_float(header, key):
    val = header.get(key)
    if val is None:
        return None
    return float(val.decode() if isinstance(val, bytes) else val)


def get_header_str(header, key):
    val = header.get(key)
    if val is None:
        return None
    return val.decode() if isinstance(val, bytes) else str(val)


def extract_camera_metadata(header):
    camera = {
        "camera_name": get_header_str(header, "unreal/cameraName"),
        "position": {
            "x": get_header_float(header, "unreal/camera/curPos/x"),
            "y": get_header_float(header, "unreal/camera/curPos/y"),
            "z": get_header_float(header, "unreal/camera/curPos/z"),
        },
        "rotation": {
            "pitch": get_header_float(header, "unreal/camera/curRot/pitch"),
            "yaw": get_header_float(header, "unreal/camera/curRot/yaw"),
            "roll": get_header_float(header, "unreal/camera/curRot/roll"),
        },
        "prev_position": {
            "x": get_header_float(header, "unreal/camera/prevPos/x"),
            "y": get_header_float(header, "unreal/camera/prevPos/y"),
            "z": get_header_float(header, "unreal/camera/prevPos/z"),
        },
        "prev_rotation": {
            "pitch": get_header_float(header, "unreal/camera/prevRot/pitch"),
            "yaw": get_header_float(header, "unreal/camera/prevRot/yaw"),
            "roll": get_header_float(header, "unreal/camera/prevRot/roll"),
        },
        "focal_length": get_header_float(header, "unreal/camera/FinalImage/focalLength"),
        "fov": get_header_float(header, "unreal/camera/FinalImage/fov"),
        "fstop": get_header_float(header, "unreal/camera/FinalImage/fstop"),
        "focal_distance": get_header_float(header, "unreal/camera/FinalImage/focalDistance"),
        "sensor_width": get_header_float(header, "unreal/camera/FinalImage/sensorWidth"),
        "sensor_height": get_header_float(header, "unreal/camera/FinalImage/sensorHeight"),
        "sensor_aspect_ratio": get_header_float(header, "unreal/camera/FinalImage/sensorAspectRatio"),
    }
    return camera


def write_camera_json(path, camera):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(camera, f)

def _axis_stats(prefix, col):
    """Return flat dict with keys like prefix_min, prefix_max, etc."""
    return {
        f"{prefix}_min": float(col.min()),
        f"{prefix}_max": float(col.max()),
        f"{prefix}_median": float(np.median(col)),
        f"{prefix}_mean": float(col.mean()),
        f"{prefix}_std": float(col.std()),
        f"{prefix}_range": float(col.max() - col.min()),
    }

def compute_scene_stats(depth_frames, normal_frames):
    """Compute depth and normal statistics across all frames.

    Args:
        depth_frames: list of 2D numpy arrays (H, W) with depth values.
        normal_frames: list of 3-tuples (R, G, B) of 2D numpy arrays.

    Returns a flat dict suitable for a single dataframe row.
    """
    depth_frames = list(filter(lambda a: a.size != 0, depth_frames))
    normal_frames = list(filter(lambda a: a[0].size != 0, normal_frames))
    stats = {"num_valid_depth_frames": len(depth_frames), "num_valid_normal_frames": len(depth_frames), "error": ''}
    if not len(depth_frames) or not len(normal_frames):
        stats['e'] = f"depth or normal frames list empty."
        return stats

    # Per-frame depth stats aggregated across frames
    frame_mins = np.array([d.min() for d in depth_frames])
    frame_maxs = np.array([d.max() for d in depth_frames])
    frame_means = np.array([d.mean() for d in depth_frames])
    frame_medians = np.array([np.median(d) for d in depth_frames])
    frame_stds = np.array([d.std() for d in depth_frames])

    all_depth = np.concatenate([d.ravel() for d in depth_frames])
    stats.update(_axis_stats("depth_scene", all_depth))
    stats.update(_axis_stats("depth_frame_min", frame_mins))
    stats.update(_axis_stats("depth_frame_max", frame_maxs))
    stats.update(_axis_stats("depth_frame_mean", frame_means))
    stats.update(_axis_stats("depth_frame_median", frame_medians))
    stats.update(_axis_stats("depth_frame_std", frame_stds))

    # Normal stats per axis across all frames
    for ch, label in enumerate(["nx", "ny", "nz"]):
        all_vals = np.concatenate([n[ch].ravel() for n in normal_frames])
        stats.update(_axis_stats(f"normal_{label}_scene", all_vals))
        frame_means_n = np.array([n[ch].mean() for n in normal_frames])
        stats.update(_axis_stats(f"normal_{label}_frame_mean", frame_means_n))

    # Normal magnitude stats (should be ~1.0 for unit normals)
    mag_means = []
    for r, g, b in normal_frames:
        mag = np.sqrt(r**2 + g**2 + b**2)
        mag_means.append(mag.mean())
    stats.update(_axis_stats("normal_mag_frame_mean", np.array(mag_means)))

    return stats

def compute_camera_stats(cameras):
    """Compute position and rotation statistics across a list of camera dicts.

    Returns a flat dict suitable for a single dataframe row, with keys like
    pos_x_min, pos_x_max, rot_pitch_mean, delta_pos_total, etc.
    """

    cameras = list(filter(lambda c: len(c) > 0, cameras))
    stats = {"num_valid_cameras": len(cameras), "error": ''}
    if not len(cameras):
        stats['e'] = f"Valid camera list empty."
        return stats

    positions = np.array([[c["position"]["x"], c["position"]["y"], c["position"]["z"]] for c in cameras])
    rotations = np.array([[c["rotation"]["pitch"], c["rotation"]["yaw"], c["rotation"]["roll"]] for c in cameras])


    for i, label in enumerate(["x", "y", "z"]):
        stats.update(_axis_stats(f"pos_{label}", positions[:, i]))
    for i, label in enumerate(["pitch", "yaw", "roll"]):
        stats.update(_axis_stats(f"rot_{label}", rotations[:, i]))

    if len(cameras) > 1:
        pos_deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        rot_deltas = np.abs(np.diff(rotations, axis=0))
        stats.update(_axis_stats("delta_pos", pos_deltas))
        stats["delta_pos_total"] = float(pos_deltas.sum())
        for i, label in enumerate(["pitch", "yaw", "roll"]):
            stats.update(_axis_stats(f"delta_rot_{label}", rot_deltas[:, i]))

    return stats

def compile_scene(scene_name, frames, dataset, scenes):
    scene_path = Path(frames[scene_name]['frames_metadata'][0]['original_path']).parent
    row_scene = {"scene": [scene_name] * len(frames[scene_name]['frames_metadata'])}
    row_dataset = {
        "scene": scene_name,
        "scene_path": scene_path / f"{scene_path.stem}_scene.csv"
    }
    camera_stats = compute_camera_stats(frames[scene_name]["cameras"])
    row_dataset.update(camera_stats)
    scene_stats = compute_scene_stats(frames[scene_name]["depths"], frames[scene_name]["normals"])
    row_dataset.update(scene_stats)
    dataset.append(row_dataset)
    for row in frames[scene_name]['frames_metadata']:
        row_scene.update(row)
        scenes.append(row_scene)
    frames[scene_name] = {}


def _valid_exr(path):
    """Check that an EXR file exists and can be opened (header is intact)."""
    try:
        exr = OpenEXR.InputFile(str(path))
        exr.close()
        return True
    except Exception:
        print(f"Found invalid exr: {str(path)}")
        return False


def _valid_json(path):
    """Check that a JSON file exists and can be parsed."""
    try:
        with open(path) as f:
            json.load(f)
        return True
    except Exception:
        print(f"Found invalid json: {str(path)}")
        return False


def scene_is_complete(scene_files):
    """Return True if every output artifact exists and is valid for all frames."""
    if not scene_files:
        return False
    print("Found an already processed scene. Verifying it's integrity.")
    parent = Path('/Invalid/Scene/No/Valid/Scene/Files/Given')
    for p in tqdm(scene_files, desc="Checking scene artefacts", unit=' file'):
        parent = p.parent
        name = p.name
        stem = p.stem
        if not _valid_exr(parent / "rbg_exr" / name):
            return False
        if not _valid_exr(parent / "depth_exr" / name):
            return False
        if not _valid_exr(parent / "normals_exr" / name):
            return False
        if not _valid_json(parent / "camera" / f"{stem}.json"):
            return False
    # Scene-level CSV
    scene_csv = parent / f"{parent.stem}_scene.csv"
    if not scene_csv.exists():
        print(f"Did not find a valid scene index csv for scene: {parent}")
        return False
    return True


from itertools import groupby

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input dataset folder")
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess all scenes even if outputs already exist",
    )
    args = parser.parse_args()

    input_files = sorted(tqdm(Path(args.input).rglob("SD_*/*.exr"), desc="Locating all exr files", unit='exr'))

    # Group files by scene (parent directory name)
    grouped = groupby(input_files, key=lambda p: p.parent.name)

    dataset_rows = []

    for scene_name, scene_files in tqdm(grouped, desc="Scenes"):
        scene_files = list(scene_files)

        # --- Resume: skip completed scenes ---
        if not args.no_resume and scene_is_complete(scene_files):
            scene_path = scene_files[0].parent
            csv_path = scene_path / f"{scene_path.stem}_scene.csv"
            dataset_rows.append({
                "scene": scene_name,
                "scene_path": str(csv_path),
                "skipped": True,
            })
            print(f"  Skipping {scene_name} ({len(scene_files)} frames, already complete)")
            continue

        # --- Process this scene's files in parallel ---
        results = [None] * len(scene_files)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(extract_passes, p): i
                for i, p in enumerate(scene_files)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(scene_files),
                               desc=f"{scene_name}", unit=" exr", leave=False):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error: {scene_files[idx]}: {e}", file=sys.stderr)

        # Unpack, compute stats, write CSV, then free everything
        cameras, depths, normals, frame_metas = [], [], [], []
        for r in results:
            if r is None:
                continue
            _, camera, depth, normal, meta = r
            cameras.append(camera)
            depths.append(depth)
            normals.append(normal)
            frame_metas.append(meta)

        if not frame_metas:
            continue

        scene_path = Path(frame_metas[0]["original_path"]).parent
        csv_path = scene_path / f"{scene_path.stem}_scene.csv"

        row = {"scene": scene_name, "scene_path": str(csv_path)}
        row.update(compute_camera_stats(cameras))
        row.update(compute_scene_stats(depths, normals))
        dataset_rows.append(row)

        # Write per-scene CSV
        scene_df = pd.DataFrame(frame_metas)
        scene_df.insert(0, "scene", scene_name)
        scene_df.to_csv(csv_path, index=False)

        # Free the heavy arrays
        del cameras, depths, normals, results

    # Final dataset CSV (just metadata rows, tiny)
    pd.DataFrame(dataset_rows).to_csv(Path(args.input) / "dataset.csv", index=False)




if __name__ == "__main__":
    main()
