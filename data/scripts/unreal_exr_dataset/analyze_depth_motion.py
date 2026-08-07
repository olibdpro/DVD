"""Analyze dataset.csv for depth characteristics and depth changes correlated/uncorrelated with camera motion."""

import csv
import math

with open("dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter valid scenes with depth data
FAR_PLANE_THRESHOLD = 5_000_000  # exclude scenes hitting the 10M far-plane clamp
MIN_DEPTH_RANGE = 1.0            # exclude degenerate/flat depth scenes

all_with_depth = []
for r in rows:
    if r["skipped"] == "True" or not r["num_valid_cameras"]:
        continue
    if not r["depth_scene_range"]:
        continue
    all_with_depth.append(r)

# Apply filters
valid = []
excluded_farplane = []
excluded_degenerate = []
for r in all_with_depth:
    scene_max = float(r["depth_scene_max"]) if r["depth_scene_max"] else 0.0
    scene_range = float(r["depth_scene_range"]) if r["depth_scene_range"] else 0.0
    if scene_max >= FAR_PLANE_THRESHOLD:
        excluded_farplane.append(r["scene"])
        continue
    if scene_range < MIN_DEPTH_RANGE:
        excluded_degenerate.append(r["scene"])
        continue
    valid.append(r)

print(f"Total scenes with depth data: {len(all_with_depth)}")
print(f"Excluded (far-plane >= {FAR_PLANE_THRESHOLD}): {len(excluded_farplane)} scenes")
print(f"  e.g. {', '.join(excluded_farplane[:5])}{'...' if len(excluded_farplane) > 5 else ''}")
print(f"Excluded (degenerate range < {MIN_DEPTH_RANGE}): {len(excluded_degenerate)} scenes")
print(f"  e.g. {', '.join(excluded_degenerate[:5])}{'...' if len(excluded_degenerate) > 5 else ''}")
print(f"Valid scenes after filtering: {len(valid)}\n")


def fval(row, col):
    v = row.get(col, "")
    return float(v) if v else 0.0


def top_n(scenes, key_func, n=3, reverse=True):
    scored = [(s, key_func(s)) for s in scenes]
    scored.sort(key=lambda x: x[1], reverse=reverse)
    return scored[:n]


def print_top(scenes, key_func, extra_cols=None, n=3, reverse=True):
    results = top_n(scenes, key_func, n, reverse)
    for rank, (s, val) in enumerate(results, 1):
        extras = ""
        if extra_cols:
            parts = [f"{c}={fval(s, c):.2f}" for c in extra_cols]
            extras = "  (" + ", ".join(parts) + ")"
        print(f"  {rank}. {s['scene']:10s}  {val:.4f}{extras}")


# =========================================================================
# Compute derived metrics for each scene
# =========================================================================
for r in valid:
    # Camera motion magnitude: combine position + rotation
    r["_cam_pos_total"] = fval(r, "delta_pos_total")
    r["_cam_rot_total"] = (fval(r, "rot_pitch_range") +
                           fval(r, "rot_yaw_range") +
                           fval(r, "rot_roll_range"))
    # Normalized camera motion (pos + scaled rotation)
    # Use position total as primary, rotation as secondary signal
    r["_cam_motion"] = r["_cam_pos_total"]

    # Depth temporal variation: how much the mean depth changes across frames
    r["_depth_mean_std"] = fval(r, "depth_frame_mean_std")
    r["_depth_mean_range"] = fval(r, "depth_frame_mean_range")

    # Near-field variation: how much the closest depth changes across frames
    r["_depth_min_std"] = fval(r, "depth_frame_min_std")
    r["_depth_min_range"] = fval(r, "depth_frame_min_range")
    r["_depth_min_mean"] = fval(r, "depth_frame_min_mean")  # avg closest distance

    # Far-field variation: how much the farthest depth changes across frames
    r["_depth_max_std"] = fval(r, "depth_frame_max_std")
    r["_depth_max_range"] = fval(r, "depth_frame_max_range")
    r["_depth_max_mean"] = fval(r, "depth_frame_max_mean")  # avg farthest distance

    # Normal temporal variation (surface orientation change across frames)
    r["_normal_var"] = (fval(r, "normal_nx_frame_mean_std") +
                        fval(r, "normal_ny_frame_mean_std") +
                        fval(r, "normal_nz_frame_mean_std"))

    # Correlation proxy: depth_change / camera_motion
    # High ratio = depth changes a lot relative to camera motion (scene dynamics)
    # Low ratio = depth changes mostly from camera movement
    cam = r["_cam_motion"]
    if cam > 0:
        r["_depth_per_cam"] = r["_depth_mean_std"] / cam
        r["_farfield_per_cam"] = r["_depth_max_std"] / cam
    else:
        r["_depth_per_cam"] = float("inf") if r["_depth_mean_std"] > 0 else 0
        r["_farfield_per_cam"] = float("inf") if r["_depth_max_std"] > 0 else 0


# =========================================================================
# Percentile split for camera motion (to define "low" and "high" motion)
# =========================================================================
cam_motions = sorted([r["_cam_motion"] for r in valid])
p25 = cam_motions[len(cam_motions) // 4]
p75 = cam_motions[3 * len(cam_motions) // 4]
median_cam = cam_motions[len(cam_motions) // 2]

low_motion = [r for r in valid if r["_cam_motion"] <= p25]
high_motion = [r for r in valid if r["_cam_motion"] >= p75]

print(f"Camera motion stats: p25={p25:.1f}, median={median_cam:.1f}, p75={p75:.1f}")
print(f"Low-motion scenes (<=p25): {len(low_motion)}")
print(f"High-motion scenes (>=p75): {len(high_motion)}")


# =========================================================================
# Scenes with at least some camera motion (above p25)
has_motion = [r for r in valid if r["_cam_motion"] > p25]
print(f"Scenes with meaningful camera motion (>p25): {len(has_motion)}")

print("\n" + "=" * 70)
print("1. NARROWEST DEPTH (smallest depth range, with camera motion > p25)")
print("   Entire scene depth is compressed into a small range")
print("=" * 70)
print_top(has_motion, lambda r: fval(r, "depth_scene_range"),
          extra_cols=["depth_scene_min", "depth_scene_max", "delta_pos_total"], reverse=False)


# =========================================================================
print("\n" + "=" * 70)
print("2. DEEPEST DEPTH (largest max depth in scene)")
print("   Objects/environment extends very far from camera")
print("=" * 70)
print_top(valid, lambda r: fval(r, "depth_scene_max"),
          extra_cols=["depth_scene_range", "depth_scene_mean"])


# =========================================================================
print("\n" + "=" * 70)
print("3. NARROWEST/NEAR-FIELD DEPTH CHANGES (close to camera)")
print("   The closest depth value varies a lot across frames")
print("   = something moving/appearing near the camera")
print("=" * 70)
print("\n  By depth_frame_min_std (variation of closest point across frames):")
print_top(valid, lambda r: r["_depth_min_std"],
          extra_cols=["depth_frame_min_mean", "depth_frame_min_range"])

print("\n  Filtered: near-field changes where avg min depth < scene median depth:")
near_changes = [r for r in valid if r["_depth_min_mean"] < fval(r, "depth_scene_median")]
print_top(near_changes, lambda r: r["_depth_min_std"],
          extra_cols=["depth_frame_min_mean", "depth_frame_min_range"])


# =========================================================================
print("\n" + "=" * 70)
print("4. SMALLEST DEPTH CHANGE (most static depth across frames)")
print("   Very little changes in the depth buffer over time")
print("=" * 70)
print("\n  By depth_frame_mean_std (smallest variation of mean depth):")
print_top(valid, lambda r: r["_depth_mean_std"],
          extra_cols=["depth_frame_mean_range", "depth_scene_range"], reverse=False)

print("\n  Also checking normal stability (low normal variation = truly static):")
print_top(valid, lambda r: r["_depth_mean_std"] + r["_normal_var"] * 100,
          extra_cols=["depth_frame_mean_std", "_normal_var"], reverse=False)


# =========================================================================
print("\n" + "=" * 70)
print("5. DEEPEST/FAR-FIELD DEPTH CHANGES CORRELATED WITH CAMERA MOTION")
print("   Camera moves a lot AND far-field depth changes significantly")
print("   = camera exploring, revealing new distant geometry")
print("=" * 70)
print("\n  High-motion scenes ranked by far-field depth variation (depth_frame_max_std):")
print_top(high_motion, lambda r: r["_depth_max_std"],
          extra_cols=["depth_frame_max_mean", "_cam_pos_total", "depth_frame_max_range"])

print("\n  High-motion scenes ranked by combined far-field + normal change:")
print_top(high_motion, lambda r: r["_depth_max_std"] + r["_normal_var"] * 50,
          extra_cols=["_depth_max_std", "_normal_var", "_cam_pos_total"])


# =========================================================================
print("\n" + "=" * 70)
print("6. BIGGEST DEPTH CHANGES CORRELATED WITH CAMERA MOTION")
print("   Camera moves a lot AND overall depth distribution shifts significantly")
print("   = camera moving through varying environments")
print("=" * 70)
print("\n  High-motion scenes ranked by depth_frame_mean_std:")
print_top(high_motion, lambda r: r["_depth_mean_std"],
          extra_cols=["depth_frame_mean_range", "_cam_pos_total"])

print("\n  High-motion scenes ranked by depth_frame_mean_range:")
print_top(high_motion, lambda r: r["_depth_mean_range"],
          extra_cols=["depth_frame_mean_std", "_cam_pos_total"])


# =========================================================================
print("\n" + "=" * 70)
print("7. DEEPEST/FAR-FIELD DEPTH CHANGES UNCORRELATED WITH CAMERA")
print("   Camera barely moves BUT far-field depth changes a lot")
print("   = something moving in/out at distance (e.g. character walking away)")
print("=" * 70)
print("\n  Low-motion scenes ranked by far-field depth variation (depth_frame_max_std):")
print_top(low_motion, lambda r: r["_depth_max_std"],
          extra_cols=["depth_frame_max_mean", "_cam_pos_total", "depth_frame_max_range"])

print("\n  Low-motion scenes ranked by depth_per_cam ratio (depth change / camera motion):")
finite_low = [r for r in low_motion if r["_farfield_per_cam"] < float("inf")]
print_top(finite_low, lambda r: r["_farfield_per_cam"],
          extra_cols=["_depth_max_std", "_cam_pos_total"])


# =========================================================================
print("\n" + "=" * 70)
print("8. BIGGEST DEPTH CHANGES UNCORRELATED WITH CAMERA MOTION")
print("   Camera barely moves BUT overall depth changes a lot")
print("   = dynamic scene elements (characters, objects moving)")
print("=" * 70)
print("\n  Low-motion scenes ranked by depth_frame_mean_std:")
print_top(low_motion, lambda r: r["_depth_mean_std"],
          extra_cols=["depth_frame_mean_range", "_cam_pos_total", "_normal_var"])

print("\n  Low-motion scenes ranked by depth_per_cam ratio:")
finite_low2 = [r for r in low_motion if r["_depth_per_cam"] < float("inf")]
print_top(finite_low2, lambda r: r["_depth_per_cam"],
          extra_cols=["_depth_mean_std", "_cam_pos_total"])

print("\n  Low-motion scenes with highest normal variation (surface changes = object motion):")
print_top(low_motion, lambda r: r["_normal_var"],
          extra_cols=["_depth_mean_std", "_cam_pos_total"])


# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

categories = [
    ("Narrowest depth",              top_n(has_motion, lambda r: fval(r, "depth_scene_range"), 1, reverse=False)),
    ("Deepest depth",                top_n(valid, lambda r: fval(r, "depth_scene_max"), 1)),
    ("Near-field depth changes",     top_n(valid, lambda r: r["_depth_min_std"], 1)),
    ("Smallest depth change",        top_n(valid, lambda r: r["_depth_mean_std"], 1, reverse=False)),
    ("Far-field change + cam motion",top_n(high_motion, lambda r: r["_depth_max_std"], 1)),
    ("Big depth change + cam motion",top_n(high_motion, lambda r: r["_depth_mean_std"], 1)),
    ("Far-field change, static cam", top_n(low_motion, lambda r: r["_depth_max_std"], 1)),
    ("Big depth change, static cam", top_n(low_motion, lambda r: r["_depth_mean_std"], 1)),
]

print(f"\n  {'Category':<35s} {'Scene':10s} {'Value':>12s}")
print(f"  {'-'*35} {'-'*10} {'-'*12}")
for cat, results in categories:
    if results:
        s, v = results[0]
        print(f"  {cat:<35s} {s['scene']:10s} {v:12.2f}")


# =========================================================================
print("\n" + "=" * 70)
print("SD_483 SPOTLIGHT")
print("=" * 70)

sd483_list = [r for r in valid if r["scene"] == "SD_483"]
if not sd483_list:
    # Check if it was filtered out; look in all_with_depth
    sd483_list = [r for r in all_with_depth if r["scene"] == "SD_483"]
    if sd483_list:
        print("  (Note: SD_483 was excluded by filters, showing raw values anyway)")

if sd483_list:
    s = sd483_list[0]
    depth_cols = [c for c in s.keys() if c.startswith("depth_") or c.startswith("normal_") or c.startswith("_")]
    cam_cols = ["delta_pos_total", "delta_pos_max", "delta_pos_std",
                "rot_pitch_range", "rot_yaw_range", "rot_roll_range",
                "pos_x_range", "pos_y_range", "pos_z_range"]

    print(f"\n  Scene: {s['scene']}")
    print(f"  Cameras: {s['num_valid_cameras']}")

    print("\n  Camera motion:")
    for c in cam_cols:
        print(f"    {c:30s} = {fval(s, c):.4f}")

    print("\n  Depth stats:")
    depth_display = [
        "depth_scene_min", "depth_scene_max", "depth_scene_range",
        "depth_scene_mean", "depth_scene_std",
        "depth_frame_min_mean", "depth_frame_min_std", "depth_frame_min_range",
        "depth_frame_max_mean", "depth_frame_max_std", "depth_frame_max_range",
        "depth_frame_mean_mean", "depth_frame_mean_std", "depth_frame_mean_range",
        "depth_frame_median_mean", "depth_frame_median_std", "depth_frame_median_range",
        "depth_frame_std_mean", "depth_frame_std_std", "depth_frame_std_range",
    ]
    for c in depth_display:
        print(f"    {c:30s} = {fval(s, c):.4f}")

    print("\n  Normal stats:")
    normal_display = [
        "normal_nx_frame_mean_std", "normal_ny_frame_mean_std", "normal_nz_frame_mean_std",
        "normal_nx_frame_mean_range", "normal_ny_frame_mean_range", "normal_nz_frame_mean_range",
        "normal_mag_frame_mean_std", "normal_mag_frame_mean_range",
    ]
    for c in normal_display:
        print(f"    {c:30s} = {fval(s, c):.6f}")

    # Rank SD_483 within valid set across key metrics
    print("\n  Ranking among valid scenes:")
    rank_metrics = [
        ("depth_frame_mean_std",      "_depth_mean_std",  True),
        ("depth_frame_min_std",       "_depth_min_std",   True),
        ("depth_frame_max_std",       "_depth_max_std",   True),
        ("normal variation (sum std)", "_normal_var",      True),
        ("depth_frame_std_std",       None,               True),
    ]
    for label, key, desc in rank_metrics:
        if key:
            vals = sorted([r[key] for r in valid], reverse=desc)
            target = s.get(key, fval(s, key) if isinstance(key, str) and not key.startswith("_") else 0)
            # Recompute for SD_483 if it's a derived field
            if key == "_normal_var":
                target = (fval(s, "normal_nx_frame_mean_std") +
                          fval(s, "normal_ny_frame_mean_std") +
                          fval(s, "normal_nz_frame_mean_std"))
            elif key == "_depth_mean_std":
                target = fval(s, "depth_frame_mean_std")
            elif key == "_depth_min_std":
                target = fval(s, "depth_frame_min_std")
            elif key == "_depth_max_std":
                target = fval(s, "depth_frame_max_std")
        else:
            vals = sorted([fval(r, "depth_frame_std_std") for r in valid], reverse=desc)
            target = fval(s, "depth_frame_std_std")

        # Find rank
        rank = 1
        for v in vals:
            if (desc and v > target) or (not desc and v < target):
                rank += 1
            else:
                break
        print(f"    {label:35s}: rank {rank}/{len(valid)}  (value={target:.4f})")
else:
    print("  SD_483 not found in dataset.")
