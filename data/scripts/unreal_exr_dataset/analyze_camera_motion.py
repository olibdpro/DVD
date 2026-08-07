"""Analyze dataset.csv to find scenes with the purest camera movements.

For each movement type, we want scenes that MAXIMIZE the target axis
while MINIMIZING all other axes (translation + rotation).
"""

import csv

with open("dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter valid scenes with good depth data
FAR_PLANE_THRESHOLD = 5_000_000
MIN_DEPTH_RANGE = 1.0

all_valid = []
for r in rows:
    if r["skipped"] == "True" or not r["num_valid_cameras"]:
        continue
    all_valid.append(r)

valid = []
excluded = {"zero_depth": [], "far_plane": [], "no_depth": []}
for r in all_valid:
    depth_range = float(r["depth_scene_range"]) if r.get("depth_scene_range") else None
    depth_max = float(r["depth_scene_max"]) if r.get("depth_scene_max") else None

    if depth_range is None:
        excluded["no_depth"].append(r["scene"])
        continue
    if depth_range < MIN_DEPTH_RANGE:
        excluded["zero_depth"].append(r["scene"])
        continue
    if depth_max is not None and depth_max >= FAR_PLANE_THRESHOLD:
        excluded["far_plane"].append(r["scene"])
        continue
    valid.append(r)

print(f"Total scenes: {len(all_valid)}")
print(f"Excluded (no/zero depth): {len(excluded['no_depth']) + len(excluded['zero_depth'])} scenes")
print(f"Excluded (far-plane >= {FAR_PLANE_THRESHOLD}): {len(excluded['far_plane'])} scenes")
print(f"Valid scenes with clean depth: {len(valid)}\n")

EPS = 1e-6  # avoid division by zero


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
        print(f"  {rank}. {s['scene']:10s}  score={val:.4f}{extras}")


# =========================================================================
# Precompute normalized axis values for each scene.
# We normalize each axis by its max across all scenes so that
# translation and rotation ranges are comparable in [0, 1].
# =========================================================================
# Unreal Engine coordinate system (left-handed, Z-up):
#   X = forward,  Y = right,  Z = up
#   Yaw = around Z,  Pitch = around Y,  Roll = around X
all_axes = {
    "pos_x_range": "Dolly",      # forward/backward
    "pos_y_range": "Truck",      # left/right
    "pos_z_range": "Pedestal",   # up/down
    "rot_yaw_range": "Pan",
    "rot_pitch_range": "Tilt",
    "rot_roll_range": "Roll",
}

# Max values for normalization
axis_maxes = {}
for col in all_axes:
    axis_maxes[col] = max(fval(r, col) for r in valid) or 1.0

for r in valid:
    for col in all_axes:
        r[f"_norm_{col}"] = fval(r, col) / axis_maxes[col]


def purity_score(r, target_col):
    """Score = target_normalized / (sum_of_all_other_normalized + eps).

    High score = target axis is large AND all others are small.
    """
    target = r[f"_norm_{target_col}"]
    others = sum(r[f"_norm_{c}"] for c in all_axes if c != target_col)
    return target / (others + EPS)


def purity_score_trans(r, target_col):
    """Purity within translation axes only (ignore rotation)."""
    trans_cols = ["pos_x_range", "pos_y_range", "pos_z_range"]
    target = r[f"_norm_{target_col}"]
    others_trans = sum(r[f"_norm_{c}"] for c in trans_cols if c != target_col)
    others_rot = sum(r[f"_norm_{c}"] for c in all_axes if c not in trans_cols)
    return target / (others_trans + others_rot + EPS)


def purity_score_rot(r, target_col):
    """Purity within rotation axes only (ignore translation)."""
    rot_cols = ["rot_yaw_range", "rot_pitch_range", "rot_roll_range"]
    target = r[f"_norm_{target_col}"]
    others_rot = sum(r[f"_norm_{c}"] for c in rot_cols if c != target_col)
    others_trans = sum(r[f"_norm_{c}"] for c in all_axes if c not in rot_cols)
    return target / (others_rot + others_trans + EPS)


# =========================================================================
# For "longest": purity_score (maximize target range, minimize others)
# For "fastest": same idea but using per-frame deltas
# For "jittery": same idea but using delta std
# =========================================================================

# Delta columns per axis (for fastest / jittery)
# Translation: we only have aggregate delta_pos_max/std, not per-axis deltas.
# So for translation fastest/jittery, we use: purity of range * delta magnitude
# For rotation: we have per-axis deltas.

delta_max_cols = {
    "rot_yaw_range": "delta_rot_yaw_max",
    "rot_pitch_range": "delta_rot_pitch_max",
    "rot_roll_range": "delta_rot_roll_max",
}
delta_std_cols = {
    "rot_yaw_range": "delta_rot_yaw_std",
    "rot_pitch_range": "delta_rot_pitch_std",
    "rot_roll_range": "delta_rot_roll_std",
}

# For translation fastest/jittery: weight delta_pos_max/std by axis purity
# (if the scene is a pure dolly, then delta_pos_max is mostly dolly speed)


# =========================================================================
print("=" * 70)
print("TRANSLATION MOVEMENTS (maximize target, minimize all others)")
print("=" * 70)

trans_configs = [
    ("Dolly (X fwd)",     "pos_x_range"),
    ("Truck (Y right)",   "pos_y_range"),
    ("Pedestal (Z up)",   "pos_z_range"),
]

other_trans = {"pos_x_range", "pos_y_range", "pos_z_range"}
other_rot = {"rot_yaw_range", "rot_pitch_range", "rot_roll_range"}

for name, target_col in trans_configs:
    print(f"\n--- {name} ---")

    # Filter: target must be dominant among translation axes
    subset = [r for r in valid if fval(r, target_col) > 0 and
              all(fval(r, target_col) >= fval(r, c) for c in other_trans)]

    if not subset:
        print("  No scenes found")
        continue

    other_cols = [c for c in all_axes if c != target_col]

    print(f"\n  Longest (max {target_col}, min others):")
    print_top(subset, lambda r, t=target_col: purity_score(r, t),
              extra_cols=[target_col] + other_cols)

    print(f"\n  Fastest (highest delta_pos_max, weighted by purity):")
    print_top(subset,
              lambda r, t=target_col: fval(r, "delta_pos_max") * purity_score(r, t),
              extra_cols=["delta_pos_max", target_col] + other_cols)

    print(f"\n  Most jittery (highest delta_pos_std, weighted by purity):")
    print_top(subset,
              lambda r, t=target_col: fval(r, "delta_pos_std") * purity_score(r, t),
              extra_cols=["delta_pos_std", target_col] + other_cols)


# =========================================================================
print("\n" + "=" * 70)
print("ROTATION MOVEMENTS (maximize target, minimize all others)")
print("=" * 70)

rot_configs = [
    ("Pan (Yaw)",     "rot_yaw_range"),
    ("Tilt (Pitch)",  "rot_pitch_range"),
    ("Roll",          "rot_roll_range"),
]

for name, target_col in rot_configs:
    print(f"\n--- {name} ---")

    # Filter: target must be dominant among rotation axes
    subset = [r for r in valid if fval(r, target_col) > 0 and
              all(fval(r, target_col) >= fval(r, c) for c in other_rot)]

    if not subset:
        print("  No scenes found")
        continue

    dmax_col = delta_max_cols[target_col]
    dstd_col = delta_std_cols[target_col]
    other_cols = [c for c in all_axes if c != target_col]

    print(f"\n  Longest (max {target_col}, min others):")
    print_top(subset, lambda r, t=target_col: purity_score(r, t),
              extra_cols=[target_col] + other_cols)

    print(f"\n  Fastest (highest {dmax_col}, weighted by purity):")
    print_top(subset,
              lambda r, t=target_col, d=dmax_col: fval(r, d) * purity_score(r, t),
              extra_cols=[dmax_col, target_col] + other_cols)

    print(f"\n  Most jittery (highest {dstd_col}, weighted by purity):")
    print_top(subset,
              lambda r, t=target_col, d=dstd_col: fval(r, d) * purity_score(r, t),
              extra_cols=[dstd_col, target_col] + other_cols)


# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

summary = []
for name, target_col in trans_configs:
    subset = [r for r in valid if fval(r, target_col) > 0 and
              all(fval(r, target_col) >= fval(r, c) for c in other_trans)]
    if not subset:
        summary.append((name, "N/A", "N/A", "N/A"))
        continue
    longest = top_n(subset, lambda r, t=target_col: purity_score(r, t), 1)[0][0]["scene"]
    fastest = top_n(subset, lambda r, t=target_col: fval(r, "delta_pos_max") * purity_score(r, t), 1)[0][0]["scene"]
    jittery = top_n(subset, lambda r, t=target_col: fval(r, "delta_pos_std") * purity_score(r, t), 1)[0][0]["scene"]
    summary.append((name, longest, fastest, jittery))

for name, target_col in rot_configs:
    subset = [r for r in valid if fval(r, target_col) > 0 and
              all(fval(r, target_col) >= fval(r, c) for c in other_rot)]
    if not subset:
        summary.append((name, "N/A", "N/A", "N/A"))
        continue
    dmax_col = delta_max_cols[target_col]
    dstd_col = delta_std_cols[target_col]
    longest = top_n(subset, lambda r, t=target_col: purity_score(r, t), 1)[0][0]["scene"]
    fastest = top_n(subset, lambda r, t=target_col, d=dmax_col: fval(r, d) * purity_score(r, t), 1)[0][0]["scene"]
    jittery = top_n(subset, lambda r, t=target_col, d=dstd_col: fval(r, d) * purity_score(r, t), 1)[0][0]["scene"]
    summary.append((name, longest, fastest, jittery))

print(f"\n  {'Movement':<15s} {'Longest':10s} {'Fastest':10s} {'Most Jittery':12s}")
print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
for name, longest, fastest, jittery in summary:
    print(f"  {name:<15s} {longest:10s} {fastest:10s} {jittery:12s}")
