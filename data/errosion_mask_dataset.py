import os
import random
from math import floor

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .stereo_dataset import StereoDataset


def generate_random_bezier_path(rng, x_range, y_range, num_points=20, volatility=0.5, t_step=None):
    """
    Generates a discrete smooth path between two points.

    """
    x0, y0 = rng.randint(*x_range), rng.randint(*y_range)
    x3, y3 = rng.randint(*x_range), rng.randint(*y_range)

    # Calculate distance to scale the randomness
    dist = ((x3 - x0) ** 2 + (y3 - y0) ** 2) ** 0.5
    offset_scale = dist * volatility

    # Generate random control points (P1 and P2)
    # We base them roughly 1/3 and 2/3 along the path to maintain forward flow
    # but add a random offset to create the curve.
    p1_x = x0 + (x3 - x0) / 3 + rng.uniform(-offset_scale, offset_scale)
    p1_y = y0 + (y3 - y0) / 3 + rng.uniform(-offset_scale, offset_scale)

    p2_x = x0 + 2 * (x3 - x0) / 3 + rng.uniform(-offset_scale, offset_scale)
    p2_y = y0 + 2 * (y3 - y0) / 3 + rng.uniform(-offset_scale, offset_scale)

    def get_point(t):
        # The Bezier Formula
        # (1-t)^3 * P0 + 3(1-t)^2 * t * P1 + 3(1-t) * t^2 * P2 + t^3 * P3
        one_minus_t = 1 - t
        x = (one_minus_t**3 * x0) + (3 * one_minus_t**2 * t * p1_x) + (3 * one_minus_t * t**2 * p2_x) + (t**3 * x3)

        y = (one_minus_t**3 * y0) + (3 * one_minus_t**2 * t * p1_y) + (3 * one_minus_t * t**2 * p2_y) + (t**3 * y3)
        return (x, y)

    if t_step is not None:
        return get_point(t_step / (num_points - 1))

    path = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x, y = get_point(t)
        path.append((x, y))

    return np.array(path)


def draw_mask(img_w, img_h, x, y, line_width=20, dpi=100):
    fig = plt.figure(figsize=(img_w / dpi, img_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()  # Turn off ticks and spines
    fig.add_axes(ax)
    ax.set_facecolor("black")
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.plot(x, y, linewidth=line_width, color="white")
    fig.canvas.draw()
    rgba_buffer = np.array(fig.canvas.renderer.buffer_rgba())
    mask = Image.fromarray(rgba_buffer).convert("L")
    plt.close()
    return mask.resize((img_w, img_h))



def get_wavy_circle(rng, center=(500, 500), t=0):
    theta = np.linspace(0, 2 * np.pi, 500)
    base_r = rng.uniform(80, 120)
    amplitude = rng.uniform(9, 12)
    frequency = rng.uniform(7, 12)
    radius = base_r + amplitude * np.sin(frequency * theta)
    sx, sy = rng.uniform(0.8, 1.2), rng.uniform(2.5, 3.5)
    # Polar to Cartesian coordinates
    x = radius * np.cos(theta + t * np.pi) * sx + center[0]
    y = radius * np.sin(theta + t * np.pi) * sy + center[1]
    return x, y


def get_humanoid(rng, center=(500, 200), t=0):
    cx, cy = center
    rx, ry = rng.uniform(80, 100), rng.uniform(100, 120)
    body_height = rng.uniform(4.5, 6.5)
    theta = np.linspace(0, 2 * np.pi, 500)
    x = rx * np.sin(theta) + cx
    y = ry * np.cos(theta) + cy
    noise = np.linspace(0, 2 * np.pi, 100)
    for x_start, x_end, y_start, y_end in zip(
        [cx - 0.8 * rx, cx + 0.8 * rx, cx + 1.2 * rx, cx - 1.2 * rx],
        [cx + 0.8 * rx, cx + 1.2 * rx, cx - 1.2 * rx, cx - 0.8 * rx],
        [cy + ry, cy + ry, cy + body_height * ry, cy + body_height * ry],
        [cy + ry, cy + body_height * ry, cy + body_height * ry, cy + ry],
    ):
        xs = np.linspace(x_start, x_end, 100) + np.sin(noise + t * np.pi) * 5
        ys = np.linspace(y_start, y_end, 100) + np.cos(noise + t * np.pi) * 5
        x = np.concatenate([x, xs])
        y = np.concatenate([y, ys])

    return x, y


class ErosionMaskDataset(StereoDataset):
    def __init__(self, erosion_iterations: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.erosion_iterations = erosion_iterations

    def get_occlusion_mask(self, rgb_file_name, shot, frame, w, h, **kwargs):
        t_total = self.n_frames_in_shot(shot)
        rng = random.Random(hash(self.shot_names[shot]) % (2**32 - 1))
        shape = rng.choice(["circle", "humanoid"])
        if shape == "circle":
            center = generate_random_bezier_path(rng, (0, w), (0, h), num_points=t_total, t_step=frame)
            x, y = get_wavy_circle(rng, center, t=frame / t_total)
        elif shape == "humanoid":
            center = generate_random_bezier_path(rng, (0, w), (0, floor(h * 0.8)), num_points=t_total, t_step=frame)
            x, y = get_humanoid(rng, center, t=frame / t_total)
        else:
            raise NotImplementedError()

        mask = draw_mask(w, h, x, y, line_width=rng.randint(20, 45))
        return to_tensor(mask)
