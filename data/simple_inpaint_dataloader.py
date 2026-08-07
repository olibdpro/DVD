import logging

import torch
import kornia
from natsort import natsorted
from kornia.io import ImageLoadType

from utils.timer import Timer


class SimpleInpaintDataLoader:
    """A simple dataloader to load images from directories for inpainting inference."""

    def __init__(
        self,
        input_dirs,
        occlusion_mask_dirs,
        image_transforms,
        mask_transforms,
        output_dirs,
        max_clip_len=None,
        start_clip_idx=0,
    ):
        """
        Args:
            input_dirs (List[str]): Paths to input image directories. The assumption is that each directory contains images from a single shot.
            occlusion_mask_dirs (List[str]): Paths to corresponding occlusion mask directories.
            image_transforms (List[Callable]): Transformations to apply to input images before feeding to the model.
            mask_tgransforms (List[Callable]):  Transformations to apply to input occlusion masks before feeding to the model.
            output_dirs: Paths to corresponding output directories.
        """
        self.input_dirs = input_dirs
        self.occlusion_mask_dirs = occlusion_mask_dirs
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.output_dirs = output_dirs
        self.max_clip_len = max_clip_len
        self.start_clip_idx = start_clip_idx

    def __len__(self):
        return len(self.input_dirs)

    def __iter__(self):
        for input_dir, mask_dir, output_dir in zip(self.input_dirs, self.occlusion_mask_dirs, self.output_dirs):
            input_paths = natsorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[
                self.start_clip_idx :
            ]
            mask_paths = natsorted([p for p in mask_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[
                self.start_clip_idx :
            ]
            if self.max_clip_len is not None:
                input_paths = input_paths[: self.max_clip_len]
                mask_paths = mask_paths[: self.max_clip_len]
            if len(input_paths) != len(mask_paths):
                logging.error(
                    f"Number of input images and occlusion masks do not match in {input_dir} ({len(input_paths)}) and {mask_dir} ({len(mask_paths)}). Skipping."
                )
                continue
            if not input_paths:
                logging.error(f"No input images found in {input_dir}. Skipping.")
                continue

            with Timer("Load and Transform Input Images"):
                images = torch.stack(
                    [kornia.io.load_image(path, ImageLoadType.RGB32, device="cpu") for path in input_paths]
                )
                masks = torch.stack(
                    [kornia.io.load_image(path, ImageLoadType.GRAY32, device="cpu") for path in mask_paths]
                )
                occluded_images = images * (~masks.bool())
            batch = {
                "input_rgb": occluded_images,
                "occlusion_mask": masks,
                "output_dir": output_dir,
                "target_rgb": images,
                "og_target": images.clone(),
                "og_occlusion_mask": masks.clone(),
                "image_name": [p.name for p in input_paths],
                "input_dir": input_dir,
                "mask_dir": mask_dir,
            }
            yield self.mask_transforms(self.image_transforms(batch))
