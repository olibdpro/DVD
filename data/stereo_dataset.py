import os
import re
import bisect
import random
import logging
from glob import glob
from collections import defaultdict

import numpy as np
import torch
import kornia
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
from kornia.io import ImageLoadType
from torch.utils.data import Dataset

from utils.io import read_exr
from utils.images import dilate, connected_component_filter
from utils.stereo import warp_image_stereo, get_occlusion_mask, get_occlusion_mask_from_depth

from .shots_dataset_interface import BaseShotsDataset


class StereoDataset(BaseShotsDataset, Dataset):
    # FUTURE: Currently the logic inside this dataset assumes using left eye image as reference point. Might need to change that based on future needs.
    def __init__(
        self,
        root_dir: str,
        left_pattern: str,
        shot_pattern: str,
        frame_pattern: str,
        disp_format_str: str,
        occlusion_format_str: str,
        splat_format_str: str,
        **kwargs,
    ):
        self.root_dir = root_dir
        self.frame_pattern = re.compile(frame_pattern)
        self.depth_format_str = kwargs.get("depth_format_str", None)
        self.disp_format_str = self.validate_format_str(disp_format_str)
        self.splat_format_str = self.validate_format_str(splat_format_str)
        self.occlusion_format_str = self.validate_format_str(occlusion_format_str)
        self.occlusion_threshold = kwargs.get("occlusion_threshold", 5)

        all_file_paths = natsorted([f for f in glob(left_pattern, root_dir=root_dir)])

        if len(all_file_paths) == 0:
            raise ValueError(f"No data is found under root dir {root_dir} using the left pattern {left_pattern}")

        all_valid_file_paths = [f for f in all_file_paths if self.is_valid_data(f)]
        if len(all_valid_file_paths) == 0:
            e = ValueError(f"After filtering, no valid data is found under root dir {root_dir}.")
            logging.error(e)
            raise e
        if len(all_valid_file_paths) < len(all_file_paths):
            logging.warning(
                f"{type(self).__name__} filtered out {len(all_file_paths) - len(all_valid_file_paths)} files. See log for details."
            )

        super().__init__(all_files_paths=all_valid_file_paths, shot_pattern=shot_pattern, frame_pattern=frame_pattern)
        logging.info(f"{self.__class__.__name__} initialized from root directory {root_dir}.")

    def validate_format_str(self, f_str):
        if f_str is not None and (r"{file_name}" not in f_str and r"{base_name}" not in f_str):
            raise ValueError(f"Format string <{f_str}> does not include {{file_name}} or {{base_name}}")
        return f_str

    def build_lookup(self, all_files_paths) -> dict:
        lookup = defaultdict(list)
        for index, file in tqdm(enumerate(all_files_paths), "Indexing Stereo Data", total=len(all_files_paths)):
            shot_name, frame_idx = self.parse_file_path(file)
            bisect.insort_left(lookup[shot_name], (frame_idx, file, file.replace("left", "right")))
        return lookup

    def get_splat(self, rgb_file_name):
        splat_path = self.splat_format_str.format(file_name=self.modify_file_name(rgb_file_name))
        if not os.path.exists(splat_path):
            logging.debug(f"No {splat_path}. Generating.")
            disparity_map = self.get_disparity(rgb_file_name)
            rgb = np.array(Image.open(os.path.join(self.root_dir, rgb_file_name)))
            os.makedirs(os.path.dirname(splat_path), exist_ok=True)
            Image.fromarray(warp_image_stereo(rgb, disparity_map)).save(splat_path)
        return kornia.io.load_image(splat_path, ImageLoadType.RGB32, device="cpu")

    def get_occlusion_mask(self, rgb_file_name, **kwargs):
        mask_path = self.get_file_path(self.occlusion_format_str, rgb_file_name, type_name="disp_occlusion")
        if not os.path.exists(mask_path):
            logging.debug(f"No {mask_path}. Generating.")
            if not os.path.exists(self.get_file_path(self.disp_format_str, rgb_file_name, ".exr")):
                logging.debug(f"Disparity maps not found for {rgb_file_name}. Generating occlusion from depth instead.")
                return self.get_occlusion_mask_from_depth(rgb_file_name)
            l_name, r_name = self.get_stereo_pair(rgb_file_name)
            D_L = self.get_disparity(l_name)
            D_R = self.get_disparity(r_name)
            occlusion = get_occlusion_mask(D_L, D_R, self.occlusion_threshold, rgb_file_name == l_name)
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            Image.fromarray(occlusion * 255).save(mask_path)
        return kornia.io.load_image(mask_path, ImageLoadType.GRAY32, device="cpu")

    def get_occlusion_mask_from_depth(self, rgb_file_name, **kwargs):
        mask_path = self.get_file_path(self.occlusion_format_str, rgb_file_name, type_name="depth_occlusion")
        try:
            return kornia.io.load_image(mask_path, ImageLoadType.GRAY32, device="cpu")
        except Exception as e:
            depth = self.get_depth(rgb_file_name)
            l_name, _ = self.get_stereo_pair(rgb_file_name)
            occlusion = get_occlusion_mask_from_depth(depth, rgb_file_name == l_name, depth_tolerance=0.5)
            occlusion = dilate(occlusion, iterations=2)
            occlusion = torch.from_numpy(connected_component_filter(occlusion, 75)).unsqueeze(0)
            kernel = torch.tensor([[[[0, 0.15, 0.15], [0, 0.25, 0.15], [0, 0.15, 0.15]]]], dtype=torch.float32)
            conv_func = lambda x: F.conv2d(x.float(), kernel, stride=1, padding=1)
            for _ in range(4):
                occlusion = conv_func(occlusion)
                occlusion = torch.where(occlusion > 0.9, torch.ones_like(occlusion), torch.zeros_like(occlusion))
                occlusion = connected_component_filter(occlusion, 100)
                occlusion = dilate(occlusion, 3, 1)
                occlusion = torch.where(occlusion > 0.8, torch.ones_like(occlusion), torch.zeros_like(occlusion))

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            VF.to_pil_image(occlusion).save(mask_path)
        return kornia.io.load_image(mask_path, ImageLoadType.GRAY32, device="cpu")

    def get_depth(self, rgb_file_name, A=20, B=1.0):
        depth_path = self.get_file_path(self.depth_format_str, file_name=rgb_file_name)
        if not os.path.exists(depth_path):
            logging.error(f"Depth map does not exist: {depth_path}")
            raise FileNotFoundError(f"Depth map does not exist: {depth_path}")
        depth = np.array(Image.open(depth_path)) / 65535
        return B + A * (np.max(depth) - depth)

    def get_disparity(self, rgb_file_name):
        disp_path = self.disp_format_str.format(file_name=self.modify_file_name(rgb_file_name).replace(".png", ".exr"))
        if not os.path.exists(disp_path):
            raise FileNotFoundError(f"Disparity map does not exist: {disp_path}")
        return read_exr(disp_path).squeeze()

    def get_stereo_pair(self, file_name: str):
        # NOTE: Assumes file_name contains either "left" or "right" and the stereo pair would have otherwise the same name.
        if "left" in file_name:
            pair_name = file_name.replace("left", "right")
            l_name, r_name = file_name, pair_name
        elif "right" in file_name:
            pair_name = file_name.replace("right", "left")
            r_name, l_name = file_name, pair_name
        else:
            raise ValueError(f"{file_name} does not contain 'left' or 'right', cannot find stereo pair.")
        if not os.path.exists(os.path.join(self.root_dir, pair_name)):
            logging.info(f"No stereo pair found for {file_name}")
            return None
        return l_name, r_name

    def is_valid_data(self, file_name: str):
        if not self.get_stereo_pair(file_name):
            return False
        if not os.path.exists(
            self.get_file_path(self.depth_format_str, file_name)
        ):
            logging.debug(f"No depth map found for {file_name}.")
            return False
        l_name, r_name = self.get_stereo_pair(file_name)
        if not os.path.exists(self.get_file_path(self.depth_format_str, r_name)):
            logging.debug(f"No depth map found for {r_name}.")
            return False   
        return True

    def modify_file_name(self, file_name):
        """Converts image file name (e.g., 'xxx_frame_number.ext') to follow 'xxx.frame_number.ext' convention."""

        # HACK: This assumes that there is only one leading character before the capture group in frame_pattern.
        def path_replacer(match_obj):
            full_match = match_obj.group(0)
            return "." + full_match[1:]

        new_file_name = re.sub(self.frame_pattern, path_replacer, file_name)
        return new_file_name

    def get_file_path(self, format_str, file_name, ext=".png", **kwargs):
        if not format_str:
            return ""
        og_ext = file_name.split(".")[-1]
        file_path = format_str.format(
            file_name=self.modify_file_name(file_name), base_name=os.path.basename(file_name), **kwargs
        ).replace(f".{og_ext}", ext)
        return file_path

    def __getitem__(self, idx):
        shot, frame = self.index2shotframe(idx)
        _, file_L, file_R = self.get_item(shot, frame)

        # This is a temporal measure to give consistent left or right eye image from a certain shot.
        # TODO Ideally we shuold treat left and right eye data as two different shots in the dataset or sampler.
        rng = random.Random(self.shot_names[shot])
        file = file_L if rng.choice([0, 1]) == 0 else file_R
        image = kornia.io.load_image(os.path.join(self.root_dir, file), ImageLoadType.RGB32, device="cpu")
        # TODO Config the type of mask used
        mask = self.get_occlusion_mask_from_depth(file, shot=shot, frame=frame, w=image.shape[2], h=image.shape[1])
        occluded = image * (~mask.bool())
        return {
            "input_rgb": occluded,
            "og_input": occluded.clone(),
            "target_rgb": image,
            "occlusion_mask": mask,
            "og_occlusion_mask": mask.clone(),
            "og_target": image.clone(),
            "image_name": os.path.basename(file),
            "shot_name": self.shot_names[shot],
        }  # C, H, W
