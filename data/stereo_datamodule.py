import logging
from functools import partial

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.shots_dataset_interface import shot_wise_split

from .transforms import TransformWrapper
from .clip_sampler import ClipSampler
from .stereo_dataset import StereoDataset
from .shots_dataset_interface import ConcatShotsDataset


class StereoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_datasets: list[StereoDataset],
        val_datasets: list[StereoDataset],
        sampler: partial[ClipSampler],
        train_transforms: list[TransformWrapper] = None,
        val_transforms: list[TransformWrapper] = None,
        **kwargs,
    ):
        super().__init__()
        self.train = ConcatShotsDataset(train_datasets)
        if val_datasets is not None:
            self.val = ConcatShotsDataset(val_datasets)
        else:
            self.train, self.val = shot_wise_split(self.train, kwargs.get("val_ratio", 0.1))
        logging.info(
            f"[{self.__class__.__name__}] DataModule initialized with {len(train_datasets)} train datasets and {len(val_datasets) if val_datasets is not None else 0} val datasets."
        )
        logging.info(f"Train split size: {len(self.train)} samples.")
        logging.info(f"Validation split size: {len(self.val)} samples.")
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.sampler = sampler

        self._train_loader = None
        self._val_loader = None

    def train_dataloader(self):
        if self._train_loader is None:
            self._train_loader = DataLoader(
                self.train,
                batch_sampler=self.sampler(self.train),
                collate_fn=partial(self.collate_fn, transforms=self.train_transforms),
            )
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = DataLoader(
                self.val,
                batch_sampler=self.sampler(self.val),
                collate_fn=partial(self.collate_fn, transforms=self.val_transforms),
            )
        return self._val_loader

    def test_dataloader(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch, transforms):
        collated = torch.utils.data.default_collate(batch)
        if transforms:
            for transform in transforms:
                collated = transform(collated)
        return collated

    def state_dict(self):
        state_dict = {}
        state_dict["train_sampler"] = self.train_dataloader().batch_sampler.state_dict()
        state_dict["val_sampler"] = self.val_dataloader().batch_sampler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.train_dataloader().batch_sampler.load_state_dict(state_dict["train_sampler"])
        self.val_dataloader().batch_sampler.load_state_dict(state_dict["val_sampler"])
