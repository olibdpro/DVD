import re
import bisect
import random
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Iterable
from functools import cached_property
from itertools import accumulate

from torch.utils.data import Subset, ConcatDataset


class ShotsDatasetInterface(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def n_shots(self) -> int:
        """Number of shots/sequences in this dataset."""
        raise NotImplementedError()

    @cached_property
    @abstractmethod
    def cum_n_frames_in_shot(self) -> list[int]:
        """List of cumulative frame numbers for the shots in this dataset."""
        raise NotImplementedError()

    @abstractmethod
    def n_frames_in_shot(self, shot: int) -> int:
        raise NotImplementedError()

    def shotframe2index(self, shot_idx: int, frame_idx: int):
        return (self.cum_n_frames_in_shot[shot_idx - 1] if shot_idx > 0 else 0) + frame_idx


class BaseShotsDataset(ShotsDatasetInterface, ABC):
    """
    Interface for datasets that consist of multiple shots/sequences.
    Provides functionality to convert between total dataset index and shot index + frame index.
    """

    def __init__(self, all_files_paths, shot_pattern: str, frame_pattern: str, **kwargs):
        super().__init__(**kwargs)
        self.shot_pattern = re.compile(shot_pattern)
        self.frame_pattern = re.compile(frame_pattern)
        self.lookup = self.build_lookup(all_files_paths)
        logging.info(f"[{self.__class__.__name__}] loaded {self.n_shots} shots.")
        logging.debug(
            "Shots index, name, length:\n"
            + "\n".join(
                [
                    f"    {shot_idx} {self.shot_names[shot_idx]}: {self.n_frames_in_shot(shot_idx)}"
                    for shot_idx in range(self.n_shots)
                ]
            )
        )

    @abstractmethod
    def build_lookup(self, all_files_paths: list[str]) -> dict:
        """Build the shot-frame lookup table.
        Returns:
            Dict[str[list]]: the lookup table with shot names as keys, list of per frame data as values.
        """
        raise NotImplementedError()

    def parse_file_path(self, file_path: str) -> Tuple[str, int]:
        """Given the file path as a string, try to extract the shot name and frame index using regex matching.

        Returns:
            tuple[str, int]: shot name, frame index
        """
        try:
            shot_name = self.shot_pattern.search(file_path).group(1)
        except Exception as e:
            logging.error(
                f"Error when trying to extract shot name and index using pattern [{self.shot_pattern.pattern}] from file: {file_path}",
            )
            raise e

        try:
            frame_index = int(self.frame_pattern.search(file_path).group(1))
        except Exception as e:
            logging.error(
                f"Error when trying to extract frame index using pattern [{self.frame_pattern.pattern}] from file: {file_path}"
            )
            raise e

        return shot_name, frame_index

    def index2shotframe(self, index: int) -> tuple[int, int]:
        shot_index = bisect.bisect_right(self.cum_n_frames_in_shot, index)
        frame_index = index - (self.cum_n_frames_in_shot[shot_index - 1] if shot_index > 0 else 0)
        return shot_index, frame_index

    @cached_property
    def shot_names(self) -> list[str]:
        return sorted(list(self.lookup.keys()))

    @property
    def n_shots(self) -> int:
        return len(self.shot_names)

    @cached_property
    def cum_n_frames_in_shot(self) -> list[int]:
        return list(accumulate([self.n_frames_in_shot(shot) for shot in self.shot_names]))

    def n_frames_in_shot(self, shot: str | int) -> int:
        if isinstance(shot, int):
            shot = self.shot_names[shot]
        return len(self.lookup[shot])

    def get_item(self, shot: str | int, frame_idx: int):
        if isinstance(shot, int):
            shot = self.shot_names[shot]
        return self.lookup[shot][frame_idx]

    def __len__(self):
        return sum([len(v) for v in self.lookup.values()])


class ConcatShotsDataset(ShotsDatasetInterface, ConcatDataset):
    # TODO: This should also be an implementation of ShotsDatasetInterface, but need to change the interface definition
    def __init__(self, datasets: Iterable[BaseShotsDataset]):
        super().__init__(datasets)

    @property
    def n_shots(self) -> int:
        return sum(dataset.n_shots for dataset in self.datasets)

    @property
    def cum_n_shots_in_dataset(self) -> int:
        return list(accumulate([dataset.n_shots for dataset in self.datasets]))

    @cached_property
    def cum_n_frames_in_shot(self) -> list[int]:
        return list(
            accumulate([dataset.n_frames_in_shot(shot) for dataset in self.datasets for shot in dataset.shot_names])
        )

    def n_frames_in_shot(self, shot_idx: int) -> int:
        dataset_idx = self.shot_in_dataset(shot_idx)
        shot_idx = shot_idx - (self.cum_n_shots_in_dataset[dataset_idx - 1] if dataset_idx > 0 else 0)
        return self.datasets[dataset_idx].n_frames_in_shot(shot_idx)

    def shot_in_dataset(self, shot_idx):
        dataset_idx = bisect.bisect_right(self.cum_n_shots_in_dataset, shot_idx)
        return dataset_idx

    def index2shotframe(self, index: int):
        shot_index = bisect.bisect_right(self.cum_n_frames_in_shot, index)
        frame_index = index - (self.cum_n_frames_in_shot[shot_index - 1] if shot_index > 0 else 0)
        return shot_index, frame_index
    
    @cached_property
    def shot_names(self):
        return [shot_name for dataset in self.datasets for shot_name in dataset.shot_names]


class ShotsSubset(ShotsDatasetInterface, Subset):
    """A Subset implementation for ShotsDatasetInterface datasets."""

    def __init__(self, dataset: ShotsDatasetInterface, indices: list[int]):
        super().__init__(dataset, indices)
        self.sub_shot_indices = sorted(list(set([self.dataset.index2shotframe(i)[0] for i in indices])))
        logging.info(f"[{self.__class__.__name__}] created subset with {self.n_shots} shots: {self.sub_shot_indices}")

    @property
    def n_shots(self):
        return len(self.sub_shot_indices)

    @cached_property
    def cum_n_frames_in_shot(self) -> list[int]:
        return list(accumulate([self.n_frames_in_shot(shot) for shot in range(self.n_shots)]))

    def n_frames_in_shot(self, sub_shot_idx):
        return self.dataset.n_frames_in_shot(self.sub_shot_indices[sub_shot_idx])
   
    @cached_property
    def shot_names(self):
        return [self.dataset.shot_names[shot_idx] for shot_idx in self.sub_shot_indices]


def shot_wise_split(dataset: ShotsDatasetInterface, val_ratio=0.1) -> Tuple[Subset, Subset]:
    """Randomly split a ShotsDatasetInterface into train and validation subsets based on shots.

    Args:
        dataset (ShotsDatasetInterface):
        val_ratio (float, optional): Ratio of shots that should be in the validataion subset. Defaults to 0.1.

    Returns:
        subsets (Tuple[ShotsSubset, ShotsSubset]): train_subset, val_subset
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    n_shots = dataset.n_shots
    n_val_shots = int(n_shots * val_ratio)
    n_train_shots = n_shots - n_val_shots
    all_shot_idx = list(range(n_shots))
    random.shuffle(all_shot_idx)
    train_indices = []
    for shot_idx in sorted(all_shot_idx[:n_train_shots]):
        n_frames = dataset.n_frames_in_shot(shot_idx)
        train_indices.extend([dataset.shotframe2index(shot_idx, frame_idx) for frame_idx in range(n_frames)])

    val_indices = []
    for shot_idx in sorted(all_shot_idx[n_train_shots:]):
        n_frames = dataset.n_frames_in_shot(shot_idx)
        val_indices.extend([dataset.shotframe2index(shot_idx, frame_idx) for frame_idx in range(n_frames)])

    train_subset = ShotsSubset(dataset, train_indices)
    val_subset = ShotsSubset(dataset, val_indices)

    return train_subset, val_subset
