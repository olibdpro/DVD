from torch.utils.data import Sampler

from .shots_dataset_interface import ShotsDatasetInterface


class ShotSampler(Sampler):
    """A batch sampler that returns each entire shot in the source ShotsDataset as a batch. Mostly for inference."""

    def __init__(self, dataset: ShotsDatasetInterface, max_clip_len=None, start_clip_idx=0, shot_names=[]):
        """
        Args:
            dataset (ShotsDatasetInterface): source dataset
            shot_names (list): list of shot names to include in the sampler
        """
        if not (isinstance(dataset, ShotsDatasetInterface)):
            raise ValueError(
                f"Data source must implement {ShotsDatasetInterface.__name__} to use {self.__class__.__name__}"
            )
        self.dataset: ShotsDatasetInterface = dataset
        self.max_clip_len = max_clip_len if max_clip_len is not None else float("inf")
        self.start_clip_idx = start_clip_idx
        self.shot_names = shot_names

    def __iter__(self):
        for shot in range(self.dataset.n_shots):
            if self.shot_names and self.dataset.shot_names[shot] not in self.shot_names:
                continue
            offset = 0 if shot == 0 else self.dataset.cum_n_frames_in_shot[shot - 1]
            clip_len = min(self.max_clip_len, self.dataset.n_frames_in_shot(shot) - self.start_clip_idx)
            if clip_len <= 0:
                # TODO Error handling
                raise ValueError(
                    f"start_clip_idx {self.start_clip_idx} is out of bounds for shot {shot} with {self.dataset.n_frames_in_shot(shot)} frames"
                )
            idx = [offset + self.start_clip_idx + i for i in range(clip_len)]
            yield idx

    def __len__(self):
        return self.dataset.n_shots if not self.shot_names else len(self.shot_names)
