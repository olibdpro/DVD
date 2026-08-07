import random
import logging

from torch.utils.data import Sampler

from .shots_dataset_interface import ShotsDatasetInterface


class ClipSampler(Sampler):
    def __init__(
        self, dataset: ShotsDatasetInterface, clip_len: int = 16, stride: int | tuple[int, int] = (1, 3), rng_seed=42
    ):
        # FUTURE: Consider weighing the sampling by shot length
        """
        Args:
            dataset (ShotsDatasetInterface): source dataset
            clip_len (int, optional): number of frames in a clip. Defaults to 16.
            stride (int | tuple[int, int], optional): stride between two frames in the clip, either a fixed number or a range for random strides. Defaults to (1, 3).
        """
        if not (isinstance(dataset, ShotsDatasetInterface)):
            raise ValueError(
                f"Data source must implement {ShotsDatasetInterface.__name__} to use {self.__class__.__name__}"
            )
        self.dataset: ShotsDatasetInterface = dataset
        stride = (stride, stride) if isinstance(stride, int) else sorted(tuple(stride))
        self.min_stride, self.max_stride = stride
        self.clip_len = clip_len
        self.valid_shots = self.get_valid_shots()
    
        self._rng = random.Random(rng_seed)
        self._shots = self.get_shuffled_shots()
        self._cur_itr = 0
        self.batch_size = clip_len

    def get_shuffled_shots(self):
        self._shots_rng_state = self._rng.getstate()
        return self._rng.sample(self.valid_shots, len(self.valid_shots))

    def __iter__(self):
        while self._cur_itr < len(self._shots):
            shot = self._shots[self._cur_itr]
            shot_len = self.dataset.n_frames_in_shot(shot)
            ## Generate all intervals in one go, which is argubly cleaner but limits valid shot length
            # intervals = torch.randint(self.min_stride, self.max_stride+1, (self.clip_len,))
            # start_frame = shot_len-sum(intervals)
            # indices = start_frame+torch.cumsum(intervals, dim=0)
            # indices = [self.dataset.shotframe2index(shot, f) for f in indices]
            # yield indices
            frames = [self._rng.randint(0, shot_len - self.min_shot_len)]
            for f in range(self.clip_len - 1):
                # NOTE: For shorter shots this is likely going to end up uneven?
                frames.append(
                    min(
                        frames[-1] + self._rng.randint(self.min_stride, self.max_stride),
                        shot_len - self.min_stride * (self.clip_len - f),
                    )
                )
            self._cur_itr += 1
            self._rng_state = self._rng.getstate()
            indices = [self.dataset.shotframe2index(shot, f) for f in frames]
            logging.debug(f"[{self.__class__.__name__}]: shot {shot} - frames {frames} - indices {indices}")
            yield indices
        self._shots = self.get_shuffled_shots()
        self._cur_itr = 0

    def __len__(self):
        return len(self.valid_shots)

    @property
    def min_shot_len(self) -> int:
        return self.min_stride * self.clip_len

    def get_valid_shots(self):
        valid_shots = []
        for shot_idx in range(self.dataset.n_shots):
            if self.dataset.n_frames_in_shot(shot_idx) < self.min_shot_len:
                logging.debug(
                    f"Shot {shot_idx} in dataset has {self.dataset.n_frames_in_shot(shot_idx)} frames, which is shorter than the minimal length {self.min_shot_len}. Excluded."
                )  # FUTURE: Pad short shots instead?
            else:
                valid_shots.append(shot_idx)
        if len(valid_shots) < self.dataset.n_shots:
            logging.warning(f"{self.dataset.n_shots - len(valid_shots)} invalid shots were excluded from sampling.")
        logging.info(f"[{self.__class__.__name__}] found {len(valid_shots)} valid shots.")
        return valid_shots

    def state_dict(self):
        state_dict = {}
        state_dict["shots_rng_state"] = self._shots_rng_state
        state_dict["rng_state"] = getattr(self, "_rng_state", None)
        state_dict["cur_itr"] = self._cur_itr
        return state_dict

    def load_state_dict(self, state_dict):
        self._cur_itr = state_dict["cur_itr"]
        self._rng.setstate(state_dict["shots_rng_state"])
        self._shots = self.get_shuffled_shots()
        self._rng.setstate(state_dict["rng_state"])
