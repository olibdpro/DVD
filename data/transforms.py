import random
import logging
from typing import Literal, Callable

from kornia.augmentation import RandomPlanckianJitter
from kornia.augmentation import random_generator as rg
from torchvision.transforms import ColorJitter
from kornia.augmentation._2d.intensity.planckian_jitter import get_planckian_coeffs


class TransformWrapper:
    def __init__(self, targets: list[str], transforms: list[Callable]):
        self.targets = targets
        self.transforms = transforms

    def __call__(self, data_dict):
        for k in self.targets:
            if k not in data_dict.keys():
                raise KeyError(
                    f"{self.__class__.__name__}: Target <{k}> does not exist in the data dict. Avaliable keys: {data_dict.keys()}"
                )
            else:
                for transform in self.transforms:
                    data_dict[k] = transform(data_dict[k])
        return data_dict


class PersistentColorJitter(ColorJitter):
    """
    This class applies random ColorJitter as the base class does but keeps the random parameters persistent for <persistent_for> number of calls.
    This is to simply the transform configuration (as we may need to apply the same color jitter to both input and target).
    """

    def __init__(self, persistent_for: int = 1, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.persistent_for = persistent_for
        self._cur_counter = 0
        self._params = None

    def get_params(self, brightness, contrast, saturation, hue):
        if self._cur_counter <= 0:
            self._params = super().get_params(brightness, contrast, saturation, hue)
            self._cur_counter = self.persistent_for - 1
        else:
            self._cur_counter -= 1
        return self._params


class PersistentPlanckianJitter:
    def __init__(
        self, persistent_for: int = 1, prob: float = 1.0, mode: Literal["blackbody", "CIED"] = "blackbody", rng_seed=42
    ):
        if prob != 1.0 and persistent_for > 1:
            # FIXME
            logging.warning(
                f"Probability of applying Planckian jitter is set to {prob}, this is likely not what you want: When probability of planckian jitter is not 1, it maybe applied to some target data but not all (e.g., only applied to input but not ground truth)"
            )
        self._all_planckian_coefficients = get_planckian_coeffs("blackbody").cpu()
        self.select_from = list(range(0, 25))
        self.planckian_jitter_transform = RandomPlanckianJitter(
            mode=mode, p=prob, keepdim=True, select_from=self.select_from, same_on_batch=True
        )
        self.persistent_for: int = persistent_for
        self._cur_counter: int = 0
        self._rng: random.Random = random.Random(rng_seed)
        # This transform uses its own rng to select one set of coeff at a time so that the same jitter can be applied to multiple targets
        # (e.g., input and target)
        # Therefore for the planckian_jitter's internal param generator there is only one set of avaliable coeff at any time.
        self.planckian_jitter_transform._param_generator = rg.PlanckianJitterGenerator([0.0, 1.0])

    def __call__(self, input):
        if self._cur_counter <= 0:
            selected_coeff = [self._rng.choice(self.select_from)]
            self.planckian_jitter_transform.pl = self._all_planckian_coefficients[selected_coeff]
            self._cur_counter = self.persistent_for - 1
        else:
            self._cur_counter -= 1
        return self.planckian_jitter_transform(input)
