import os
from random import seed, choices
from typing import Optional, List, Callable
from collections import defaultdict

import numpy as np
from numpy.random import seed as np_seed

from PIL import Image

from keras.preprocessing.image import img_to_array, load_img

from optpresso.data.partition import find_test_paths

from tensorflow.random import set_seed

IMG_EXTS = [".jpg", ".png"]


def random_flip_transform(img):
    flip_choice = np.random.random()
    if flip_choice <= 0.25:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_choice <= 0.5:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if flip_choice <= 0.75:
        return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    return img


def random_zoom_transform(img, max_factor: float = 0.05):
    """
    img: Image
    max_factor: float - maximum amount of zoom, defaults to 0.1 or 10%
    """
    zoom_factor = 1 - (np.random.random() * max_factor)
    if zoom_factor < 1:
        width, height = img.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        dx = (width - new_width) // 2
        dy = (height - new_height) // 2
        img = img.crop((dx, dy, new_width + dx, new_height + dy))
    return img


def _identity_transform(img):
    return img

def compute_image_mean(loader) -> np.array:
    n = len(loader)
    mean = np.zeros(3)
    for x, _ in loader.generator():
        for y in x:
            mean += np.mean(y, axis=(0, 1)) / n
    return mean

class GroundsLoader:
    """Generator that provides lots of images of ground coffee with
    data usable for regression, no nasty classification
    """

    __slots__ = (
        "_directory",
        "_batch_size",
        "_paths",
        "_target_size",
        "_weights",
        "_scaling",
        "_transforms",
        "_mean_value",
    )

    def __init__(
        self,
        batch_size: int,
        target_size: tuple,
        directory: Optional[str] = None,
        paths: Optional[List[str]] = None,
        scaling: int = 1.0,
        transforms: Optional[List[Callable]] = None,
        mean_val: Optional[List] = None,
    ):
        self._directory = directory
        self._batch_size = batch_size
        self._paths = []
        self._target_size = target_size
        self._weights = None
        self._transforms = transforms
        if mean_val is None:
            self._mean_value = np.zeros(3)
        else:
            self._mean_value = np.asarray(mean_val)
        if self._transforms is None:
            self._transforms = [_identity_transform]
        if directory is None and paths is None:
            raise RuntimeError("Must provide directory or paths")
        self._scaling = scaling
        if directory is not None:
            for time, path in find_test_paths(directory):
                if "flip" in path:
                    continue
                self._paths.append((time * scaling, path))
        if paths is not None:
            for path in paths:
                try:
                    time = float(os.path.basename(os.path.dirname(path)))
                except Exception:
                    print("Skipping path", path)
                    continue
                self._paths.append((time * scaling, path))

    @property
    def weights(self):
        """
        Returns a numpy array indexed by integer time to the correspoding
        weights.
        """
        if self._weights is None:
            bins = defaultdict(int)
            max_time = 0
            for time, path in self._paths:
                bins[int(time)] += 1
                max_time = max(max_time, time)
            max_count = max(bins.values())

            weights = np.ones(int(max_time) + 1)
            # Aim for 1.0 for the highest count, with increasing weights with lower counts
            # TODO evaluate trying a non linear relation for the lower counts
            for x in bins.items():
                weights[x[0]] = 2.0 - (x[1] / max_count)
            self._weights = weights
        return self._weights

    def __len__(self):
        return len(self._paths)

    def training_gen(self):
        while True:
            # I kind of get it, but still hate the infinite generator
            for batch in self.generator():
                yield batch

    def weighted_training_gen(self):
        while True:
            for batch in self.weighted_generator():
                yield batch

    def _base_gen(self, meth):
        total_size = len(self._paths)
        np.random.shuffle(self._paths)
        batch_start = 0
        batch_end = self._batch_size
        while batch_start < total_size:
            yield meth(batch_start, min(batch_end, total_size))
            batch_start += self._batch_size
            batch_end += self._batch_size

    def generator(self):
        return self._base_gen(self.get_batch)

    def generator_with_paths(self):
        return self._base_gen(self.get_batch_with_paths)

    def weighted_generator(self):
        return self._base_gen(self.get_weighted_batch)

    def get_batch(self, start: int, end: int):
        files = self._paths[start:end]
        x = np.empty((len(files), self._target_size[0], self._target_size[1], 3))
        y = np.empty((len(files),))
        for i, (time, path) in enumerate(files):
            x[i] = img_to_array(
                load_img(path, target_size=(self._target_size[0], self._target_size[1]))
            ) - self._mean_value
            y[i] = time
        return x, y

    def get_batch_with_paths(self, start: int, end: int):
        return [x[1] for x in self._paths[start:end]], self.get_batch(start, end)

    def get_weighted_batch(self, start: int, end: int):
        x, y = self.get_batch(start, end)
        weights = np.empty(y.shape)
        for i, time in enumerate(y):
            weights[i] = self.weights[int(time)]
        return x, y, weights


def set_random_seed(seed_num: int):
    seed(seed_num)
    np_seed(seed_num)
    set_seed(seed_num)
