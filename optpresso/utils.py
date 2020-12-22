import os
from random import shuffle
from typing import Optional, List

import numpy as np

from keras.preprocessing.image import img_to_array, load_img

from optpresso.data.partition import find_test_paths

IMG_EXTS = [".jpg", ".png"]


class GroundsLoader:
    """Generator that provides lots of images of ground coffee with
    data usable for regression, no nasty classification
    """

    __slots__ = ("_directory", "_batch_size", "_paths", "_target_size")

    def __init__(self, batch_size: int, target_size: tuple, directory: Optional[str] = None, paths: Optional[List[str]] = None):
        self._directory = directory
        self._batch_size = batch_size
        self._paths = []
        self._target_size = target_size
        if directory is None and paths is None:
            raise RuntimeError("Must provide directory or paths")
        if directory is not None:
            for time, path in find_test_paths(directory):
                self._paths.append((time, path))
        if paths is not None:
            for path in paths:
                try:
                    time = float(os.path.basename(os.path.dirname(path)))
                except Exception:
                    print("Skipping path", path)
                    continue
                self._paths.append((time, path))

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
        shuffle(self._paths)
        batch_start = 0
        batch_end = self._batch_size
        while batch_start < total_size:
            limit = min(batch_end, total_size)
            yield meth(batch_start, limit)
            batch_start += self._batch_size
            batch_end += self._batch_size

    def generator(self):
        return self._base_gen(self.get_batch)

    def weighted_generator(self):
        return self._base_gen(self.get_weighted_batch)

    def get_batch(self, start: int, end: int):
        files = self._paths[start:end]
        x = np.zeros((len(files), self._target_size[0], self._target_size[1], 3))
        y = np.zeros((len(files),))
        i = 0
        for time, path in files:
            x[i] = img_to_array(load_img(path, target_size=self._target_size))
            y[i] = time
            i += 1
        return x, y

    def get_weighted_batch(self, start: int, end: int):
        x, y = self.get_batch(start, end)
        weights = np.ones(y.shape)
        for i, time in enumerate(y):
            if time <= 15.0:
                weights[i] = 1.5
            elif time <= 25.0 or time >= 45.0:
                weights[i] = 1.25
            elif time >= 28 and time <= 36:
                weights[i] = 0.8
        return x, y, weights