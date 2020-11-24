import os
from random import shuffle

import numpy as np

from keras.preprocessing.image import img_to_array, load_img

IMG_EXTS = [".jpg", ".png"]


class GroundsLoader:
    """Generator that provides lots of images of ground coffee with
    data usable for regression, no nasty classification
    """

    __slots__ = ("_directory", "_batch_size", "_paths", "_target_size")

    def __init__(self, directory: str, batch_size: int, target_size: tuple):
        self._directory = directory
        self._batch_size = batch_size
        self._paths = []
        self._target_size = target_size
        for root, dirs, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() not in IMG_EXTS:
                    continue
                self._paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self._paths)

    def training_gen(self):
        while True:
            for batch in self.generator():
                yield batch

    def generator(self):
        # I kind of get it, but still hate the infinite generator
        total_size = len(self._paths)
        shuffle(self._paths)
        batch_start = 0
        batch_end = self._batch_size
        while batch_start < total_size:
            limit = min(batch_end, total_size)
            yield self.get_batch(batch_start, limit)
            batch_start += self._batch_size
            batch_end += self._batch_size

    def get_batch(self, start: int, end: int):
        files = self._paths[start:end]
        x = np.zeros((len(files), self._target_size[0], self._target_size[1], 3))
        y = np.zeros((len(files),))
        for i, path in enumerate(files):
            x[i] = img_to_array(load_img(path, target_size=self._target_size))
            y[i] = float(os.path.basename(os.path.dirname(path)))
        return x, y
