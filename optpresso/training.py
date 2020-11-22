"""
The inspiration/code came from the following:

- https://github.com/rsyamil/cnn-regression
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import os
from typing import List, Any
from argparse import Namespace, ArgumentParser

import numpy as np

from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from keras.preprocessing.image import img_to_array, load_img

IMG_EXTS = [".jpg", ".png"]


def train(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--batch-size", default=5, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args(leftover)

    generator = GroundsLoader(args.directory, args.batch_size, (255, 255))

    model = build_model((255, 255, 3))

    model.summary()

    model.fit(
        generator.generator(),
        epochs=args.epochs,
        steps_per_epoch=len(generator) // args.batch_size,
        shuffle=True,
        batch_size=args.batch_size,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=1.0, patience=500, mode="min"),
            ModelCheckpoint("checkpoint.hf", monitor="loss", save_best_only=True),
            ReduceLROnPlateau(monitor="loss", factor=0.2, patience=10, min_lr=0.00001),
        ],
    )
    model.save("junk-model.h5")
    plot_model(model, to_file="junk-model.png")


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

    def generator(self):
        # I kind of get it, but still hate the infinite generator
        total_size = len(self._paths)
        while True:
            batch_start = 0
            batch_end = self._batch_size
            while batch_start < total_size:
                limit = min(batch_end, total_size)
                yield self.get_batch(batch_start, limit)
                batch_start += self._batch_size
                batch_end += self._batch_size

    def get_batch(self, start: int, end: int):
        files = self._paths[start:end]
        x = np.zeros((len(files), self._target_size[0], self._target_size[0], 3))
        y = np.zeros((len(files),))
        for i, path in enumerate(files):
            x[i] = img_to_array(load_img(path, target_size=self._target_size))
            y[i] = float(os.path.basename(os.path.dirname(path)))
        return x, y


def build_model(shape: List[int], alpha: float = 0.3) -> List[Any]:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("linear"))
    # From regression example
    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="mse", metrics=["mse"])
    return model
