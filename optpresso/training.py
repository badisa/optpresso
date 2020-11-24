"""
The inspiration/code came from the following:

- https://github.com/rsyamil/cnn-regression
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
from typing import List, Any
from argparse import Namespace, ArgumentParser

import numpy as np

from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU

from optpresso.utils import GroundsLoader


def train(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(description="Train the Optpresso CNN model")
    parser.add_argument("directory")
    parser.add_argument("--validation-directory", default=None)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--height", default=255, type=int)
    parser.add_argument("--width", default=255, type=int)
    parser.add_argument("--model-path", default="model.h5", help="Output path of the model")
    args = parser.parse_args(leftover)

    generator = GroundsLoader(args.directory, args.batch_size, (args.height, args.width))
    validation = None
    if args.validation_directory:
        # Should rewrite the grounds loader into a Sequence class
        validation = GroundsLoader(args.validation_directory, args.batch_size, (args.height, args.width))
        validation = np.array(list(validation.generator(0, len(validation))))

    model = build_model((args.height, args.width, 3))

    model.summary()

    model.fit(
        generator.training_gen(),
        epochs=args.epochs,
        steps_per_epoch=len(generator) // args.batch_size,
        batch_size=args.batch_size,
        callbacks=[
            EarlyStopping(monitor="loss", min_delta=1.0, patience=500, mode="min"),
            ModelCheckpoint("checkpoint.hf", monitor="loss", save_best_only=True),
            ReduceLROnPlateau(monitor="loss", factor=0.2, patience=10, min_lr=0.00001),
        ],
        validation_data=validation,
    )
    model.save(args.model_path)


def build_model(shape: List[int], alpha: float = 0.3) -> List[Any]:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(0.5))

    # model.add(Dense(64))
    # model.add(LeakyReLU(alpha=alpha))

    # Dense layer of size 1 with linear activation to get that glorious regression
    model.add(Dense(1))
    model.add(Activation("linear"))
    # A low learning rate seems better, at least when data was ~100 images
    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="mse", metrics=["mse"])
    return model
