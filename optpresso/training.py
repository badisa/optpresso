"""
The inspiration/code came from the following:

- https://github.com/rsyamil/cnn-regression
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import os
from typing import List, Any
from argparse import Namespace, ArgumentParser

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


def train(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--batch_size", default=5, type=int)
    args = parser.parse_args(leftover)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1,
        rotation_range=90,
    )

    # Lots of work around if this is classifying things correctly or if i
    # need to roll my own data loader
    labels = [int(x) for x in sorted(os.listdir(args.directory)) if os.path.isdir(x)]

    generator = train_datagen.flow_from_directory(
        args.directory,
        target_size=(255, 255),
        class_mode="binary",
        classes=labels,
        batch_size=args.batch_size
    )
    model = build_model((255, 255, 3))

    model.summary()
    plot_model(model, to_file="junk-model.png")

    model.fit(
        generator,
        epochs=200,
        shuffle=True,
        batch_size=args.batch_size,
    )
    model.save("junk-model.h5")


def build_model(shape: List[int], alpha: float = 0.3) -> List[Any]:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # From regression example
    # opt = keras.optimizers.Adam(lr=1e-4)
    # model.compile(optimizer=opt, loss="mse", metrics=["mse"])
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    return model
