"""
The inspiration/code came from the following:

- https://github.com/rsyamil/cnn-regression
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
from typing import List, Any
from argparse import Namespace, ArgumentParser

import numpy as np

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from optpresso.utils import GroundsLoader
from optpresso.models.networks import MODEL_CONSTRUCTORS


def train(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(description="Train the Optpresso CNN model")
    parser.add_argument("directory")
    parser.add_argument("--validation-directory", default=None)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--height", default=255, type=int)
    parser.add_argument("--width", default=255, type=int)
    parser.add_argument(
        "--model-name", choices=list(MODEL_CONSTRUCTORS.keys()), default="optpresso"
    )
    parser.add_argument(
        "--output-path", default="model.h5", help="Output path of the model"
    )
    args = parser.parse_args(leftover)

    generator = GroundsLoader(
        args.directory, args.batch_size, (args.height, args.width)
    )
    validation = None
    if args.validation_directory:
        # Should rewrite the grounds loader into a Sequence class
        validation = GroundsLoader(
            args.validation_directory, args.batch_size, (args.height, args.width)
        )
        validation = validation.get_batch(0, len(validation))

    model = MODEL_CONSTRUCTORS[args.model_name]((args.height, args.width, 3))

    model.summary()

    model.fit(
        generator.training_gen(),
        epochs=args.epochs,
        steps_per_epoch=len(generator) // args.batch_size,
        batch_size=args.batch_size,
        callbacks=[
            # EarlyStopping(monitor="loss", min_delta=1.0, patience=500, mode="min"),
            # ModelCheckpoint("checkpoint.hf", monitor="loss", save_best_only=True),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=10, min_lr=0.00001
            ),
        ],
        validation_data=validation,
    )
    model.save(args.output_path)
