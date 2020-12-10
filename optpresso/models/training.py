"""
The inspiration/code came from the following:

- https://github.com/rsyamil/cnn-regression
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import os
import sys
import math
from typing import List, Any
from argparse import Namespace, ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from optpresso.utils import GroundsLoader
from optpresso.data.partition import find_test_paths, k_fold_partition
from optpresso.models.networks import MODEL_CONSTRUCTORS


def train(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(description="Train the Optpresso CNN model")
    parser.add_argument("directory")
    parser.add_argument("--validation-directory", default=None)
    parser.add_argument("--k-folds", default=None, type=int, help="Run K Folds on directory, not supported with --validation-directory flag")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--height", default=240, type=int)
    parser.add_argument("--width", default=320, type=int)
    parser.add_argument(
        "--write", action="store_true", help="Write out loss graph to loss_graph.png"
    )
    parser.add_argument(
        "--model-name", choices=list(MODEL_CONSTRUCTORS.keys()), default="optpresso"
    )
    parser.add_argument(
        "--output-path", default="model.h5", help="Output path of the model"
    )
    args = parser.parse_args(leftover)
    if args.validation_directory is not None and args.k_folds is not None:
        print("Can't provide K Folds and Validation directory")
        sys.exit(1)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=25,
            mode="min",
            restore_best_weights=True,
        ),
        # ModelCheckpoint("checkpoint", monitor="val_loss", save_best_only=True),
        # ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-6),
    ]

    if args.k_folds is None:    
        generator = GroundsLoader(
            args.batch_size, (args.height, args.width), directory=args.directory,
        )
        if len(generator) <= 0:
            print(f"No files in directory {args.directory}")
            sys.exit(1)
        validation = None
        if args.validation_directory:
            # Should rewrite the grounds loader into a Sequence class
            validation = GroundsLoader(
                args.batch_size, (args.height, args.width), directory=args.validation_directory
            )
            validation = validation.get_batch(0, len(validation))

        model = MODEL_CONSTRUCTORS[args.model_name]((args.height, args.width, 3))

        model.summary()

        fit_hist = model.fit(
            generator.training_gen(),
            epochs=args.epochs,
            steps_per_epoch=int(math.ceil(len(generator) / args.batch_size)),
            batch_size=args.batch_size,
            callbacks=callbacks,
            validation_data=validation,
        )
        model.save(args.output_path)
        x = np.linspace(
            0, len(fit_hist.history["val_loss"]), len(fit_hist.history["val_loss"])
        )
        plt.plot(x, fit_hist.history["val_loss"], label="Validation Loss")
        plt.plot(x, fit_hist.history["loss"], label="Training Loss")
        plt.yscale("log")
        plt.legend()
        if args.write:
            plt.savefig(f"{args.model_name}_loss_graph.png")
        else:
            plt.show()
    else:
        folds_dir = k_fold_partition(args.directory, folds=args.k_folds)
        fold_to_path = {}
        for i in range(args.k_folds):
            fold_to_path[i] = [x[1] for x in find_test_paths(os.path.join(folds_dir.name, str(i)))]
        fold_min = []
        for i in range(args.k_folds):
            validation_paths = fold_to_path[i]
            test_paths = []
            for key, paths in fold_to_path.items():
                if key == i:
                    continue
                test_paths.extend(paths)

            generator = GroundsLoader(
                args.batch_size, (args.height, args.width), paths=test_paths,
            )
            if len(generator) <= 0:
                print(f"No files in directory {args.directory}")
                sys.exit(1)
            # Should rewrite the grounds loader into a Sequence class
            validation = GroundsLoader(
                args.batch_size, (args.height, args.width), paths=validation_paths,
            )
            validation = validation.get_batch(0, len(validation))

            model = MODEL_CONSTRUCTORS[args.model_name]((args.height, args.width, 3))

            fit_hist = model.fit(
                generator.training_gen(),
                epochs=args.epochs,
                steps_per_epoch=int(math.ceil(len(generator) / args.batch_size)),
                batch_size=args.batch_size,
                callbacks=callbacks,
                validation_data=validation,
            )
            fold_min.append(min(fit_hist.history["val_loss"]))
            model.save(f"fold-{i}-{args.output_path}")
            x = np.linspace(
                0, len(fit_hist.history["val_loss"]), len(fit_hist.history["val_loss"])
            )
            plt.plot(x, fit_hist.history["val_loss"], label=f"Validation Loss: Fold {i}")
            plt.plot(x, fit_hist.history["loss"], label=f"Training Loss: Fold {i}")
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"{args.model_name}_loss_graph-with-folds.png")
        print("Average Validation Loss: {}".format(np.array(fold_min).mean()))