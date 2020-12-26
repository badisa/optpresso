import os
import sys
import math
from typing import List, Any, Optional, Dict
from argparse import Namespace, ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, Callback

from optpresso.utils import GroundsLoader, set_random_seed
from optpresso.data.partition import find_test_paths, k_fold_partition
from optpresso.models.networks import MODEL_CONSTRUCTORS
from optpresso.models.eval import graph_model
from optpresso.data.config import load_config


class CycleWeightSaver(Callback):

    def __init__(self, monitor: str, cycles: int, num_epochs: int, mode: str = "min", save_prefix: str = "cycle-"):
        super().__init__()
        self.monitor = monitor
        self.cycles = cycles
        self.num_epochs = num_epochs
        if mode not in ["min", "max"]:
            raise ValueError(f"Invalid mode: {mode}")
        if mode == "min":
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
        self.save_prefix = save_prefix
        self.best_weights = [None for _ in range(self.cycles)]
        self.cycle_bests = [None for _ in range(self.cycles)]

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            print(f"No logs for epoch {epoch}")
            return
        if self.monitor not in logs:
            print(f"Unable to find metric {self.monitor}")
            return
        cycle = epoch // (self.num_epochs // self.cycles)
        cur_cy_weight = self.cycle_bests[cycle]
        mon_val = logs[self.monitor]
        if self.best_weights[cycle] is None or cur_cy_weight is None:
            self.best_weights[cycle] = self.model.get_weights()
            self.cycle_bests[cycle] = mon_val
        elif self.monitor_op(mon_val, cur_cy_weight):
            self.best_weights[cycle] = self.model.get_weights()
            self.cycle_bests[cycle] = mon_val

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        if logs is None:
            print(f"No logs for epoch {epoch}")
            return
        if self.monitor not in logs:
            print(f"Unable to find metric {self.monitor}")
            return
        best_cycle = 0
        for i in range(self.cycles):
            self.model.set_weights(self.best_weights[i])
            self.model.save(f"{self.save_prefix}{i}.h5")
            if self.monitor_op(self.cycle_bests[i], self.cycle_bests[best_cycle]):
                best_cycle = i
        # Also save the model with only the very best weights
        self.model.set_weights(self.best_weights[best_cycle])




class PolynomialDecay:

    __slots__ = ("num_epochs", "learning_rate", "power")

    def __init__(
        self, num_epochs: int = 100, learning_rate: float = 1e-3, power: float = 1.0
    ):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.power = power

    def __call__(self, epoch: int) -> float:
        decay = (1 - (epoch / float(self.num_epochs))) ** self.power
        return float(self.learning_rate * decay)


class CyclicCosineAnnealing:
    """Cycle the training rate to attempt to find multiple minima. Intended
    to be used to build ensembles, seems like it could also be used just to find
    the lowest minima.

    Comes from page 4 of https://arxiv.org/pdf/1704.00109.pdf
    """

    __slots__ = ("num_cycles", "epoches", "initial_rate")

    def __init__(self, num_cycles: int, epoches: int, learning_rate: float = 0.001):
        self.num_cycles = num_cycles
        self.epoches = epoches
        self.initial_rate = learning_rate

    def __call__(self, epoch: int) -> float:
        t_d_m = self.epoches / self.num_cycles
        numerator = np.pi * (epoch % t_d_m)
        lr = (self.initial_rate / 2) * (np.cos(numerator / t_d_m) + 1)
        print("New Learning Rate", lr)
        return lr


def train_model(
    args,
    model,
    training,
    validation,
    callbacks: Optional[List[Any]] = None,
    fold: Optional[int] = None,
):
    validation_batch = None
    if args.weighted:
        training_gen = training.weighted_training_gen()
        if validation is not None:
            validation_batch = validation.get_weighted_batch(0, len(validation))
    else:
        training_gen = training.training_gen()
        if validation is not None:
            validation_batch = validation.get_batch(0, len(validation))
    fit_hist = model.fit(
        training_gen,
        epochs=args.epochs,
        steps_per_epoch=int(math.ceil(len(training) / args.batch_size)),
        batch_size=args.batch_size,
        callbacks=callbacks,
        validation_data=validation_batch,
    )
    output_path = args.output_path
    name, ext = os.path.splitext(output_path)
    if fold is not None:
        output_path = f"{name}-fold-{fold}{ext}"
    model.save(output_path)
    x = np.linspace(
        0, len(fit_hist.history["val_loss"]), len(fit_hist.history["val_loss"])
    )
    val_label = "Validation Loss"
    train_label = "Training Loss"
    if fold is not None:
        val_label += f" Fold {fold}"
        train_label += f" Fold {fold}"
    plt.plot(x, fit_hist.history["val_loss"], label=val_label)
    plt.plot(x, fit_hist.history["loss"], label=train_label)
    plt.yscale("log")
    plt.legend()
    if args.write:
        if fold is None:
            plt.savefig(f"{name}_loss_graph.png")
        else:
            plt.savefig(f"{name}_loss_graph_fold_{fold}.png")
    else:
        plt.show()
    plt.close()
    if args.eval:
        graph_model(output_path, model, validation, write=args.write)
    return fit_hist


def train(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(description="Train the Optpresso CNN model")
    parser.add_argument("directory")
    parser.add_argument("--validation-directory", default=None)
    parser.add_argument(
        "--k-folds",
        default=None,
        type=int,
        help="Run K Folds on directory, not supported with --validation-directory flag",
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--height", default=240, type=int)
    parser.add_argument("--width", default=320, type=int)
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use sample weights to aim for better fit for tail ends of data",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Generate evaluation graphs for the validation data",
    )
    parser.add_argument(
        "--write", action="store_true", help="Write out loss graph to loss_graph.png"
    )
    parser.add_argument(
        "--model-name", choices=list(MODEL_CONSTRUCTORS.keys()), default="optpresso"
    )
    parser.add_argument(
        "--output-path", default="model.h5", help="Output path of the model"
    )
    parser.add_argument("--mode", choices=["patience", "annealing"], default="patience")
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args(leftover)
    if args.validation_directory is not None and args.k_folds is not None:
        print("Can't provide K Folds and Validation directory")
        sys.exit(1)
    if args.seed is not None:
        set_random_seed(args.seed)
    model_name, ext = os.path.splitext(args.output_path)
    callbacks = []
    if args.mode == "patience":
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=1.0,
                patience=10,
                mode="min",
                restore_best_weights=True,
            )
        )
    elif args.mode == "annealing":
        cycles = 5
        callbacks.extend(
            [
                LearningRateScheduler(CyclicCosineAnnealing(cycles, args.epochs)),
                CycleWeightSaver("val_loss", cycles, args.epochs, mode="min", save_prefix=f"{model_name}-cycle-"),
            ]
        )

    config = load_config()
    comp_model = None
    if config is not None:
        comp_model = load_model(config.model)

    if args.k_folds is None:
        generator = GroundsLoader(
            args.batch_size,
            (args.height, args.width),
            directory=args.directory,
        )
        if len(generator) <= 0:
            print(f"No files in directory {args.directory}")
            sys.exit(1)
        validation = None
        if args.validation_directory:
            # Should rewrite the grounds loader into a Sequence class
            validation = GroundsLoader(
                args.batch_size,
                (args.height, args.width),
                directory=args.validation_directory,
            )
        model = MODEL_CONSTRUCTORS[args.model_name]((args.height, args.width, 3))
        train_model(args, model, generator, validation, callbacks=callbacks)
        if comp_model is not None and args.eval and validation is not None:
            graph_model(f"comp-{model_name}", comp_model, validation, write=args.write)
    else:
        folds_dir = k_fold_partition(args.directory, folds=args.k_folds)
        fold_to_path = {}
        for i in range(args.k_folds):
            fold_to_path[i] = [
                x[1] for x in find_test_paths(os.path.join(folds_dir.name, str(i)))
            ]
        fold_min = []
        for i in range(args.k_folds):
            validation_paths = fold_to_path[i]
            test_paths = []
            for key, paths in fold_to_path.items():
                if key == i:
                    continue
                test_paths.extend(paths)

            generator = GroundsLoader(
                args.batch_size,
                (args.height, args.width),
                paths=test_paths,
            )
            if len(generator) <= 0:
                print(f"No files in directory {args.directory}")
                sys.exit(1)
            # Should rewrite the grounds loader into a Sequence class
            validation = GroundsLoader(
                args.batch_size,
                (args.height, args.width),
                paths=validation_paths,
            )

            model = MODEL_CONSTRUCTORS[args.model_name]((args.height, args.width, 3))
            fit_hist = train_model(
                args, model, generator, validation, callbacks=callbacks, fold=i
            )
            fold_min.append(min(fit_hist.history["val_loss"]))
            if comp_model is not None and args.eval:
                graph_model(
                    f"comp-{model_name}-fold-{i}", comp_model, validation, write=args.write
                )
        print(
            "Average Validation Loss: {}, All: {}".format(
                np.array(fold_min).mean(), fold_min
            )
        )
