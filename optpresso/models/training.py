import os
import sys
from typing import List, Any, Optional, Dict
from argparse import Namespace, ArgumentParser

import numpy as np

from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateScheduler,
    Callback,
)
from tensorflow.keras.optimizers import Adam


from optpresso.utils import GroundsLoader, set_random_seed
from optpresso.data.partition import find_test_paths, k_fold_partition
from optpresso.models.networks import MODEL_CONSTRUCTORS
from optpresso.models.eval import graph_model
from optpresso.data.config import load_config
from optpresso.models.serialization import load_model
from optpresso.models.metrics import (
    psuedo_huber_loss,
)

import wandb
from wandb.keras import WandbCallback


class CycleWeightSaver(Callback):
    def __init__(
        self,
        monitor: str,
        cycles: int,
        num_epochs: int,
        mode: str = "min",
        save_prefix: str = "cycle-",
    ):
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
            print("No logs")
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
        return lr


def train_model(
    args,
    model,
    training,
    validation,
    learning_rate: float = 3e-4,
    callbacks: Optional[List[Any]] = None,
    fold: Optional[int] = None,
):

    validation_gen = None
    training_gen = training.to_tensorflow_dataset()
    if validation is not None:
        validation_gen = validation.to_tensorflow_dataset()
    wandb.config.update(
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "train_size": len(training),
            "validation_set": 0 if validation is None else len(validation),
            "learning_rate": learning_rate,
            "weighted": args.weighted,
            "target_size": [args.height, args.width],
            "finetuning": args.base_model is not None,
            "fold": fold,
        }
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=psuedo_huber_loss,
        metrics=["mse", "mae", psuedo_huber_loss],
    )
    fit_hist = model.fit(
        training_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=validation_gen,
    )
    output_path = args.output_path
    name, ext = os.path.splitext(output_path)
    if fold is not None:
        output_path = f"{name}-fold-{fold}{ext}"
    model.save(output_path)
    return fit_hist


def train(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(description="Train the Optpresso CNN model")
    parser.add_argument("directory")
    parser.add_argument(
        "--validation-dir", default=None, help="Directory containing validation set"
    )
    parser.add_argument(
        "--test-dir", default=None, help="Directory containing test set"
    )
    parser.add_argument(
        "--k-folds",
        default=None,
        type=int,
        help="Run K Folds on directory, not supported with --validation-directory/--test-directory flag",
    )
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=500, type=int)
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
        "--comp", action="store_true", help="Compare against model configured"
    )
    parser.add_argument(
        "--write", action="store_true", help="Write out loss graph to loss_graph.png"
    )
    parser.add_argument(
        "--model-name", choices=list(MODEL_CONSTRUCTORS.keys()), default="optpresso"
    )
    parser.add_argument(
        "--patience",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--output-path", default="model.h5", help="Output path of the model"
    )
    parser.add_argument(
        "--mode", choices=["annealing", "checkpoint"], default="checkpoint"
    )
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument(
        "--base-model", default=None, type=str, help="Model to train further against"
    )
    args = parser.parse_args(leftover)
    if (
        args.validation_dir is not None
        and args.test_dir is not None
        and args.k_folds is not None
    ):
        print("Can't provide K Folds and Validation or Test directory")
        sys.exit(1)
    if args.seed is not None:
        set_random_seed(args.seed)
    model_name, ext = os.path.splitext(args.output_path)

    group_id = None
    callbacks = []
    if args.k_folds is None:
        wandb.init(project="optpresso", entity="optpresso")
        callbacks.append(WandbCallback())

    if args.mode == "checkpoint":
        if not os.path.isdir(model_name):
            os.mkdir(model_name)
        callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(
                    model_name, model_name + "-{epoch}-{val_loss:.3f}.h5"
                ),
                mode="min",
                monitor="val_loss",
                save_best_only=True,
            ),
        )
    elif args.mode == "annealing":
        cycles = 5
        callbacks.extend(
            [
                LearningRateScheduler(CyclicCosineAnnealing(cycles, args.epochs)),
                CycleWeightSaver(
                    "val_loss",
                    cycles,
                    args.epochs,
                    mode="min",
                    save_prefix=f"{model_name}-cycle-",
                ),
            ]
        )
    if args.patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.001,
                patience=args.patience,
                mode="min",
                restore_best_weights=True,
            )
        )

    config = load_config()
    comp_model = None
    if config is not None and args.comp:
        comp_model = load_model(config.model, compile=False)

    if args.k_folds is None:
        generator = GroundsLoader(
            args.batch_size,
            (args.height, args.width),
            directory=args.directory,
            weighted=args.weighted,
        )
        if len(generator) <= 0:
            print(f"No files in directory {args.directory}")
            sys.exit(1)
        validation = None
        if args.validation_dir:
            validation = GroundsLoader(
                args.batch_size,
                (args.height, args.width),
                directory=args.validation_dir,
            )
        if args.base_model is not None:
            model = load_model(args.base_model, compile=False)
        else:
            model = MODEL_CONSTRUCTORS[args.model_name]((args.height, args.width, 3))
        train_model(args, model, generator, validation, callbacks=callbacks)
        if args.eval:
            comparison_set = validation
            graph_title = model_name + "-validation"
            if args.test_dir:
                comparison_set = GroundsLoader(
                    args.batch_size,
                    (args.height, args.width),
                    directory=args.test_dir,
                )
                graph_title = model_name + "-test"
            if comparison_set is not None:
                graph_model(graph_title, model, comparison_set, write=args.write)
                if comp_model is not None:
                    graph_model(
                        f"{graph_title}-comp",
                        comp_model,
                        comparison_set,
                        write=args.write,
                    )
    else:
        exp_id = wandb.util.generate_id()
        group_id = f"{exp_id} k folds"
        folds_dir = k_fold_partition(args.directory, folds=args.k_folds)
        fold_to_path = {}
        for i in range(args.k_folds):
            fold_to_path[i] = [
                x[1] for x in find_test_paths(os.path.join(folds_dir.name, str(i)))
            ]

        fold_min = []
        for i in range(args.k_folds):
            folds_callbacks = callbacks.copy()
            wandb.init(project="optpresso", entity="optpresso", group=group_id)
            folds_callbacks.append(WandbCallback())
            os.environ["WANDB_RUN_GROUP"] = f"{group_id} fold {i}"
            test_paths = []
            for key, paths in fold_to_path.items():
                if key == i:
                    continue
                test_paths.extend(paths)

            generator = GroundsLoader(
                args.batch_size,
                (args.height, args.width),
                paths=test_paths,
                weighted=args.weighted,
            )
            if len(generator) <= 0:
                print(f"No files in k-fold paths: {test_paths}")
                sys.exit(1)
            validation = GroundsLoader(
                args.batch_size,
                (args.height, args.width),
                paths=fold_to_path[i],
            )
            model = MODEL_CONSTRUCTORS[args.model_name]((args.height, args.width, 3))
            fit_hist = train_model(
                args, model, generator, validation, callbacks=folds_callbacks, fold=i
            )
            fold_min.append(min(fit_hist.history["val_loss"]))
            if args.eval:
                graph_model(
                    f"{model_name}-fold-{i}-test", model, validation, write=args.write
                )
                if comp_model is not None:
                    graph_model(
                        f"{model_name}-comp-fold-{i}-test",
                        comp_model,
                        validation,
                        write=args.write,
                    )
            clear_session()
            wandb.finish()

        print(
            "Average Validation Loss: {}, All: {}".format(
                np.array(fold_min).mean(), fold_min
            )
        )
