import os
import sys
from typing import List
from argparse import ArgumentParser, Namespace

from keras.models import load_model

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor

from optpresso.utils import GroundsLoader
from optpresso.data.config import load_config


def graph_model(model_path, model, loader, write: bool = False):
    i = 0
    y_predict = []
    y_test = []
    for x, y in loader.generator():
        y_pred = model.predict(x)
        y_predict.extend(list(y_pred))
        y_test.extend(list(y))
    y_predict = np.array(y_predict)
    y_test = np.array(y_test)
    plt.plot(y_test, y_predict, "ro", label="Prediction values", alpha=0.3)
    plt.axline([0, 0], [1, 1])
    plt.plot(
        np.unique(y_test),
        np.poly1d(np.polyfit(y_test, y_predict.squeeze(), 1))(np.unique(y_test)),
        color="black",
        label="Poly1D fit",
    )
    plt.annotate(
        "r^2 = {:.3f}".format(r2_score(y_test, y_predict)),
        (0.7, 0.04),
        xycoords="axes fraction",
    )
    plt.annotate(
        "mse = {:.3f}".format(mean_squared_error(y_test, y_predict)),
        (0.7, 0.01),
        xycoords="axes fraction",
    )
    plt.ylabel("Predicted Pull time (s)")
    plt.xlabel("Actual Pull time (s)")
    plt.legend(loc="upper left")
    model_name = os.path.basename(model_path).split(".")[0]
    plt.suptitle(f"Model: {model_name}")
    if not write:
        plt.show()
    else:
        plt.savefig(f"eval-{model_name}.png")
    plt.close()


def evalulate_model(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--batch-size", default=128)
    args = parser.parse_args(leftover)
    if not os.path.isdir(args.directory):
        print(f"No such directory: {args.directory}")
        sys.exit(1)

    models = []
    if args.models is None:
        config = load_config()
        if config is None:
            print("No model provided and no default model configured")
            sys.exit(1)
        models.append(config.model)
    else:
        models.extend(args.models)

    for model_path in models:
        model = load_model(model_path, compile=False)
        loader = GroundsLoader(
            args.batch_size,
            (model.input_shape[1], model.input_shape[2]),
            directory=args.directory,
        )
        graph_model(model_path, model, loader, write=args.write)
