import os
import sys
from typing import List
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor

from optpresso.utils import GroundsLoader
from optpresso.data.config import load_config
from optpresso.models.serialization import load_model


def graph_model(model_path, model, loader, write: bool = False):
    _, y_true, y_predict = predict_values(model, loader)
    write_evaluation(model_path, y_true, y_predict, write=write)


def write_evaluation(model_path, y_true, y_predict, write: bool = False):
    plt.plot(y_true, y_predict, "ro", label="Prediction values", alpha=0.3)
    plt.axline([0, 0], [1, 1])
    uni_y_true = np.unique(y_true)
    plt.plot(
        uni_y_true,
        np.poly1d(np.polyfit(y_true, y_predict, 1))(uni_y_true),
        color="black",
        label="Poly1D fit",
    )
    plt.annotate(
        "r^2 = {:.2f}".format(r2_score(y_true, y_predict)),
        (0.7, 0.04),
        xycoords="axes fraction",
    )
    plt.annotate(
        "mse = {:.2f}".format(mean_squared_error(y_true, y_predict)),
        (0.7, 0.01),
        xycoords="axes fraction",
    )
    plt.annotate(
        "mae = {:.2f}".format(mean_absolute_error(y_true, y_predict)),
        (0.7, 0.07),
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


def predict_values(model, loader):
    paths = np.empty(len(loader), dtype=object)
    y_predict = np.empty(len(loader), dtype=np.float64)
    y_actual = np.empty(len(loader), dtype=np.float64)
    offset = 0
    for fpaths, (x, y) in loader.generator_with_paths():
        count = len(y)
        y_predict[offset : offset + count] = model.predict(x).squeeze()
        y_actual[offset : offset + count] = y
        paths[offset : offset + count] = np.array(fpaths)
        offset += count
    return paths, y_actual, y_predict


def evalulate_model(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(
        "Evaluate a directory against either the configure model or a different model"
    )
    parser.add_argument("directory")
    parser.add_argument(
        "--write", action="store_true", help="Write out the graph to file"
    )
    parser.add_argument(
        "--models", nargs="+", help="Path to a list of models to evaluate"
    )
    parser.add_argument(
        "--batch-size", default=512, type=int, help="Batch size to evaluate models"
    )
    parser.add_argument(
        "--outlier-output",
        default=None,
        type=str,
        help="Write out line delimited file of top outliers",
    )
    parser.add_argument(
        "--num-outlier", default=50, type=int, help="Number of outliers to write out"
    )
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
        paths, y_act, y_pred = predict_values(model, loader)
        if args.outlier_output is not None:
            diff = np.abs(y_act - y_pred)
            path_diff_stack = np.dstack((paths, diff))
            # Avoid getting a shape of (1, len(paths), 2), which borks the sort
            path_diff_stack.resize(len(paths), 2)
            # Get the reverse sort
            reverse_sort = np.argsort(path_diff_stack[:, 1])[::-1]
            path_diff_stack = path_diff_stack[reverse_sort]
            count = 0
            with open(args.outlier_output, "w") as ifs:
                for x in path_diff_stack:
                    # print(x)
                    count += 1
                    ifs.write(f"{x[0]}: {x[1]}\n")
                    if count > args.num_outlier:
                        break
                print("Total", count, args.num_outlier)
        write_evaluation(model_path, y_act, y_pred, write=args.write)
