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


def evalulate_model(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--batch-size", default=16)
    args = parser.parse_args(leftover)

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
        model = load_model(model_path)
        loader = GroundsLoader(
            args.directory,
            args.batch_size,
            (model.input_shape[1], model.input_shape[2]),
        )
        x_test, y_test = loader.get_batch(0, len(loader))

        # Give us a clever best fit
        # https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html
        y_predict = model.predict(x_test)
        plt.plot(y_test, y_predict, "ro", label="Prediction values", alpha=0.3)
        gp = GaussianProcessRegressor()
        gp.fit(y_test[:, np.newaxis], y_predict)
        # Start at 3 as that is the floor for the model based on the machine
        xfit = np.linspace(3, max(y_predict.max(), y_test.max()), 120)
        yfit, mse_err = gp.predict(xfit[:, np.newaxis], return_std=True)
        yfit = yfit.squeeze()
        dyfit = 2 * np.sqrt(mse_err)  # 2*sigma ~ 95% confidence region
        plt.plot(xfit, yfit, "-", color="gray", label="GP Fit")
        plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color="gray", alpha=0.2)
        plt.axline([0, 0], [1, 1])
        plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_predict.squeeze(), 1))(np.unique(y_test)), color="black", label="Poly1D fit")
        plt.annotate("r-squared = {:.3f}".format(r2_score(y_test, y_predict)), (0, 1))
        plt.annotate("mse = {:.3f}".format(mean_squared_error(y_test, y_predict)), (0, 3))
        plt.ylabel("Predicted Pull time (s)")
        plt.xlabel("Actual Pull time (s)")
        plt.legend()
        plt.suptitle(f"Model: {model_path}")
        if not args.write:
            plt.show()
        else:
            model_name = os.path.basename(model_path).split(".")[0]
            plt.savefig(f"eval-{model_name}.png")
        plt.close()
