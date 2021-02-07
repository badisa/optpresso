import os
import sys
import json
import shutil

from typing import List, Any, Optional
from dataclasses import dataclass, asdict
from argparse import Namespace, ArgumentParser

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor


DEFAULT_CONFIG_DIR = os.path.expanduser("~/.optpresso")
OPTPRESSO_CONFIG_ENV = "OPTPRESSO_DIR"


@dataclass
class OptpressoConfig:
    model: str
    data_path: str = os.path.join(DEFAULT_CONFIG_DIR, "model.npy")
    machine: str = ""
    grinder: str = ""
    use_secondary_model: bool = True
    _model_data: Optional[np.array] = None

    @property
    def model_data(self):
        if self._model_data is None:
            if not self.use_secondary_model:
                raise RuntimeError("Unable to use personal model, its disabled")
            if not os.path.isfile(self.data_path):
                self._model_data = np.array([[], []])
            else:
                self._model_data = np.load(self.data_path)
        return self._model_data

    def update_secondary_model(self, predictions: List[float], actual: float):
        data = self.model_data
        for pred in predictions:
            data = np.concatenate((data, [[pred], [actual]]), axis=1)
        np.save(self.data_path, data)
        self._model_data = None

    def load_secondary_model(self) -> Optional[GaussianProcessRegressor]:
        data = self.model_data
        model = GaussianProcessRegressor()
        if not data.any():
            return model
        # Use the difference, in which case the default value of 0 for the GP
        # function is just added to the model's values.
        y = data[1] - data[0]
        model.fit(np.array(data[0]).reshape(-1, 1), np.array(y).reshape(-1, 1))
        return model

    def save(self):
        directory = DEFAULT_CONFIG_DIR
        if OPTPRESSO_CONFIG_ENV in os.environ:
            directory = os.environ[OPTRESSO_CONFIG_ENV]
        directory = os.path.expanduser(directory)
        path = os.path.join(directory, "config")
        with open(path, "w") as ofs:
            json.dump(asdict(self), ofs)


def load_config() -> Optional[OptpressoConfig]:
    directory = DEFAULT_CONFIG_DIR
    if OPTPRESSO_CONFIG_ENV in os.environ:
        directory = os.environ[OPTRESSO_CONFIG_ENV]
    directory = os.path.expanduser(directory)
    path = os.path.join(directory, "config")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as ifs:
        config = json.load(ifs)
    return OptpressoConfig(**config)


def init_optpresso(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(description="Initialize Optpresso")
    parser.add_argument(
        "model",
        help="Model to use as default, check the README for a link to the default model",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy the model to the config directory",
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_CONFIG_DIR,
        type=str,
        help=f"Directory to store config in, if not default set the env variable {OPTPRESSO_CONFIG_ENV}",
    )
    parser.add_argument(
        "--machine", default=None, help="Name of the espresso machine being used"
    )
    parser.add_argument(
        "--grinder", default=None, help="Name of the espresso grinder used"
    )
    parser.add_argument(
        "--disable-secondary-model",
        action="store_true",
        help="Disable secondary model intended to better fit individual workflows",
    )
    args = parser.parse_args(leftover)
    config_dir = os.path.expanduser(args.dir)
    config_path = os.path.join(config_dir, "config")
    if os.path.isdir(config_dir) and os.path.isfile(config_path):
        print(f"Optpresso already configured at {config_dir}")
        sys.exit(1)
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)
    model_path = args.model
    if not args.no_copy:
        model_path = os.path.join(config_dir, os.path.basename(args.model))
        shutil.copy(args.model, model_path)
    config = OptpressoConfig(
        model=model_path,
        grinder=args.grinder,
        machine=args.machine,
        use_secondary_model=not args.disable_secondary_model,
        data_path=os.path.join(config_dir, "model.npy"),
    )
    config.save()
    print("Configured Optpresso, good luck!")
