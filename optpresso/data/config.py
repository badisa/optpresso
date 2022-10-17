import os
import sys
import json
import shutil

from typing import List, Optional, Any
from dataclasses import dataclass, asdict
from argparse import Namespace, ArgumentParser

from sklearn.gaussian_process import GaussianProcessRegressor


DEFAULT_CONFIG_DIR = os.path.expanduser("~/.optpresso")
OPTPRESSO_CONFIG_ENV = "OPTPRESSO_DIR"

INIT = "init"
UPDATE = "update"
CONFIG = "config"


@dataclass
class OptpressoConfig:
    model: str
    data_path: str = ""
    machine: str = ""
    grinder: str = ""
    use_secondary_model: bool = False
    _model_data: Any = None

    def save(self):
        directory = DEFAULT_CONFIG_DIR
        if OPTPRESSO_CONFIG_ENV in os.environ:
            directory = os.environ[OPTPRESSO_CONFIG_ENV]
        directory = os.path.expanduser(directory)
        path = os.path.join(directory, CONFIG)
        with open(path, "w") as ofs:
            json.dump(asdict(self), ofs)

    @classmethod
    def load(cls, path: str):
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise ValueError(f"No such path: {path}")
        with open(path, "r") as ifs:
            config = json.load(ifs)
        return OptpressoConfig(**config)


def load_config() -> Optional[OptpressoConfig]:
    directory = DEFAULT_CONFIG_DIR
    if OPTPRESSO_CONFIG_ENV in os.environ:
        directory = os.environ[OPTPRESSO_CONFIG_ENV]
    path = os.path.join(directory, CONFIG)
    try:
        return OptpressoConfig.load(path)
    except ValueError:
        return None


def config_optpresso(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser(description="Configure Optpresso")
    parser.add_argument(
        "subcmd", help="Configuration command", choices=["init", "update"]
    )
    args, leftover = parser.parse_known_args(leftover)
    update_config(parent_args, leftover, init=args.subcmd == INIT)


def update_config(parent_args: Namespace, leftover: List[str], init: bool = True):
    parser = ArgumentParser(description="Set Optpresso config")
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
        help=f"Directory to store config in, if not default set the env variable '{OPTPRESSO_CONFIG_ENV}'",
    )
    parser.add_argument(
        "--machine", default=None, help="Name of the espresso machine being used"
    )
    parser.add_argument(
        "--grinder", default=None, help="Name of the espresso grinder used"
    )
    args = parser.parse_args(leftover)
    config_dir = os.path.expanduser(args.dir)
    config_path = os.path.join(config_dir, CONFIG)
    if init:
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
        use_secondary_model=False,
    )
    config.save()
    print("Configured Optpresso, good luck!")
