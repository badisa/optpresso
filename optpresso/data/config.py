import os
import sys
import json
import shutil

from typing import List, Any, Optional
from dataclasses import dataclass, asdict
from argparse import Namespace, ArgumentParser


DEFAULT_CONFIG_DIR = "~/.optpresso"
OPTPRESSO_CONFIG_ENV = "OPTPRESSO_DIR"


@dataclass
class OptpressoConfig:
    model: str
    machine: str


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
        default="~/.optpresso",
        type=str,
        help=f"Directory to store config in, if not default set the env variable {OPTPRESSO_CONFIG_ENV}",
    )
    parser.add_argument(
        "--machine", default=None, help="Name of the espresso machine being used"
    )
    args = parser.parse_args(leftover)
    config_dir = os.path.expanduser(args.dir)
    config_path = os.path.join(config_dir, "config")
    if os.path.isdir(config_dir) and os.path.isfile(config_path):
        print(f"Optpresso already configured at {config_dir}")
        sys.exit(1)
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)
    if not args.no_copy:
        shutil.copy(args.model, os.path.join(config_dir, os.path.basename(args.model)))
    model_path = args.model
    config = OptpressoConfig(
        model=model_path,
        machine=args.machine,
    )
    with open(config_path, "w") as ofs:
        json.dump(asdict(config), ofs)
    print("Configured Optpresso, good luck!")
