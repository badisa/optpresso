import os
import sys

from typing import List
from argparse import ArgumentParser, Namespace

import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

from optpresso.data.config import load_config


def predict(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("file_path", nargs="*")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(leftover)

    model_path = args.model
    if model_path is None:
        config = load_config()
        if config is None:
            print("No model provided and no default model configured")
            sys.exit(1)
        model_path = config.model
    model = load_model(model_path)

    # TODO optimize this images array
    images = []
    for i, path in enumerate(args.file_path):
        images.append(
            img_to_array(
                load_img(
                    os.path.expanduser(path),
                    target_size=(model.input_shape[1], model.input_shape[2]),
                )
            )
        )
    predictions = model.predict(np.array(images))
    for path, predict in zip(args.file_path, predictions):
        print(f"{path}: Predicted pull time {predict[0]}s")
