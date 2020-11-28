import os

from typing import List
from argparse import ArgumentParser, Namespace

import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


def predict(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("file_path", nargs="*")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(leftover)

    model_path = args.model
    if model_path is None:
        model_path = "junk-model.h5"
    model = load_model(model_path)

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
