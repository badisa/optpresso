import os

from typing import List
from argparse import ArgumentParser, Namespace

import numpy as np

from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


def predict(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(leftover)

    model_path = args.model
    if model_path is None:
        model_path = "junk-model.h5"
    model = load_model(model_path)

    img_arr = img_to_array(
        load_img(os.path.expanduser(args.file_path), target_size=(255, 255))
    )
    print(model.predict(np.array([img_arr])))
