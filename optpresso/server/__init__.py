import os
import sys
import json
import time
import random
import webbrowser

from typing import Dict
from tempfile import mkstemp
from urllib.request import urlopen
from argparse import ArgumentParser
from dataclasses import asdict, fields

import cv2
import numpy as np

from flask import Flask, request, send_from_directory
from markupsafe import escape
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import optpresso
from optpresso.utils import set_random_seed
from optpresso.data.config import load_config, OptpressoConfig
from optpresso.models.serialization import load_model


optpresso_dir = os.path.dirname(os.path.dirname(optpresso.__file__))

app = Flask(__name__)

config = load_config()
model = None

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static/compiled/")

config_to_remove = ("data_path", "model", "use_secondary_model", "_model_data")

args = None

OUTPUTS = ["train", "validation", "test"]


@app.route("/")
def do_get():
    with open(os.path.join(TEMPLATES_DIR, "index.html"), "r") as ifs:
        return ifs.read()


@app.route("/static/<path>")
def serve_static(path):
    static_path = os.path.join(STATIC_DIR, path)
    if os.path.isfile(static_path):
        with open(static_path) as ifs:
            return ifs.read()
    else:
        print("Oh no", static_path)


@app.route("/config/", methods=["POST", "GET"])
def get_config():
    if request.method == "POST":
        config_fields = [x.name for x in fields(config)]
        for field, val in request.form.items():
            if field in config_fields:
                setattr(config, field, val)
        config.save()
        return {}
    else:
        if config is None:
            return {}
        data = asdict(config)
        for key in list(data):
            if key in config_to_remove:
                data.pop(key)
        return data


@app.route("/predict/", methods=["POST"])
def predict():
    img = request.form["image"]
    resp = urlopen(img)
    # Support windows, its a total hack
    fs, path = mkstemp(suffix=".png")
    os.close(fs)
    try:
        with open(path, "wb") as ofs:
            ofs.write(resp.file.read())
        keras_frame = img_to_array(
            load_img(path, target_size=(model.input_shape[1], model.input_shape[2]))
        )
    finally:
        os.remove(path)
    pred = model.predict(np.array([keras_frame]))
    return {"prediction": float(pred[0][0])}


@app.route("/capture/", methods=["POST"])
def capture():
    if args is None:
        raise RuntimeError("Something horribly wrong has happened")
    if "pullTime" not in request.form:
        return {"error": "No pull time!"}, 400
    pull_time = request.form["pullTime"]
    if not pull_time.isdigit():
        return {"error": "Invalid pull time"}, 400
    pull_time = int(pull_time)
    img = request.form["image"]
    resp = urlopen(img)

    machine = request.form["machine"]
    now = int(time.time())
    ext = ".png"
    name = f"{now}-{machine}{ext}"
    fs, path = mkstemp(suffix=ext)
    os.close(fs)
    try:
        with open(path, "wb") as ofs:
            ofs.write(resp.file.read())
        img = load_img(path)
    finally:
        os.remove(path)
    # Simple way of storing more information about the image
    metadata = PngInfo()
    fields_to_not_attach = {"image"}
    for key, value in request.form.items():
        if key in fields_to_not_attach:
            continue
        metadata.add_text(key, value)

    # Don't do this every request
    if args.capture_split:
        split_ratios = [int(x) for x in args.split_ratio.split(",")]
        ratio_total = sum(split_ratios)
        rand = random.randint(0, ratio_total)
        limit = 0
        for i, split in enumerate(split_ratios):
            limit += split
            if rand <= limit:
                output_dir = os.path.join(args.capture_dir, OUTPUTS[i])
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                output_dir = os.path.join(output_dir, str(pull_time))
                break
    else:
        output_dir = os.path.join(args.capture_dir, str(pull_time))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    img.save(os.path.join(output_dir, name), pnginfo=metadata)
    return {}


def serve_server(parent_args, leftover):
    global args
    global model
    parser = ArgumentParser(description="Optpresso Web UI")
    parser.add_argument(
        "capture_dir", default=".", help="Directory to write out captured images to"
    )
    parser.add_argument("--browser", action="store_true", help="Open the browser")
    parser.add_argument("--port", default=8888, type=int, help="Port to serve on")
    parser.add_argument(
        "--seed", default=0, type=int, help="Seed for placement of captured images"
    )
    parser.add_argument(
        "--capture-split",
        action="store_true",
        help="Split the images between a validation, train and test directory",
    )
    parser.add_argument(
        "--split-ratio",
        default="7,2,1",
        help="Ratios of train, validation and test, default is '7,2,1', should sum to 10",
    )
    args = parser.parse_args(leftover)
    if len(args.split_ratio.split(",")) != 3:
        print("Split Ratio requires 3 comma delimited integers, such as 7,2,1")
        sys.exit(1)
    if not all([x.isdigit() for x in args.split_ratio.split(",")]):
        print("Split Ratio values must all be positive integers")
        sys.exit(1)
    set_random_seed(args.seed)
    if config is None:
        print("Run optpresso init first")
        sys.exit(1)
    if args.capture_split:
        args.capture_dir = os.path.expanduser(args.capture_dir)
    model = load_model(config.model, compile=False)
    if args.browser:
        webbrowser.open_new_tab(f"http://localhost:{args.port}")
    app.run(port=args.port)
