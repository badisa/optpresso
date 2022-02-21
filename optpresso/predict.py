import os
import sys

from typing import List
from argparse import ArgumentParser, Namespace

import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img

import cv2

from optpresso.models.serialization import load_model
from optpresso.data.config import load_config, OptpressoConfig
from optpresso.capture import get_user_input


def predict_from_camera(model, camera: int, config: OptpressoConfig):
    predictions = np.array([], dtype=np.float32)
    secondary_model = None
    if config is not None and config.use_secondary_model:
        secondary_model = config.load_secondary_model()
    cam = cv2.VideoCapture(camera)
    # Copy pasta
    cv2.namedWindow("capture")
    print("-Starting Prediction-")
    print("---------------------")
    print("Keyboard Shortcuts")
    print("------------------")
    print("c/enter - capture image for prediction")
    print("d - delete last predict")
    # TODO add a clear command
    print("p - print current prediction values")
    if secondary_model is not None:
        print("t - view secondary model predictions")
        print("u - update secondary model")
    print("q/esc - quit")
    print("---------------------")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame")
                return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(
                "Focus (higher is better): {:.2f}".format(
                    cv2.Laplacian(gray, cv2.CV_64F).var()
                ),
                end="\r",
            )
            cv2.imshow("capture", frame)
            k = cv2.waitKey(1)
            if k == 99 or k == 13:  # Hit c or enter for capture
                # Width and height are reversed
                keras_frame = cv2.resize(
                    frame, (model.input_shape[2], model.input_shape[1])
                )
                keras_frame = keras_frame[..., ::-1].astype(
                    np.float32
                )  # Reverse BGR to RGB and convert to float32
                pred = model.predict(np.array([keras_frame]))
                predictions = np.concatenate((predictions, pred), axis=None)
                print(
                    "Captured image has predicted time: {:.2f}".format(float(pred[0]))
                )
                print(
                    f"Points: {len(predictions)}, Mean = {predictions.mean()}, Std Dev = {predictions.std()}"
                )
            elif k == 113 or k == 27:  # Hit q or esc to quit
                print("Quitting")
                break
            elif k == 112:
                print()
                print("-- Current Predictions --")
                print(list(predictions))
                print(
                    f"# Predictions = {len(predictions)}, Mean = {predictions.mean()}, Std Dev = {predictions.std()}"
                )
            elif k == 100:
                print()
                if len(predictions):
                    print("Dropping last prediction")
                    predictions = predictions[:-1]
                else:
                    print("No predictions left")
            elif (
                k == 116 and secondary_model is not None
            ):  # Hit t to predict secondary model
                print()
                if not len(predictions):
                    print("No predictions to test yet")
                    continue
                preds = (
                    secondary_model.predict(predictions.reshape(-1, 1)).reshape(
                        len(predictions)
                    )
                    + predictions
                )
                for new, old in zip(preds, predictions):
                    print(f"Secondary Model: {new}, Original Model: {old}")
            elif (
                k == 117 and secondary_model is not None
            ):  # Hit u to take predictions and update with the 'real' value
                print()
                value = get_user_input("Actual Espresso pull time:", float)
                config.update_secondary_model(predictions, value)
                # Reload the model, probably should create a wrapper around
                # the Gaussian model to make this less goofy
                secondary_model = config.load_secondary_model()

    finally:
        cam.release()
        cv2.destroyAllWindows()


def predict(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("file_path", nargs="*")
    parser.add_argument("--camera", default=None, type=int)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(leftover)
    if len(args.file_path) == 0 and args.camera is None:
        print("Must provide file paths or a camera for predictions")
        sys.exit(1)

    config = load_config()
    model_path = args.model
    if model_path is None:
        if config is None:
            print("No model provided and no default model configured")
            sys.exit(1)
        model_path = config.model
    model = load_model(model_path, compile=False)

    if len(args.file_path):
        # TODO optimize this images array
        images = []
        for path in args.file_path:
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
        print(predictions.mean(), predictions.std())
    else:
        predict_from_camera(model, args.camera, config)
