import os
import shutil
import random

from collections import defaultdict
from optpresso.utils import GroundsLoader

IMG_EXTS = [".jpg", ".png"]

TRAIN_DIR_NAME = "train"
VALIDATION_DIR_NAME = "validation"


def find_test_paths(directory: str):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() not in IMG_EXTS:
                continue
            data_path = os.path.join(root, file)
            try:
                pull_time = int(os.path.basename(os.path.dirname(data_path)))
            except (TypeError, ValueError):
                continue
            yield pull_time, data_path


def partition_data(input_dir: str, output_dir: str, validation_ratio: float):
    if validation_ratio > 1.0 or validation_ratio < 0.0:
        raise RuntimeError(
            f"validation_ratio must be between 0.0 and 1.0, got {validation_ratio}"
        )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    timings = defaultdict(list)
    for time, path in find_test_paths(input_dir):
        timings[time].append(path)
    for time, paths in timings.items():
        # Don't partition them the same way every time
        random.shuffle(paths)
        to_move = max(1, int(len(paths) * validation_ratio))
        if to_move != len(paths):
            validation_dir = os.path.join(output_dir, VALIDATION_DIR_NAME)
            if not os.path.isdir(validation_dir):
                os.mkdir(validation_dir)
            time_dir = os.path.join(validation_dir, str(time))
            if not os.path.isdir(time_dir):
                os.mkdir(time_dir)
            for path in paths[:to_move]:
                new_path = os.path.join(time_dir, os.path.basename(path))
                if os.path.isfile(new_path):
                    continue
                shutil.copy(path, new_path)
            paths = paths[to_move:]
        train_dir = os.path.join(output_dir, TRAIN_DIR_NAME)
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
        time_dir = os.path.join(train_dir, str(time))
        if not os.path.isdir(time_dir):
            os.mkdir(time_dir)
        for path in paths:
            new_path = os.path.join(time_dir, os.path.basename(path))
            if os.path.isfile(new_path):
                continue
            shutil.copy(path, new_path)
