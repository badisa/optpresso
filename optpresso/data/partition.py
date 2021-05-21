import os
import sys
import math
import shutil

from argparse import ArgumentParser
from tempfile import TemporaryDirectory

import numpy as np

from collections import defaultdict

IMG_EXTS = [".jpg", ".png"]

TEST_DIR_NAME = "test"
TRAIN_DIR_NAME = "train"
VALIDATION_DIR_NAME = "validation"


def find_test_paths(directory: str):
    for root, dirs, files in os.walk(directory):
        try:
            pull_time = int(os.path.basename(root))
        except (TypeError, ValueError):
            continue
        for file in files:
            if os.path.splitext(file)[1].lower() not in IMG_EXTS:
                continue
            data_path = os.path.join(root, file)
            yield pull_time, data_path


def k_fold_partition(input_dir: str, folds: int = 10) -> TemporaryDirectory:
    timings = defaultdict(list)
    for time, path in find_test_paths(input_dir):
        timings[time].append(path)
    for paths in timings.values():
        np.random.shuffle(paths)
    tmpdir = TemporaryDirectory()
    batch_size = math.ceil(len(paths) / folds)
    for fold in range(folds):
        fold_dir = os.path.join(tmpdir.name, str(fold))
        os.makedirs(fold_dir)
        for time, paths in timings.items():
            output_dir = os.path.join(fold_dir, str(time))
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            count = math.ceil(len(paths) / folds)
            for path in paths[count * fold : (count * fold) + count]:
                shutil.copy(path, os.path.join(output_dir, os.path.basename(path)))
    return tmpdir


def partition_data(
    input_dir: str, output_dir: str, validation_ratio: float, test_ratio: float = 0.1
):
    if validation_ratio + test_ratio > 1.0 or validation_ratio + test_ratio < 0.0:
        raise RuntimeError(
            f"validation_ratio must be between 0.0 and 1.0, got {validation_ratio}"
        )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    timings = defaultdict(list)
    for time, path in find_test_paths(input_dir):
        timings[time].append(path)
    validation_dir = os.path.join(output_dir, VALIDATION_DIR_NAME)
    if not os.path.isdir(validation_dir):
        os.mkdir(validation_dir)
    train_dir = os.path.join(output_dir, TRAIN_DIR_NAME)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    test_dir = os.path.join(output_dir, TEST_DIR_NAME)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    for time, paths in timings.items():
        # Don't partition them the same way every time
        np.random.shuffle(paths)
        for ratio, out_dir in [
            (validation_ratio, validation_dir),
            (test_ratio, test_dir),
        ]:
            to_move = max(1, int(len(paths) * ratio))
            if to_move != len(
                paths
            ):  # Never move the full set into the validation/test set
                time_dir = os.path.join(out_dir, str(time))
                if not os.path.isdir(time_dir):
                    os.mkdir(time_dir)
                for path in paths[:to_move]:
                    new_path = os.path.join(time_dir, os.path.basename(path))
                    shutil.copy(path, new_path)
                paths = paths[to_move:]
        time_dir = os.path.join(train_dir, str(time))
        if not os.path.isdir(time_dir):
            os.mkdir(time_dir)
        for path in paths:
            dir_name = os.path.dirname(path)
            name, ext = os.path.splitext(os.path.basename(path))
            new_path = os.path.join(time_dir, f"{name}{ext}")
            shutil.copy(path, new_path)


def partition_cmd(parent_args, leftover):
    parser = ArgumentParser(description="Partition data")
    parser.add_argument("src", help="Directory to partition")
    parser.add_argument("dest", help="Where to place partitioned data")
    parser.add_argument("validation_portion", type=float)
    parser.add_argument("--test-portion", default=0.0, type=float)
    parser.add_argument(
        "--update", action="store_true", help="Update an existing partition"
    )
    args = parser.parse_args(leftover)
    total_portion = args.validation_portion + args.test_portion
    if (total_portion - 1.0) > 0.0:
        print(f"Proportions must less than 1.0, got {total_portion}")
        sys.exit(1)
    print(f"Partitioning {args.src} into {args.dest}")
    src = os.path.expanduser(args.src)
    dest = os.path.expanduser(args.dest)
    tempdir = None
    if args.update:
        tempdir = TemporaryDirectory()
        to_move = set(
            [os.path.basename(path) for _, path in find_test_paths(src)]
        ).difference([os.path.basename(path) for _, path in find_test_paths(dest)])
        for time, path in find_test_paths(src):
            if os.path.basename(path) not in to_move:
                continue
            new_path = os.path.join(tempdir.name, str(time))
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            shutil.copy(path, os.path.join(new_path, os.path.basename(path)))
        src = tempdir.name
    partition_data(
        src, args.dest, args.validation_portion, test_ratio=args.test_portion
    )
    if tempdir is not None:
        tempdir.cleanup()
