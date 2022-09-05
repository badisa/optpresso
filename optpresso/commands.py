import os
import sys

from argparse import ArgumentParser


CONFIG = "config"
EVAL = "eval"
TRAIN = "train"
SERVE = "serve"
PARTITION = "partition"

TF_LOG_LEVEL = "TF_CPP_MIN_LOG_LEVEL"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


def main():
    if TF_LOG_LEVEL not in os.environ:
        os.environ[TF_LOG_LEVEL] = "2"
    # Don't add help, complicates things
    parser = ArgumentParser(description="OptPresso: ML for espresso", add_help=False)
    parser.add_argument(
        "cmd", choices=[TRAIN, EVAL, CONFIG, PARTITION, SERVE]
    )
    # Args to be added, probably
    args, leftover = parser.parse_known_args()

    # Do the imports like this because tensorflow is a monster
    if args.cmd == TRAIN:
        from optpresso.models.training import train

        train(args, leftover)

        predict(args, leftover)
    elif args.cmd == EVAL:
        from optpresso.models.eval import evalulate_model

        evalulate_model(args, leftover)
    elif args.cmd == CONFIG:
        from optpresso.data.config import config_optpresso

        config_optpresso(args, leftover)
    elif args.cmd == PARTITION:
        from optpresso.data.partition import partition_cmd

        partition_cmd(args, leftover)
    elif args.cmd == "serve":
        from optpresso.server import serve_server

        serve_server(args, leftover)
    else:
        print(f"Unknown cmd: {args.cmd}")
        sys.exit(1)
