import os
import sys

from argparse import ArgumentParser

INIT = "init"
EVAL = "eval"
TRAIN = "train"
CAPTURE = "capture"
PREDICT = "predict"
PARTITION = "partition"

TF_LOG_LEVEL = "TF_CPP_MIN_LOG_LEVEL"


def main():
    if TF_LOG_LEVEL not in os.environ:
        os.environ[TF_LOG_LEVEL] = "2"
    # Don't add help, complicates things
    parser = ArgumentParser(description="OptPresso: ML for espresso", add_help=False)
    parser.add_argument("cmd", choices=[TRAIN, CAPTURE, PREDICT, EVAL, INIT, PARTITION])
    # Args to be added, probably
    args, leftover = parser.parse_known_args()

    # Do the imports like this because tensorflow is a monster
    if args.cmd == TRAIN:
        from optpresso.models.training import train

        train(args, leftover)
    elif args.cmd == CAPTURE:
        from optpresso.capture import capture

        capture(args, leftover)
    elif args.cmd == PREDICT:
        from optpresso.predict import predict

        predict(args, leftover)
    elif args.cmd == EVAL:
        from optpresso.models.eval import evalulate_model

        evalulate_model(args, leftover)
    elif args.cmd == INIT:
        from optpresso.data.config import init_optpresso

        init_optpresso(args, leftover)
    elif args.cmd == PARTITION:
        from optpresso.data.partition import partition_cmd

        partition_cmd(args, leftover)
    else:
        print(f"Unknown cmd: {args.cmd}")
        sys.exit(1)
