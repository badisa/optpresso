import sys

from argparse import ArgumentParser

from optpresso.capture import capture
from optpresso.predict import predict
from optpresso.models.training import train
from optpresso.models.eval import evalulate_model
from optpresso.data.config import init_optpresso

INIT = "init"
EVAL = "eval"
TRAIN = "train"
CAPTURE = "capture"
PREDICT = "predict"


def main():
    parser = ArgumentParser(description="OPTPresso: ML for espresso")
    parser.add_argument("cmd", choices=[TRAIN, CAPTURE, PREDICT, EVAL, INIT])
    # Args to be added, probably
    args, leftover = parser.parse_known_args()
    if args.cmd == TRAIN:
        train(args, leftover)
    elif args.cmd == CAPTURE:
        capture(args, leftover)
    elif args.cmd == PREDICT:
        predict(args, leftover)
    elif args.cmd == EVAL:
        evalulate_model(args, leftover)
    elif args.cmd == INIT:
        init_optpresso(args, leftover)
    else:
        print(f"Unknown cmd: {args.cmd}")
        sys.exit(1)
