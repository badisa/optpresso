import sys

from argparse import ArgumentParser

from optpresso.training import train
from optpresso.capture import capture
from optpresso.predict import predict

TRAIN = "train"
CAPTURE = "capture"
PREDICT = "predict"


def main():
    parser = ArgumentParser(description="OPTPresso: ML for espresso")
    parser.add_argument("cmd", choices=[TRAIN, CAPTURE, PREDICT])
    # Args to be added, probably
    args, leftover = parser.parse_known_args()
    if args.cmd == TRAIN:
        train(args, leftover)
    elif args.cmd == CAPTURE:
        capture(args, leftover)
    elif args.cmd == PREDICT:
        predict(args, leftover)
    else:
        print(f"Unknown cmd: {args.cmd}")
        sys.exit(1)
