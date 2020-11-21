import os
import time

from typing import List, Any
from argparse import Namespace, ArgumentParser

import cv2


def get_user_input(prompt: str, converter) -> Any:
    while True:
        val = input(prompt)
        try:
            return converter(val)
        except Exception as e:
            print(f"Invalid value: {e}")


def capture(parent_args: Namespace, leftover: List[str]):
    parser = ArgumentParser()
    parser.add_argument("camera", default=0, type=int, help="Camera index to use")
    parser.add_argument("--output-dir", default=".", type=str)
    parser.add_argument(
        "--machine", default="linea_mini", type=str, help="Espresso machine used"
    )
    args = parser.parse_args(leftover)
    if not os.path.isdir(args.output_dir):
        print(f"No such directory: {args.output_dir}")
        return
    cam = cv2.VideoCapture(args.camera)
    cv2.namedWindow("capture")
    print("-Starting Capture-")
    print("------------------")
    print("Keyboard Shortcuts")
    print("------------------")
    print("c/enter - capture image and prompt for pull time")
    print("q/esc - quite")
    print("------------------")
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame")
                return
            cv2.imshow("capture", frame)
            k = cv2.waitKey(1)
            if k == 99 or k == 13:  # Hit C or enter for capture
                value = get_user_input("Espresso pull time:", float)
                now = int(time.time())
                output_path = os.path.join(args.output_dir, f"{now}-{args.machine}-{value}.png")
                cv2.imwrite(output_path, frame)
                print(f"Wrote frame to {output_path}")
            elif k == 113 or k == 27:  # Hit q or esc to quit
                print("Quitting")
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()
