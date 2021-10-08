import os
import time

from typing import List, Any
from argparse import Namespace, ArgumentParser

import cv2
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img


def flip_image(path: str, flip_img: bool = True, mirror_img: bool = False):
    """Flip the image and save files in the same place with the -flip-<int> as a suffix

    If the int is 0 then it was a horizontal flip
    If the int is 1 then it was a vertical flip
    if the int is 2 then it was both
    """
    name, ext = os.path.splitext(path)
    img = load_img(path)
    if mirror_img:
        mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror.save(f"{name}-flip-0{ext}")
    if flip_img:
        flip = img.transpose(Image.FLIP_TOP_BOTTOM)
        flip.save(f"{name}-flip-1{ext}")
    if flip_img and mirror_img:
        img = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        img.save(f"{name}-flip-2{ext}")


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
        "--no-flip", action="store_true", help="Disable flipping and mirroring images"
    )
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
    print("q/esc - quit")
    print("------------------")
    flip_images = not args.no_flip
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
            if k == 99 or k == 13:  # Hit C or enter for capture
                print()
                value = get_user_input("Espresso pull time:", int)
                now = int(time.time())
                dir_name = os.path.join(args.output_dir, str(value))
                if not os.path.isdir(dir_name):
                    os.mkdir(dir_name)
                output_path = os.path.join(dir_name, f"{now}-{args.machine}.png")
                cv2.imwrite(output_path, frame)
                if flip_images:
                    flip_image(output_path, flip_img=True, mirror_img=True)
                print(f"Wrote frame to {output_path}, Flipped: {flip_images}")
            elif k == 113 or k == 27:  # Hit q or esc to quit
                print("Quitting")
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
