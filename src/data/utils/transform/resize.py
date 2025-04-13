import cv2
import numpy as np
from typing import Tuple
from src.utils.timers import Timer

class Resize:
    def __init__(self, target_size: Tuple[int, int], mode: str = "bilinear", pad_value: int = 0):
        self.target_size = target_size
        self.mode = mode
        self.pad_value = pad_value

        self.interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA
        }.get(mode, cv2.INTER_LINEAR)

    def __call__(self, inputs: dict) -> dict:
        with Timer("Resize"):
            imgs = inputs.get("images")  # list of length T, each element [C, H, W]
            labels = inputs.get("labels")  # [T] list of labels per frame

            if imgs is None or labels is None:
                raise ValueError("'images' and 'labels' keys must be present in inputs.")

            T = len(imgs)
            C, H, W = imgs[0].shape
            target_height, target_width = self.target_size

            scale = min(target_width / W, target_height / H)
            new_width, new_height = int(W * scale), int(H * scale)

            resized_imgs = []

            for t in range(T):
                img_hwc = np.transpose(imgs[t], (1, 2, 0))
                resized_img = cv2.resize(img_hwc, (new_width, new_height), interpolation=self.interpolation)

                padded_img = cv2.copyMakeBorder(
                    resized_img,
                    top=0,
                    bottom=target_height - new_height,
                    left=0,
                    right=target_width - new_width,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[self.pad_value] * C
                )

                resized_imgs.append(np.transpose(padded_img, (2, 0, 1)))

                # 各フレームごとのlabelをスケール
                for label in labels[t]:
                    bbox = label["bbox"]
                    bbox = [coord * scale for coord in bbox]
                    label["bbox"] = bbox

            inputs["images"] = resized_imgs
            inputs["labels"] = labels

        return inputs