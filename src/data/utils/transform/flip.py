import numpy as np

from src.utils.timers import Timer


class Flip:
    def __init__(self, vertical: bool = False, horizontal: bool = False):
        """
        Args:
            vertical: 垂直方向の反転を有効にするか。
            horizontal: 水平方向の反転を有効にするか。
        """
        self.vertical = vertical
        self.horizontal = horizontal

    def __call__(self, inputs: dict) -> dict:
        """
        Args:
            inputs: dict
                "images": list of ndarray [C, H, W], length T
                "labels": list (len=T), 各要素はbboxの辞書リスト

        Returns:
            flipped inputs
        """
        with Timer("Flip"):

            images = inputs.get("images")  # list of [C, H, W]
            labels = inputs.get("labels")

            if images is None or labels is None:
                raise ValueError("inputs must contain 'images' and 'labels'")

            T = len(images)
            flipped_images = []

            for t in range(T):
                C, H, W = images[t].shape
                image = np.transpose(images[t], (1, 2, 0))  # [C, H, W] -> [H, W, C]

                if self.vertical:
                    image = np.flip(image, axis=0)
                    for label in labels[t]:
                        bbox = label["bbox"]
                        bbox_top = bbox[1]
                        bbox_height = bbox[3] - bbox[1]
                        bbox[1] = H - (bbox_top + bbox_height)
                        bbox[3] = bbox[1] + bbox_height
                        label["bbox"] = bbox

                if self.horizontal:
                    image = np.flip(image, axis=1)
                    for label in labels[t]:
                        bbox = label["bbox"]
                        bbox_left = bbox[0]
                        bbox_width = bbox[2] - bbox[0]
                        bbox[0] = W - (bbox_left + bbox_width)
                        bbox[2] = bbox[0] + bbox_width
                        label["bbox"] = bbox

                flipped_images.append(np.transpose(image, (2, 0, 1)))  # [H, W, C] -> [C, H, W]

            inputs['images'] = flipped_images
            inputs['labels'] = labels

        return inputs
