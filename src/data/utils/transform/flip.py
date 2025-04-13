import numpy as np
from src.utils.timers import Timer


class Flip:
    def __init__(self, vertical: bool = False, horizontal: bool = False):
        """
        Args:
            vertical: 垂直方向の反転を行うか
            horizontal: 水平方向の反転を行うか
        """
        self.vertical = vertical
        self.horizontal = horizontal

    def __call__(self, inputs: dict) -> dict:
        with Timer("Flip"):
            images = inputs.get("images")  # [T, C, H, W]
            labels = inputs.get("labels")  # [T] list of label dicts
            events = inputs.get("events")  # [T, C, H, W] (optional)

            if images is None or labels is None:
                raise ValueError("inputs must contain 'images' and 'labels'")

            T, C, H, W = images.shape

            for t in range(T):
                image = np.transpose(images[t], (1, 2, 0))  # [C, H, W] -> [H, W, C]

                # Flip vertically
                if self.vertical:
                    image = np.flip(image, axis=0)
                    for label in labels[t]:
                        y1 = label["bbox"][1]
                        y2 = label["bbox"][3]
                        label["bbox"][1] = H - y2
                        label["bbox"][3] = H - y1

                # Flip horizontally
                if self.horizontal:
                    image = np.flip(image, axis=1)
                    for label in labels[t]:
                        x1 = label["bbox"][0]
                        x2 = label["bbox"][2]
                        label["bbox"][0] = W - x2
                        label["bbox"][2] = W - x1

                images[t] = np.transpose(image, (2, 0, 1))  # [H, W, C] -> [C, H, W]

                # events も反転
                if events is not None:
                    for c in range(events.shape[1]):
                        if self.vertical:
                            events[t, c] = np.flip(events[t, c], axis=0)
                        if self.horizontal:
                            events[t, c] = np.flip(events[t, c], axis=1)

            inputs["images"] = images
            inputs["labels"] = labels
            if events is not None:
                inputs["events"] = events

            return inputs
