import math
import cv2
import numpy as np
from src.utils.timers import Timer


class Rotate:
    def __init__(self, angle: float = 0.0):
        """
        Args:
            angle (float):
                画像を回転させる角度（度単位）。
                正の値で反時計回り、負の値で時計回り。
        """
        self.angle = angle

    def __call__(self, inputs: dict) -> dict:

        with Timer("Rotate"):
            images = inputs.get("images")  # list of [C, H, W]
            labels = inputs.get("labels")  # [T] list of labels

            if images is None or labels is None:
                raise ValueError("inputs must contain 'images' and 'labels'")

            T = len(images)
            rotated_images = []

            for t in range(T):
                C, H, W = images[t].shape
                image = np.transpose(images[t], (1, 2, 0))  # [C, H, W] -> [H, W, C]

                rotated_image = self.rotate_image(image, self.angle)
                rotated_labels = self.rotate_bboxes(labels[t], self.angle, W, H)

                rotated_images.append(np.transpose(rotated_image, (2, 0, 1)))  # [H, W, C] -> [C, H, W]
                labels[t] = rotated_labels

            inputs["images"] = rotated_images
            inputs["labels"] = labels

        return inputs

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        H, W = image.shape[:2]
        center = (W / 2, H / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (W, H), borderValue=(114, 114, 114))
        return rotated_image

    def rotate_bboxes(self, labels: list, angle: float, img_w: int, img_h: int) -> list:
        theta = -math.radians(angle)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        cx0 = img_w / 2.0
        cy0 = img_h / 2.0

        for label in labels:
            bbox = label["bbox"]  # [left, top, right, bottom]

            corners_x = np.array([bbox[0], bbox[2], bbox[0], bbox[2]])
            corners_y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]])

            corners_x_rot = cos_t * (corners_x - cx0) - sin_t * (corners_y - cy0) + cx0
            corners_y_rot = sin_t * (corners_x - cx0) + cos_t * (corners_y - cy0) + cy0

            x_min = np.clip(np.min(corners_x_rot), 0, img_w)
            x_max = np.clip(np.max(corners_x_rot), 0, img_w)
            y_min = np.clip(np.min(corners_y_rot), 0, img_h)
            y_max = np.clip(np.max(corners_y_rot), 0, img_h)

            label["bbox"] = [x_min, y_min, x_max, y_max]

        return labels
