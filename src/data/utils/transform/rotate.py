import math
import numpy as np
import cv2
from src.utils.timers import Timer

class Rotate:
    def __init__(self, angle: float = 0.0):
        """
        Args:
            angle (float):
                回転角度（度単位）。正値で反時計回り、負値で時計回り。
        """
        self.angle = angle

    def __call__(self, inputs: dict) -> dict:
        with Timer("Rotate"):
            images = inputs.get("images")  # [T, C, H, W]
            labels = inputs.get("labels")  # [T] list of labels
            events = inputs.get("events")  # [T, C, H, W] (optional)

            if images is None or labels is None:
                raise ValueError("inputs must contain 'images' and 'labels'")

            T, C, H, W = images.shape

            for t in range(T):
                image = np.transpose(images[t], (1, 2, 0))  # [C, H, W] → [H, W, C]
                rotated_image = self.rotate_image(image, self.angle)
                rotated_labels = self.rotate_bboxes(labels[t], self.angle, W, H)

                images[t] = np.transpose(rotated_image, (2, 0, 1))  # [H, W, C] → [C, H, W]
                labels[t] = rotated_labels

            inputs["images"] = images
            inputs["labels"] = labels

            # イベントも回転（各チャンネル独立）
            if events is not None:
                T_e, C_e, H_e, W_e = events.shape
                rotated_events = np.empty_like(events)
                for t in range(T_e):
                    for c in range(C_e):
                        rotated_events[t, c] = self.rotate_event_channel(events[t, c], self.angle)
                inputs["events"] = rotated_events

            return inputs

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        画像を指定された角度で回転。
        """
        H, W = image.shape[:2]
        center = (W / 2, H / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (W, H), borderValue=(114, 114, 114))
        return rotated_image

    def rotate_event_channel(self, channel: np.ndarray, angle: float) -> np.ndarray:
        """
        単一チャネルのイベントマップを回転。
        """
        H, W = channel.shape
        center = (W / 2, H / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        return cv2.warpAffine(channel, rotation_matrix, (W, H), borderValue=0)

    def rotate_bboxes(self, labels: list, angle: float, img_w: int, img_h: int) -> list:
        """
        各 bbox の4頂点を回転させ、新しい外接矩形を計算。
        """
        theta = -math.radians(angle)  # 回転方向調整（OpenCVは反時計回り）
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        cx0 = img_w / 2.0
        cy0 = img_h / 2.0

        for label in labels:
            bbox = label["bbox"]  # [x1, y1, x2, y2]

            corners_x = np.array([bbox[0], bbox[2], bbox[0], bbox[2]])
            corners_y = np.array([bbox[1], bbox[1], bbox[3], bbox[3]])

            # 原点を中心に回転 → 元の位置へ戻す
            corners_x_rot = cos_t * (corners_x - cx0) - sin_t * (corners_y - cy0) + cx0
            corners_y_rot = sin_t * (corners_x - cx0) + cos_t * (corners_y - cy0) + cy0

            x_min = np.clip(np.min(corners_x_rot), 0, img_w)
            x_max = np.clip(np.max(corners_x_rot), 0, img_w)
            y_min = np.clip(np.min(corners_y_rot), 0, img_h)
            y_max = np.clip(np.max(corners_y_rot), 0, img_h)

            label["bbox"] = [x_min, y_min, x_max, y_max]

        return labels
