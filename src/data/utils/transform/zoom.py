import cv2
import numpy as np
import random
from typing import Tuple
from src.utils.timers import Timer

def _find_zoom_center(labels):
    if not labels:
        return None
    centers = []
    for label in labels:
        bbox = label["bbox"]
        x1, y1, x2, y2 = bbox
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    if not centers:
        return None
    avg_x = int(np.mean([c[0] for c in centers]))
    avg_y = int(np.mean([c[1] for c in centers]))
    return avg_x, avg_y

class RandomZoom:
    def __init__(self,
                 prob_weight=(8, 2),
                 in_scale=(1.0, 1.5),
                 out_scale=(1.0, 1.2),
                 center_margin_ratio=0.2,):
        self.prob_weight = prob_weight
        self.in_scale = in_scale
        self.out_scale = out_scale
        self.center_margin_ratio = center_margin_ratio

    def _get_random_center(self, H, W):
        margin_x = int(W * self.center_margin_ratio)
        margin_y = int(H * self.center_margin_ratio)
        center_x = random.randint(margin_x, W - margin_x)
        center_y = random.randint(margin_y, H - margin_y)
        return center_x, center_y

    def __call__(self, inputs: dict) -> dict:
        with Timer("RandomZoom"):
            images = inputs.get("images")  # [T, C, H, W]
            labels = inputs.get("labels")  # list of list of dict
            events = inputs.get("events", None)  # [T, C, H, W] or None

            if images is None or labels is None:
                raise ValueError("inputs must contain 'images' and 'labels'")

            T, C, H, W = images.shape
            zoom_type = random.choices(["in", "out"], weights=self.prob_weight, k=1)[0]
            scale = random.uniform(*(self.in_scale if zoom_type == "in" else self.out_scale))

            for t in range(T):
                center = _find_zoom_center(labels[t])
                if center is None:
                    center = self._get_random_center(H, W)

                if zoom_type == "in":
                    images[t], labels[t], event_out = self.zoom_in(
                        images[t], labels[t],
                        events[t] if events is not None else None,
                        scale, center, H, W
                    )
                else:
                    images[t], labels[t], event_out = self.zoom_out(
                        images[t], labels[t],
                        events[t] if events is not None else None,
                        scale, center, H, W
                    )
                if events is not None:
                    events[t] = event_out

            inputs["images"] = images
            inputs["labels"] = labels
            if events is not None:
                inputs["events"] = events
            return inputs


    def zoom_in(self, img, label_list, event, scale, center, H, W):
        cx, cy = center
        new_H, new_W = int(H / scale), int(W / scale)
        x1 = max(0, min(int(cx - new_W // 2), W - new_W))
        y1 = max(0, min(int(cy - new_H // 2), H - new_H))

        # 画像処理
        img_hwc = np.transpose(img, (1, 2, 0))
        cropped = img_hwc[y1:y1 + new_H, x1:x1 + new_W]
        zoomed = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_CUBIC)
        img_out = np.transpose(zoomed, (2, 0, 1))

        # イベント処理（各チャネル individually）
        event_out = None
        if event is not None:
            C = event.shape[0]
            event_out = np.empty((C, H, W), dtype=event.dtype)
            for c in range(C):
                cropped_event = event[c, y1:y1 + new_H, x1:x1 + new_W]
                event_out[c] = cv2.resize(cropped_event, (W, H), interpolation=cv2.INTER_NEAREST)

        # bboxスケーリング
        new_labels = []
        for label in label_list:
            bbox = label["bbox"]
            x1b = (bbox[0] - x1) * scale
            y1b = (bbox[1] - y1) * scale
            x2b = (bbox[2] - x1) * scale
            y2b = (bbox[3] - y1) * scale
            x1b = np.clip(x1b, 0, W)
            y1b = np.clip(y1b, 0, H)
            x2b = np.clip(x2b, 0, W)
            y2b = np.clip(y2b, 0, H)
            if x2b > x1b and y2b > y1b:
                new_label = label.copy()
                new_label["bbox"] = [x1b, y1b, x2b, y2b]
                new_labels.append(new_label)

        return img_out, new_labels, event_out


    def zoom_out(self, img, label_list, event, scale, center, H, W):
        new_H, new_W = int(H / scale), int(W / scale)
        img_hwc = np.transpose(img, (1, 2, 0))
        resized = cv2.resize(img_hwc, (new_W, new_H), interpolation=cv2.INTER_CUBIC)

        canvas = np.zeros((H, W, 3), dtype=img.dtype)
        cx, cy = center
        x1 = max(min(cx - new_W // 2, W - new_W), 0)
        y1 = max(min(cy - new_H // 2, H - new_H), 0)
        canvas[y1:y1 + new_H, x1:x1 + new_W] = resized
        img_out = np.transpose(canvas, (2, 0, 1))

        # event 処理
        event_out = None
        if event is not None:
            C = event.shape[0]
            event_canvas = np.zeros((C, H, W), dtype=event.dtype)
            for c in range(C):
                resized_event = cv2.resize(event[c], (new_W, new_H), interpolation=cv2.INTER_NEAREST)
                event_canvas[c, y1:y1 + new_H, x1:x1 + new_W] = resized_event
            event_out = event_canvas

        # bboxスケーリング
        new_labels = []
        for label in label_list:
            bbox = label["bbox"]
            x1b = bbox[0] / scale + x1
            y1b = bbox[1] / scale + y1
            x2b = bbox[2] / scale + x1
            y2b = bbox[3] / scale + y1
            x1b = np.clip(x1b, 0, W)
            y1b = np.clip(y1b, 0, H)
            x2b = np.clip(x2b, 0, W)
            y2b = np.clip(y2b, 0, H)
            if x2b > x1b and y2b > y1b:
                new_label = label.copy()
                new_label["bbox"] = [x1b, y1b, x2b, y2b]
                new_labels.append(new_label)

        return img_out, new_labels, event_out   

class ZoomPerSequence:
    def __init__(self,
                 zoom_type: str = None,
                 scale: float = None,
                 center: Tuple[int, int] = None,
                 prob_weight=(8, 2),
                 in_scale=(1.0, 1.5),
                 out_scale=(1.0, 1.2),
                 center_margin_ratio=0.2,
                 seed: int = None):
        """
        Streaming 用のズーム：同じシーケンスには同じズームを適用
        """
        self.zoom_type = zoom_type
        self.scale = scale
        self.center = center
        self.prob_weight = prob_weight
        self.in_scale = in_scale
        self.out_scale = out_scale
        self.center_margin_ratio = center_margin_ratio
        self.rng = random.Random(seed)

        # RandomZoomのインスタンスを使う
        self._zoom_helper = RandomZoom(
            prob_weight=prob_weight,
            in_scale=in_scale,
            out_scale=out_scale,
            center_margin_ratio=center_margin_ratio
        )

    def _choose_zoom_params(self, H, W, labels):
        if self.zoom_type is not None and self.scale is not None and self.center is not None:
            return  # すでに固定されていれば何もしない

        self.zoom_type = self.rng.choices(["in", "out"], weights=self.prob_weight, k=1)[0]
        self.scale = self.rng.uniform(*(self.in_scale if self.zoom_type == "in" else self.out_scale))

        center = _find_zoom_center(labels)
        if center is None:
            margin_x = int(W * self.center_margin_ratio)
            margin_y = int(H * self.center_margin_ratio)
            cx = self.rng.randint(margin_x, W - margin_x)
            cy = self.rng.randint(margin_y, H - margin_y)
            self.center = (cx, cy)
        else:
            self.center = center

    def __call__(self, inputs: dict) -> dict:
        with Timer("ZoomPerSequence"):
            images = inputs["images"]  # [T, C, H, W]
            labels = inputs["labels"]  # list of list of dict
            events = inputs.get("events", None)  # [T, C, H, W] または None
            T, C, H, W = images.shape

            self._choose_zoom_params(H, W, labels[0])

            for t in range(T):
                if self.zoom_type == "in":
                    img_out, label_out, event_out = self._zoom_helper.zoom_in(
                        images[t], labels[t],
                        events[t] if events is not None else None,
                        self.scale, self.center, H, W
                    )
                else:
                    img_out, label_out, event_out = self._zoom_helper.zoom_out(
                        images[t], labels[t],
                        events[t] if events is not None else None,
                        self.scale, self.center, H, W
                    )

                images[t] = img_out
                labels[t] = label_out
                if events is not None:
                    events[t] = event_out

            inputs["images"] = images
            inputs["labels"] = labels
            if events is not None:
                inputs["events"] = events

        return inputs
