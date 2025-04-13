import random
import numpy as np
import cv2
from einops import rearrange, reduce

def ev_repr_to_img(input: np.ndarray):
    """
    イベント表現 (正の極性と負の極性) を RGB 画像に変換します。
    """
    ch, ht, wd = input.shape[-3:]
    assert ch > 1 and ch % 2 == 0, "Input channels must be a positive even number."
    ev_repr_reshaped = rearrange(input, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img

# ---------------------------
# 画像にラベル（2D boxとトラックID）を描画する関数
# ---------------------------

def draw_labels_on_image(image: np.ndarray, labels: list):
    """
    image: (H, W, 3) のRGB画像
    labels: 各ラベルは辞書形式で、少なくとも "bbox" (左, 上, 右, 下) と "track_id" を含むとする。
    """
    image_with_boxes = image.copy()
    for label in labels:
        bbox = label.get("bbox", None)
        track_id = label.get("track_id", None)
        if bbox is not None and track_id is not None:
            # bbox の値が float の場合は整数に変換
            x1, y1, x2, y2 = list(map(int, bbox))
            # トラックIDに基づいて色を取得
            color = get_color_for_id(track_id)
            # バウンディングボックスの描画（取得した色を使用、線幅2）
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness=2)
            # トラックIDのテキスト描画（白色で表示）
            cv2.putText(image_with_boxes, f"ID:{track_id}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
    return image_with_boxes


def get_color_for_id(track_id):
    random.seed(track_id)  # トラックIDから一意の色を決定
    return tuple(random.randint(0, 255) for _ in range(3))  # BGR