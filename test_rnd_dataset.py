import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

from src.utils.visualize import draw_labels_on_image, ev_repr_to_img
from src.data.dataset import build_random_dataset

if __name__ == '__main__':
    # build_random_dataset() は既に実装済みとし、RandomConcatDataset を返す関数です。
    # ※ 下記パラメータは各自のデータセット環境に合わせてください。
    data_dir = Path("/path/to/kitti_dataset")      # データセットのルートディレクトリ
    ev_repr_name = "default_ev_repr"                # イベント表現データが保存されているサブディレクトリ名
    seq_len = 5                                    # 各サンプルを構成する連続フレーム数
    
    # ランダムアクセス用データセットの生成
    random_dataset = build_random_dataset(data_dir, ev_repr_name, seq_len)
    print("Random dataset length:", len(random_dataset))
    
    # 例として、先頭サンプルを取得
    sample = random_dataset[0]
    # sample は辞書型で以下のキーを持つ
    # "images": [seq_len 枚のRGB画像],
    # "labels": 各フレームごとに対応するラベル（リスト）,
    # "events": numpy 配列 (形状: (seq_len, ch, H, W) と仮定),
    # "reset_state": bool
    
    print("Sample keys:", sample.keys())
    print("Reset state:", sample["reset_state"])
    
    images = sample["images"]
    labels_seq = sample["labels"]
    events = sample["events"]
    
    num_frames = len(images)
    
    plt.figure(figsize=(15, 8))
    for i in range(num_frames):
        # カメラ画像にラベル描画
        rgb_img = images[i]
        rgb_img_with_boxes = draw_labels_on_image(rgb_img, labels_seq[i])
        
        # イベント表現から画像に変換
        # events[i] は (ch, H, W) の形状と仮定
        ev_img = ev_repr_to_img(events[i])
        # イベント画像にも同じラベルを描画（必要なら用途に合わせ調整）
        ev_img_with_boxes = draw_labels_on_image(ev_img, labels_seq[i])
        
        # 1フレームにつき、左がRGB画像、右がイベント画像として表示
        plt.subplot(num_frames, 2, i*2 + 1)
        plt.imshow(rgb_img_with_boxes)
        plt.title(f"Frame {i} - RGB")
        plt.axis("off")
        
        plt.subplot(num_frames, 2, i*2 + 2)
        plt.imshow(ev_img_with_boxes)
        plt.title(f"Frame {i} - Events")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
