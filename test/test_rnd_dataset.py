import sys
sys.path.append("..")  # 親ディレクトリをパスに追加

import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.visualize import draw_labels_on_image, ev_repr_to_img
from src.data.dataset import build_random_dataset, build_stream_dataset

if __name__ == '__main__':
        
    # テスト用のパラメータ（各自のデータパスに合わせて変更してください）
    data_dir = Path("/path/to/kitti_dataset")  # 例: KITTI のルートディレクトリ
    ev_repr_name = "default_ev_repr"            # イベント表現のサブディレクトリ名
    seq_len = 5                                 # 各サンプルを構成する連続フレーム数

    # ストリーミング版データセットの生成
    streaming_dataset = build_stream_dataset(data_dir=data_dir,
                                             ev_repr_name=ev_repr_name,
                                             seq_len=seq_len,
                                             seq_ids=None,
                                             downsample=False)

    # streaming_dataset はイテレータなので、最初のいくつかのサンプルを取り出して可視化する
    # ここでは例として最初のサンプルを取り出します
    stream_iter = iter(streaming_dataset)
    sample = next(stream_iter)

    print("Streaming sample keys:", sample.keys())
    print("Streaming sample reset_state:", sample["reset_state"])

    # サンプルは以下のキーを持っています:
    # "images": [seq_len 枚のRGB画像]
    # "labels": フレームごとのラベルリスト
    # "events": numpy 配列 (形状: (seq_len, ch, H, W) と仮定)
    # "reset_state": bool

    # サンプル内の各フレームの画像とイベントデータを可視化
    num_frames = len(sample["images"])
    plt.figure(figsize=(15, 3 * num_frames))
    for i in range(num_frames):
        # カメラ画像にラベル描画
        rgb_img = sample["images"][i]
        rgb_img_with_boxes = draw_labels_on_image(rgb_img, sample["labels"][i])
        
        # イベント表現を画像に変換し、同じくラベルを描画
        ev_img = ev_repr_to_img(sample["events"][i])
        ev_img_with_boxes = draw_labels_on_image(ev_img, sample["labels"][i])
        
        # 1フレームにつき、上段がRGB、下段がイベント画像となるように表示
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
