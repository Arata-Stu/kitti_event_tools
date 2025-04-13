import sys
sys.path.append("..")  # 親ディレクトリをパスに追加

import matplotlib.pyplot as plt
from pathlib import Path

# 以下は可視化用関数（src/utils/visualize.py）に定義されたものと仮定します
from src.utils.visualize import draw_labels_on_image, ev_repr_to_img
# データセット生成関数（src/data/dataset.py）に定義された build_random_dataset, build_stream_dataset をインポート
from src.data.dataset import build_random_dataset, build_stream_dataset

if __name__ == '__main__':
    # テスト用のパラメータ（各自のデータパスに合わせて変更してください）
    data_dir = Path("/path/to/kitti_dataset")  # 例: KITTI のルートディレクトリ
    ev_repr_name = "default_ev_repr"            # イベント表現のサブディレクトリ名
    seq_len = 5                                 # 各サンプルを構成する連続フレーム数

    # ストリーミング版データセットの生成
    # build_stream_dataset の引数として、seq_ids が None の場合は内部でデフォルトが利用されるか、
    # また downsample のフラグでオリジナルデータとサイズ違いのデータが利用されます（ここでは downsample=False）
    streaming_dataset = build_stream_dataset(data_dir=data_dir,
                                             ev_repr_name=ev_repr_name,
                                             seq_len=seq_len,
                                             seq_ids=None,
                                             downsample=False)

    # streaming_dataset はイテレータですので、連続していくつかのサンプルを取得して検証します。
    num_samples_to_show = 5  # 例として最初の5サンプルを取得
    stream_iter = iter(streaming_dataset)
    
    all_samples = []
    for _ in range(num_samples_to_show):
        try:
            sample = next(stream_iter)
            all_samples.append(sample)
        except StopIteration:
            break

    print(f"取得したサンプル数: {len(all_samples)}")
    if len(all_samples) == 0:
        print("ストリーミングデータセットからサンプルが取得できませんでした。")
        sys.exit(1)

    # 各サンプルについて、その中の連続フレームの画像とイベントを可視化する
    for sample_idx, sample in enumerate(all_samples):
        print(f"Sample {sample_idx} keys:", sample.keys())
        print(f"Sample {sample_idx} reset_state:", sample["reset_state"])
        
        num_frames = len(sample["images"])
        plt.figure(figsize=(15, 3 * num_frames))
        plt.suptitle(f"Sample {sample_idx}", fontsize=16)
        for i in range(num_frames):
            # カメラ画像にラベル描画
            rgb_img = sample["images"][i]
            rgb_img_with_boxes = draw_labels_on_image(rgb_img, sample["labels"][i])
            
            # イベント表現を画像に変換し、同じくラベルを描画
            ev_img = ev_repr_to_img(sample["events"][i])
            ev_img_with_boxes = draw_labels_on_image(ev_img, sample["labels"][i])
            
            # 1フレームにつき、上段にRGB、下段にイベント画像を表示
            plt.subplot(num_frames, 2, i*2 + 1)
            plt.imshow(rgb_img_with_boxes)
            plt.title(f"Frame {i} - RGB")
            plt.axis("off")
            
            plt.subplot(num_frames, 2, i*2 + 2)
            plt.imshow(ev_img_with_boxes)
            plt.title(f"Frame {i} - Events")
            plt.axis("off")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
