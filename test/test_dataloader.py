import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# ※ここでは build_stream_dataset 関数を stream_dataset.py 内に実装している前提です
from data.util.stream_dataset import build_stream_dataset

def test_stream_data_loader():
    # テスト用の dataset_config（実際のディレクトリパス・イベント表現名などを指定）
    dataset_config = {
        "path": "/path/to/your/dataset",  # dataset_root ディレクトリ（例：images/, labels/, preprocessed/ が含まれる）
        "ev_repr_name": "ev_repr_sample",  # イベント表現のサブディレクトリ名（例：preprocessed/ev_repr_sample）
        "sequence_length": 5,             # シーケンス長（例：5フレームずつの非重複シーケンス）
        "downsample": False,              # ダウンサンプリングの有無
        "transform": None                 # 必要に応じて変換関数を指定
    }
    
    # 動作確認のため、train モードの streaming dataset を構築（学習用は ConcatStreamingDataset）
    batch_size = 2       # お好みのバッチサイズを指定
    num_workers = 0      # テスト時はシングルプロセスで動作確認するとシンプル
    dataloader = build_stream_dataset("train", dataset_config, batch_size, num_workers)

    # 数バッチ分ループして出力を確認する
    for i, batch in enumerate(dataloader):
        print(f"===== Batch {i} =====")
        # custom_collate_rnd を利用している場合、バッチの形式は以下のようになっているはず：
        # {'data': { ... collated sample ... }, 'worker_id': int}
        print("Worker ID:", batch.get("worker_id", "N/A"))
        
        data = batch["data"]
        # data には例えば以下のキーが含まれると想定：
        # "images": Tensor or np.array, "labels": collated labels, "events": Tensor or np.array,
        # "is_padded_mask": list or array, "reset_state": bool
        print("Images shape:", data["images"].shape)
        print("Events shape:", data["events"].shape)
        print("Labels:", data["labels"])
        print("is_padded_mask:", data["is_padded_mask"])
        print("reset_state:", data["reset_state"])
        
        if i >= 2:  # 3バッチ程度確認したら終了
            break

if __name__ == "__main__":
    test_stream_data_loader()
