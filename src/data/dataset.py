from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from src.data.stream_dataset import (
    ConcatStreamingDataset,
    SharedStreamingDataset,
)
from src.data.utils.collate import (
    custom_collate_rnd,
    custom_collate_streaming,
)
from src.data.sequence_map import SequenceForMap


def build_stream_dataset(dataset_mode: str, dataset_config: Dict[str, Any],
                         batch_size: int, num_workers: int) -> DataLoader:
    """
    データセットのルートディレクトリ配下の、train/val/test 各モードのシーケンスデータから
    ストリーミング用 DataLoader を構築する関数です。

    Parameters:
        dataset_mode (str): 'train', 'val', 'test' のいずれか
        dataset_config (dict): 次のキーを含む設定辞書
            - path: データセットのルートディレクトリ
            - ev_repr_name: イベント表現の名前
            - sequence_length: シーケンス長
            - downsample: ダウンサンプリングの有無（bool）
            - transform: (optional) サンプルに適用する変換関数
        batch_size (int): バッチサイズ（内部で streaming dataset によりバッチが作られるので、DataLoader の batch_size は None）
        num_workers (int): DataLoader に割り当てるワーカー数

    Returns:
        DataLoader: ストリーミングデータローダ
    """
    dataset_path = Path(dataset_config["path"])
    # モード文字列のマッピング（例として train/val/test のサブディレクトリがある前提）
    mode_mapping = {"train": "train", "val": "val", "test": "test"}
    mode_str = mode_mapping.get(dataset_mode.lower())
    if mode_str is None:
        raise ValueError(f"dataset_mode は 'train', 'val', 'test' のいずれかでなければなりません。: {dataset_mode}")
    
    mode_dir = dataset_path / mode_str
    if not mode_dir.is_dir():
        raise FileNotFoundError(f"指定されたモードのディレクトリが存在しません: {mode_dir}")

    # 各シーケンスは mode_dir 配下のサブディレクトリとして管理されていると仮定
    sequence_dirs = [p for p in mode_dir.iterdir() if p.is_dir()]
    sequence_datasets = []
    for seq_dir in sequence_dirs:
        # SequenceForMap の初期化時は、データセットのルートディレクトリとシーケンス名、
        # ev_repr_name、sequence_length、downsampleオプション、必要に応じ transform を渡す
        seq_ds = SequenceForMap(
            data_dir=dataset_path,
            sequence_name=seq_dir.name,
            ev_repr_name=dataset_config["ev_repr_name"],
            seq_len=dataset_config["sequence_length"],
            downsample=dataset_config.get("downsample", False),
            transform=dataset_config.get("transform", None)
        )
        sequence_datasets.append(seq_ds)

    # 学習と評価/テストで利用するストリーミングデータセットを切り分け
    if dataset_mode.lower() == "train":
        # ConcatStreamingDataset は複数シーケンスからランダムにサンプルを得て、無限に生成する設計
        streaming_dataset = ConcatStreamingDataset(
            sequence_datasets=sequence_datasets,
            batch_size=batch_size,
            pad_last=True,
            # 各シーケンスは get_padding_sample() メソッドでパディング用のサンプルを返すように実装してください
            pad_func=lambda: sequence_datasets[0].get_padding_sample(),
            transform=dataset_config.get("transform", None)
        )
        collate_fn = custom_collate_rnd
    else:
        # SharedStreamingDataset は各ワーカーにシーケンスを割り当て、重複なく有限のバッチを生成します
        streaming_dataset = SharedStreamingDataset(
            sequence_datasets=sequence_datasets,
            batch_size=batch_size,
            pad_last=True,
            pad_func=lambda: sequence_datasets[0].get_padding_sample(),
            transform=dataset_config.get("transform", None)
        )
        collate_fn = custom_collate_streaming

    # DataLoader では、streaming dataset 自体がバッチを yield するため、
    # DataLoader の batch_size は None に設定します。
    dataloader = DataLoader(
        streaming_dataset,
        batch_size=None,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return dataloader
