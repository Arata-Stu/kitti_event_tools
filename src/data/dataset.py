from pathlib import Path
from omegaconf import DictConfig

from src.data.utils.transform_factory import TransformFactory
from src.data.stream_dataset import StreamingConcatDataset
from src.data.sequence_stream import SequenceForStreaming
from src.data.sequence_rnd import SequenceForRandom
from src.data.rnd_dataset import RandomConcatDataset


def build_stream_dataset(dataset_mode: str,
                         seq_ids: list,
                         dataset_cfg: DictConfig) -> list:
    
    data_path = Path(dataset_cfg.path)
    ev_repr_name = dataset_cfg.ev_repr_name
    seq_len = dataset_cfg.sequence_length
    downsample = dataset_cfg.downsample
    transform_cfg = dataset_cfg.transform
    transform_factory = TransformFactory(dataset_mode, transform_cfg)

    if seq_ids is None:
        seq_ids = [f"{i:04d}" for i in range(21)]

    datasets = []
    for seq_id in seq_ids:
        transform = transform_factory.build_for_stream(seq_id)
        seq = SequenceForStreaming(data_path, seq_id, ev_repr_name, seq_len, downsample=downsample, transform=transform)
        ds = StreamingConcatDataset(seq)
        datasets.append(ds)

    return datasets  # train/testで使い分ける


def build_random_dataset(data_dir: Path,
                         ev_repr_name: str,
                         seq_len: int,
                         seq_ids: list = None,
                         downsample: bool = False,
                         transform=None):
    """
    指定した複数シーケンス（シーケンスID）のランダムアクセス用データセットを作成するメソッドです。

    Parameters:
      data_dir (Path): データセットのルートディレクトリ
      ev_repr_name (str): イベント表現ファイルが置かれているサブディレクトリの名前
      seq_len (int): サンプルごとに連続して取得するフレーム数
      seq_ids (list, optional): 使用するシーケンスIDのリスト。指定がなければ、"0000"～"0020" が使われる
      downsample (bool): True の場合、画像を1/2サイズにリサイズし、
                         対応するラベルの bbox も1/2に変換する

    Returns:
      RandomConcatDataset: 複数シーケンスを連結したランダムアクセス用データセット
    """
    # seq_ids が指定されなければ、"0000"～"0020" のデフォルトリストを使用
    if seq_ids is None:
        seq_ids = [f"{i:04d}" for i in range(21)]
    
    # 各シーケンスを SequenceForRandom としてインスタンス化する
    sequences = [
        SequenceForRandom(data_dir, seq_id, ev_repr_name, seq_len, downsample, transform)
        for seq_id in seq_ids
    ]
    
    # 複数シーケンスを連結したデータセットを返す
    return RandomConcatDataset(sequences)
