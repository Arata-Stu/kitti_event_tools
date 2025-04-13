from pathlib import Path

from src.data.stream_dataset import StreamingConcatDataset
from src.data.sequence_stream import SequenceForStreaming
from src.data.rnd_dataset import RandomConcatDataset

def build_stream_dataset(data_dir: Path, seq_names: list, ev_repr_name: str, seq_len: int, downsample: bool = False):
    """
    指定した複数のシーケンス（シーケンス名のリスト）を連結して、
    ストリーミングデータセットを作成するメソッドです。
    
    Parameters:
      data_dir (Path): データセットのルートディレクトリ（例: kitti ディレクトリ）
      seq_names (list): 使用するシーケンス名のリスト（例: ["0000", "0001", "0002"]）
      ev_repr_name (str): イベント表現ファイルが置かれているサブディレクトリの名前
      seq_len (int): サンプルごとに連続して取得するフレーム数
      downsample (bool): True の場合、画像を1/2サイズにリサイズし、
                         対応するラベルの bbox も1/2に変換する
      
    Returns:
      StreamingConcatDataset: 複数シーケンスを連結したストリーミングデータセット
    """
    # 各シーケンスについて、SequenceForStreaming のインスタンスを生成
    streaming_sequences = [
        SequenceForStreaming(data_dir, seq_name, ev_repr_name, seq_len, downsample=downsample)
        for seq_name in seq_names
    ]
    # 複数のシーケンスを連結したストリームとして扱うクラスを返す
    return StreamingConcatDataset(streaming_sequences)


def build_random_dataset(data_dir: Path, ev_repr_name: str, seq_len: int, seq_ids: list = None, downsample: bool = False):
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
      RandomConcatDataset: 複数シーケンスを ConcatDataset で統合したランダムアクセス用データセット
    """
    # seq_ids が指定されなければ、"0000"～"0020" のデフォルトリストを使用
    if seq_ids is None:
        seq_ids = [f"{i:04d}" for i in range(21)]
    # RandomConcatDataset のインスタンスを返す
    return RandomConcatDataset(data_dir, ev_repr_name, seq_len, seq_ids, downsample=downsample)
