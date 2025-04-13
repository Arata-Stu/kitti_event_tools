from pathlib import Path
from src.data.stream_dataset import StreamingConcatDataset
from src.data.sequence_stream import SequenceForStreaming
from src.data.sequence_rnd import SequenceForRandom
from src.data.rnd_dataset import RandomConcatDataset

def build_stream_dataset(data_dir: Path,
                         ev_repr_name: str,
                         seq_len: int,
                         seq_ids: list = None,
                         downsample: bool = False,
                         transform=None):
    """
    指定した複数シーケンス（シーケンスID）のストリーミングデータセットを作成するメソッドです。
    
    Parameters:
      data_dir (Path): データセットのルートディレクトリ（例: kitti ディレクトリ）
      ev_repr_name (str): イベント表現ファイルが置かれているサブディレクトリの名前
      seq_len (int): サンプルごとに連続して取得するフレーム数
      seq_ids (list, optional): 使用するシーケンスIDのリスト。指定がなければ、"0000"～"0020" が使われる
      downsample (bool): True の場合、画像を1/2サイズにリサイズし、
                         対応するラベルの bbox も1/2に変換する
      
    Returns:
      StreamingConcatDataset: 複数シーケンスを連結したストリーミングデータセット
    """
    # seq_ids が指定されなければ、"0000"～"0020" のデフォルトリストを使用
    if seq_ids is None:
        seq_ids = [f"{i:04d}" for i in range(21)]
    
    # 各シーケンスについて、SequenceForStreaming のインスタンスを生成
    streaming_sequences = [
        SequenceForStreaming(data_dir, seq_id, ev_repr_name, seq_len, downsample=downsample, transform=transform)
        for seq_id in seq_ids
    ]
    
    # 複数のシーケンスを連結したストリームとして扱うクラスを返す
    return StreamingConcatDataset(streaming_sequences)



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
