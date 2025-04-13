from torch.utils.data import Dataset
from pathlib import Path

from .sequence_base import SequenceBase
# ----- SequenceRandom クラス -----
class SequenceRandom(SequenceBase, Dataset):
    """
    SequenceRandom はランダムアクセスを前提とするデータセットです。
    連続した seq_len フレームのサンプルが取得できるようにし、
    辞書型 ({"images", "labels", "events", "reset_state"}) で返します。
    
    ※ DataLoader のシャッフルにより、シーケンス内の任意の位置から連続サンプルが取得されます。
    """
    def __init__(self, data_dir: Path, sequence_name: str, ev_repr_name: str, seq_len: int):
        super().__init__(data_dir, sequence_name, ev_repr_name, seq_len=seq_len)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
