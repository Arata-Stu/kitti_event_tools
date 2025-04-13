from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from .sequence_rnd import SequenceForRandom

# ----- 複数シーケンス統合の RandomDataset クラス -----
class RandomConcatDataset(Dataset):
    """
    IntegratedSequenceRandomDataset は、シーケンスID "0000" ～ "0020" の
    複数シーケンスを統合したランダムアクセス用データセットです。
    
    各シーケンスは SequenceRandom として読み込まれ、内部では torch の ConcatDataset により
    統合されます。返り値は各サンプルごとに
      {
         "images": [seq_len 枚の画像],
         "labels": [各画像のラベル情報リスト],
         "events": numpy 配列,
         "reset_state": bool
      }
    の辞書型となります。
    """
    def __init__(self, data_dir: Path, ev_repr_name: str, seq_len: int, seq_ids=None, downsample: bool = False):
        # seq_ids が指定されなければ "0000"～"0020" とする
        if seq_ids is None:
            seq_ids = [f"{i:04d}" for i in range(21)]
        self.sequences = [
            SequenceForRandom(data_dir, seq_id, ev_repr_name, seq_len=seq_len, downsample=downsample)
            for seq_id in seq_ids
        ]
        self.concat_dataset = ConcatDataset(self.sequences)
    
    def __len__(self):
        return len(self.concat_dataset)
    
    def __getitem__(self, idx):
        return self.concat_dataset[idx]