import numpy as np
from pathlib import Path
import h5py
try:
    import hdf5plugin
except ImportError:
    pass
from torch.utils.data import IterableDataset, get_worker_info

from .sequence_base import SequenceBase

class SequenceForStreaming(SequenceBase):
    """
    SequenceBase を利用して、シーケンス内の連続サンプルを逐次 yield するストリーミング版データセットです。
    
    1回のイテレーションで、シーケンス内の全連続した seq_len フレームのサンプルを順番に生成します。
    """
    def __init__(self, data_dir: Path, sequence_name: str, ev_repr_name: str, seq_len: int, downsample: bool = False, transform=None):
        """
        コンストラクタで downsample 引数を受け取り、親クラスの初期化時に渡す。
        """
        super().__init__(data_dir, sequence_name, ev_repr_name, seq_len=seq_len, downsample=downsample, transform=transform)

    def __iter__(self):
        # 🔄 Worker ID を取得して transform を再構築
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        if hasattr(self.transform, "rebuild"):
            self.transform = self.transform.rebuild(self.sequence_name, worker_id)

        # 🔁 通常のstreamingループ
        with h5py.File(str(self.event_file), 'r') as f:
            index = 0
            while index + self.seq_len <= self.total_frames:
                images = []
                labels_seq = []
                for i in range(index, index + self.seq_len):
                    img = self.load_image(i)
                    images.append(img)
                    labels_seq.append(self.labels.get(i, []))
                images = np.stack(images, axis=0)  # [seq_len, C, H, W]

                events = f["data"][index : index + self.seq_len]
                reset_state = (index == 0)

                outputs = {
                    "images": images,
                    "labels": labels_seq,
                    "events": events,
                    "reset_state": reset_state
                }

                if self.transform:
                    outputs = self.transform(outputs)

                yield outputs

                index += self.seq_len


class StreamingSequenceDataset(IterableDataset):
    def __init__(self, sequence: SequenceForStreaming):
        self.sequence = sequence

    def __iter__(self):
        return iter(self.sequence)