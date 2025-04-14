import numpy as np
from pathlib import Path
import h5py
try:
    import hdf5plugin
except ImportError:
    pass
from torch.utils.data import get_worker_info

from .sequence_base import SequenceBase

class SequenceForStreaming(SequenceBase):
    def __init__(self, data_dir: Path, sequence_name: str, ev_repr_name: str,
                 seq_len: int, downsample: bool = False, transform=None):
        super().__init__(data_dir, sequence_name, ev_repr_name, seq_len=seq_len,
                         downsample=downsample, transform=transform)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0

        if hasattr(self.transform, "rebuild"):
            self.transform = self.transform.rebuild(self.sequence_name, worker_id)

        index = 0
        while index + self.seq_len <= self.total_frames:
            # --- images + labels ---
            images = []
            labels_seq = []
            for i in range(index, index + self.seq_len):
                img = self.load_image(i)  # numpy [C, H, W]
                images.append(img)
                labels_seq.append(self.labels.get(i, []))
            images = np.stack(images, axis=0)  # [T, C, H, W]

            # --- events (安全なh5py読み出し) ---
            with h5py.File(str(self.event_file), 'r') as f:
                events_np = np.array(f["data"][index : index + self.seq_len])  # copyして閉じる

            # --- オプション：即座にTensor化してpin memory考慮するなら以下でもOK ---
            # events = torch.from_numpy(events_np).float()  # optional

            reset_state = (index == 0)
            outputs = {
                "images": images,
                "labels": labels_seq,
                "events": events_np,  # or events
                "reset_state": reset_state
            }

            if self.transform:
                outputs = self.transform(outputs)

            yield outputs
            index += self.seq_len
