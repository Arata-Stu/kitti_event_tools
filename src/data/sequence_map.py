import numpy as np
from pathlib import Path
import h5py
import cv2
from torch.utils.data import Dataset
from collections import defaultdict


class SequenceForMap(Dataset):
    def __init__(self, data_dir: Path, sequence_name: str, ev_repr_name: str,
                 seq_len: int, downsample: bool = False, transform=None):
        self.data_dir = data_dir
        self.sequence_name = sequence_name
        self.ev_repr_name = ev_repr_name
        self.seq_len = seq_len
        self.downsample = downsample
        self.transform = transform

        self.images_dir = self.data_dir / "images" / sequence_name
        self.labels_file = self.data_dir / "labels" / f"{sequence_name}.txt"
        self.events_dir = self.data_dir / "preprocessed" / ev_repr_name
        self.event_file = self.events_dir / f"{sequence_name}.h5"

        self.image_files = sorted(self.images_dir.glob("*.png"), key=lambda p: int(p.stem))
        with h5py.File(self.event_file, "r") as f:
            self.num_event_frames = f["data"].shape[0]

        self.total_frames = min(len(self.image_files), self.num_event_frames)
        self.length = self.total_frames - seq_len + 1

        self.labels = self._load_labels()

    def __len__(self):
        return self.length

    def _load_labels(self):
        if not self.labels_file.exists():
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {self.labels_file}")

        labels_per_frame = defaultdict(list)
        with open(self.labels_file, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if not fields:
                    continue
                try:
                    frame = int(fields[0])
                    bbox = list(map(float, fields[6:10]))
                    if self.downsample:
                        bbox = [coord / 2 for coord in bbox]

                    label = {
                        "track_id": int(fields[1]),
                        "type": fields[2],
                        "truncated": float(fields[3]),
                        "occluded": int(fields[4]),
                        "alpha": float(fields[5]),
                        "bbox": bbox,
                        "dimensions": list(map(float, fields[10:13])),
                        "location": list(map(float, fields[13:16])),
                        "rotation_y": float(fields[16])
                    }
                    labels_per_frame[frame].append(label)
                except Exception as e:
                    raise ValueError(f"ラベル行のパースに失敗しました: {line}\nエラー: {e}")
        return labels_per_frame

    def _load_image(self, index: int):
        path = self.image_files[index]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.downsample:
            h, w = img.shape[:2]
            img = cv2.resize(img, (w // 2, h // 2))
        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        return img

    def __getitem__(self, index: int):
        images = []
        labels_seq = []
        for i in range(index, index + self.seq_len):
            images.append(self._load_image(i))
            labels_seq.append(self.labels.get(i, []))  # ラベルがない場合は空リスト

        images = np.stack(images)  # [T, C, H, W]

        with h5py.File(self.event_file, 'r') as f:
            events = np.array(f["data"][index : index + self.seq_len])

        sample = {
            "images": images,
            "labels": labels_seq,
            "events": events,
            "reset_state": index == 0
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
