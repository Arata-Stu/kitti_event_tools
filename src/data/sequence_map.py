import numpy as np
from pathlib import Path
import h5py
import cv2
from torch.utils.data import Dataset
from collections import defaultdict
import math

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

        # 実際に利用できるフレーム数（画像とイベントのどちらも）を決定
        self.total_frames = min(len(self.image_files), self.num_event_frames)
        # オーバーラップなしなので、サンプル数はシーケンス長で割った商+余りがあれば1つ追加
        self.num_samples = math.ceil(self.total_frames / self.seq_len)

        self.labels = self._load_labels()

    def __len__(self):
        return self.num_samples

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

    def get_padding_sample(self):
        """
        パディング用の1フレーム分のサンプルを生成する。
        画像はゼロで埋めた配列、イベントもゼロ配列、ラベルは空リストを返す。
        """
        pad_image = None
        # 実際の画像の形状は先頭フレームから取得
        if len(self.image_files) > 0:
            sample_image = self._load_image(0)
            pad_image = np.zeros_like(sample_image)
        else:
            raise RuntimeError("画像ファイルが存在しません。")
            
        # イベントは h5py から形状を取得
        with h5py.File(self.event_file, "r") as f:
            events_shape = list(f["data"].shape)
        # 1フレーム分のイベントの形状（T が 1）
        pad_events = np.zeros(events_shape[1:], dtype=np.float32)
        pad_labels = []  # ラベルは空リスト
        return {
            "images": pad_image,
            "labels": pad_labels,
            "events": pad_events,
            "reset_state": False
        }

    def __getitem__(self, index: int):
        start = index * self.seq_len
        end = start + self.seq_len

        images = []
        labels_seq = []
        is_padded_mask = []  # 各フレームがパディングかどうかを示すマスク

        # 利用可能なフレームを収集
        for i in range(start, min(end, self.total_frames)):
            images.append(self._load_image(i))
            labels_seq.append(self.labels.get(i, []))
            is_padded_mask.append(False)  # 実データの場合は False

        # フレーム数が不足している場合はパディングを追加
        if len(images) < self.seq_len:
            pad_count = self.seq_len - len(images)
            pad_sample = self.get_padding_sample()
            for _ in range(pad_count):
                images.append(pad_sample["images"])
                labels_seq.append(pad_sample["labels"])
                is_padded_mask.append(True)  # パディング部分は True とする

        images = np.stack(images)

        with h5py.File(self.event_file, 'r') as f:
            if end <= self.total_frames:
                events = np.array(f["data"][start:end])
            else:
                available_events = np.array(f["data"][start:self.total_frames])
                pad_events = [pad_sample["events"]] * (self.seq_len - available_events.shape[0])
                events = np.concatenate([available_events, np.stack(pad_events)], axis=0)

        sample = {
            "images": images,
            "labels": labels_seq,
            "events": events,
            "is_padded_mask": is_padded_mask,  # ここでマスクを付与
            "reset_state": index == 0
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

