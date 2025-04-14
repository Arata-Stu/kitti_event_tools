from pathlib import Path
from data.sequence_map import SequenceForMap
from torch.utils.data import ConcatDataset

def get_seq_ids(mode: str):
    if mode == "train":
        return [f"{i:04d}" for i in range(17)]
    else:
        return [f"{i:04d}" for i in range(17, 21)]

def build_random_dataset(data_dir, ev_repr_name, seq_len, seq_ids,
                         downsample=False, transform=None):
    return ConcatDataset([
        SequenceForMap(
            data_dir=data_dir,
            sequence_name=seq_id,
            ev_repr_name=ev_repr_name,
            seq_len=seq_len,
            downsample=downsample,
            transform=transform
        )
        for seq_id in seq_ids
    ])


# 🌀 ストリーム用（各シーケンスを個別に返す）
def build_stream_datasets(data_dir: Path,
                           ev_repr_name: str,
                           seq_len: int,
                           seq_ids: list[str],
                           downsample: bool = False,
                           transform=None):
    return [
        SequenceForMap(
            data_dir=data_dir,
            sequence_name=seq_id,
            ev_repr_name=ev_repr_name,
            seq_len=seq_len,
            downsample=downsample,
            transform=transform
        )
        for seq_id in seq_ids
    ]