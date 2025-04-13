from pathlib import Path
from src.data.sequence_stream import SequenceForStreaming, StreamingSequenceDataset
from src.data.utils.transform_factory import make_transform_for_sequence


def build_stream_dataset(data_dir: Path,
                         ev_repr_name: str,
                         seq_len: int,
                         seq_ids: list = None,
                         downsample: bool = False):
    if seq_ids is None:
        seq_ids = [f"{i:04d}" for i in range(21)]

    datasets = []
    for seq_id in seq_ids:
        transform = make_transform_for_sequence(seq_id)
        seq = SequenceForStreaming(data_dir, seq_id, ev_repr_name, seq_len, downsample=downsample, transform=transform)
        ds = StreamingSequenceDataset(seq)
        datasets.append(ds)

    return datasets  # train/testで使い分ける