from omegaconf import DictConfig
from pathlib import Path
from src.data.sequence_stream import SequenceForStreaming, StreamingSequenceDataset
from src.data.utils.transform_factory import TransformFactory


def build_stream_dataset(dataset_mode: str,
                         seq_ids: list,
                         dataset_cfg: DictConfig) -> list:
    
    data_path = Path(dataset_cfg.path)
    ev_repr_name = dataset_cfg.ev_repr_name
    seq_len = dataset_cfg.sequence_length
    downsample = dataset_cfg.downsample
    transform_cfg = dataset_cfg.transform
    transform_factory = TransformFactory(dataset_mode, transform_cfg)

    if seq_ids is None:
        seq_ids = [f"{i:04d}" for i in range(21)]

    datasets = []
    for seq_id in seq_ids:
        transform = transform_factory.build_for_stream(seq_id)
        seq = SequenceForStreaming(data_path, seq_id, ev_repr_name, seq_len, downsample=downsample, transform=transform)
        ds = StreamingSequenceDataset(seq)
        datasets.append(ds)

    return datasets  # train/testで使い分ける