from typing import Literal
from data.dataset import build_random_dataset, build_stream_datasets
from data.utils.multi_stream_sampler import MultiStreamSampler
from data.utils.sharded_stream_sampler import ShardedSequenceSampler
from data.utils.collate import custom_collate_rnd, custom_collate_streaming
from torch.utils.data import DataLoader, get_worker_info

def get_seq_ids(mode: str):
    if mode == "train":
        return [f"{i:04d}" for i in range(17)]
    else:
        return [f"{i:04d}" for i in range(17, 21)]

def build_random_dataloader(mode: Literal["train", "val", "test"], cfg):
    seq_ids = get_seq_ids(mode)

    dataset = build_random_dataset(
        data_dir=cfg.data_dir,
        ev_repr_name=cfg.ev_repr_name,
        seq_len=cfg.seq_len,
        seq_ids=seq_ids,
        downsample=cfg.get("downsample", False),
        transform=cfg.get("transform", None),
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size.train if mode == "train" else cfg.batch_size.eval,
        shuffle=(mode == "train"),
        drop_last=(mode == "train"),
        num_workers=cfg.hardware.num_workers.train if mode == "train" else cfg.hardware.num_workers.eval,
        pin_memory=True,
        collate_fn=custom_collate_rnd,
    )

def build_stream_dataloader(mode: Literal["train", "val", "test"],
                             cfg) -> DataLoader:
    seq_ids = get_seq_ids(mode)

    datasets = build_stream_datasets(
        data_dir=cfg.data_dir,
        ev_repr_name=cfg.ev_repr_name,
        seq_len=cfg.seq_len,
        seq_ids=seq_ids,
        downsample=cfg.get("downsample", False),
        transform=cfg.get("transform", None),
    )

    # Sampler selection
    if mode == "train":
        sampler = MultiStreamSampler(datasets, batch_size=cfg.batch_size.train)
    else:
        sampler = ShardedSequenceSampler(datasets, batch_size=cfg.batch_size.eval)

    def _wrapped_iter():
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        for batch in sampler:
            yield batch, worker_id

    return DataLoader(
        dataset=_wrapped_iter(),
        batch_size=None,
        num_workers=cfg.hardware.num_workers.train if mode == "train" else cfg.hardware.num_workers.eval,
        pin_memory=True,
        collate_fn=custom_collate_streaming,
    )
