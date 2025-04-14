from torch.utils.data import DataLoader, get_worker_info
from omegaconf import DictConfig

from src.data.dataset import build_stream_dataset, build_random_dataset
from torch.utils.data import IterableDataset

from src.data.utils.multi_stream_sampler import MultiStreamSampler
from src.data.utils.sharded_stream_sampler import ShardedSequenceSampler
from src.data.utils.collate import custom_collate_streaming, custom_collate_rnd

def get_seq_ids(dataset_mode: str) -> list:
    if dataset_mode == "train":
        return [f"{i:04d}" for i in range(0, 17)]
    elif dataset_mode in ("val", "test"):
        return [f"{i:04d}" for i in range(17, 21)]
    else:
        raise ValueError(f"Invalid dataset mode: {dataset_mode}")


class WrappedSamplerDataset(IterableDataset):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        for batch in self.sampler:
            yield batch, worker_id


def build_stream_dataloader(dataset_mode: str = "test",
                            dataset_cfg: DictConfig = None,
                            batch_size: int = 4,
                            num_workers: int = 4,
                            pin_memory: bool = True):
    """
    ストリーミング用 DataLoader を構築します。
    """
    seq_ids = get_seq_ids(dataset_mode)

    dataset = build_stream_dataset(
        dataset_mode=dataset_mode,
        seq_ids=seq_ids,
        dataset_cfg=dataset_cfg.data_dir,  # ※ここたぶん dataset_cfg 全体を渡すべき？
    )

    if dataset_mode == "train":
        sampler = MultiStreamSampler(dataset.datasets, batch_size=batch_size)
    else:
        sampler = ShardedSequenceSampler(dataset.datasets, batch_size=batch_size)

    wrapped_dataset = WrappedSamplerDataset(sampler)

    return DataLoader(
        dataset=wrapped_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_streaming,
    )


def build_random_dataloader(dataset_mode: str = "train",
                            dataset_cfg: DictConfig = None,
                            batch_size: int = 4,
                            num_workers: int = 4,
                            shuffle: bool = True,
                            drop_last: bool = True,
                            pin_memory: bool = True):
    """
    ランダムアクセス用の DataLoader を構築します。

    Parameters:
        dataset_mode (str): "train", "val", "test" など
        dataset_cfg (DictConfig): データ設定情報
        batch_size (int): ミニバッチサイズ
        shuffle (bool): データをシャッフルするか（通常は True）
        drop_last (bool): 最後の不完全バッチを捨てるか
        pin_memory (bool): pinned memory を使うか

    Returns:
        DataLoader: PyTorch用の DataLoader オブジェクト
    """
    seq_ids = get_seq_ids(dataset_mode)

    dataset = build_random_dataset(
        data_dir=dataset_cfg.data_dir,
        ev_repr_name=dataset_cfg.ev_repr_name,
        seq_len=dataset_cfg.seq_len,
        seq_ids=seq_ids,
        downsample=dataset_cfg.get("downsample", False),
        transform=dataset_cfg.get("transform", None),
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=custom_collate_rnd,
    )

