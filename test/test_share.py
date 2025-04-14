# test_sharded.py
import sys
sys.path.append("..")
from omegaconf import OmegaConf
from src.data.dataset import build_stream_dataset
from src.data.utils.sharded_stream_sampler import ShardedSequenceSampler
from torch.utils.data import DataLoader
from pathlib import Path

def main():
    config_path = "../config/test.yaml"
    cfg = OmegaConf.load(config_path)
    seq_ids = ["0000"]  # シーケンス複数で分配確認

    datasets = build_stream_dataset(dataset_mode="train", seq_ids=seq_ids, dataset_cfg=cfg.dataset)
    sampler = ShardedSequenceSampler(datasets, batch_size=2)  # シンプルにbatch=2で確認
    loader = DataLoader(sampler, batch_size=None, num_workers=2)

    for i, batch in enumerate(loader):
        print(f"\n[Batch {i}]")
        print(batch)
        if i > 5:  # サンプルだけで止めたいとき
            break

if __name__ == "__main__":
    main()
