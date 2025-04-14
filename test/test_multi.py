# test.py
import sys
sys.path.append("..")
from omegaconf import OmegaConf
from src.data.dataset import build_stream_dataset
from src.data.utils.multi_stream_sampler import MultiStreamSampler
from torch.utils.data import DataLoader
from pathlib import Path

def main():
    

    config_path = "../config/test.yaml"
    cfg = OmegaConf.load(config_path)
    seq_ids = ["0000"]  # シーケンス複数で分配確認

    datasets = build_stream_dataset(dataset_mode="train", seq_ids=seq_ids, dataset_cfg=cfg.dataset)
    sampler = MultiStreamSampler(datasets, batch_size=4)
    loader = DataLoader(sampler, batch_size=None, num_workers=4)

    for batch in loader:
        print(batch)  # or your eval/train loop


if __name__ == "__main__":
    main()
