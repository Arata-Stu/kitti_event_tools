# test.py
import sys
sys.path.append("..")
from src.data.stream_dataset import build_stream_dataset
from src.data.utils.multi_stream_sampler import MultiStreamSampler
from torch.utils.data import DataLoader
from pathlib import Path

def main():
    

    data_dir = Path("/Users/at/dataset/mini_kitti")
    ev_repr_name = "accum_10000_histogram"
    seq_len = 5
    seq_ids = ["0000"]

    datasets = build_stream_dataset(data_dir, ev_repr_name, seq_len, seq_ids, downsample=True)
    sampler = MultiStreamSampler(datasets, batch_size=4)
    loader = DataLoader(sampler, batch_size=None, num_workers=4)

    for batch in loader:
        print(batch)  # or your eval/train loop


if __name__ == "__main__":
    main()
