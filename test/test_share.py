# test_sharded.py
import sys
sys.path.append("..")
from src.data.stream_dataset import build_stream_dataset
from src.data.utils.sharded_stream_sampler import ShardedSequenceSampler
from torch.utils.data import DataLoader
from pathlib import Path

def main():
    data_dir = Path("/Users/at/dataset/mini_kitti")
    ev_repr_name = "accum_10000_histogram"
    seq_len = 5
    seq_ids = ["0000"]  # シーケンス複数で分配確認

    datasets = build_stream_dataset(data_dir, ev_repr_name, seq_len, seq_ids, downsample=True)
    sampler = ShardedSequenceSampler(datasets, batch_size=2)  # シンプルにbatch=2で確認
    loader = DataLoader(sampler, batch_size=None, num_workers=2)

    for i, batch in enumerate(loader):
        print(f"\n[Batch {i}]")
        print(batch)
        if i > 5:  # サンプルだけで止めたいとき
            break

if __name__ == "__main__":
    main()
