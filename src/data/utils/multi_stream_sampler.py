# data/utils/multi_stream_sampler.py
import torch
from torch.utils.data import IterableDataset

class MultiStreamSampler(IterableDataset):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size

    def __iter__(self):
        iters = [iter(ds) for ds in self.datasets]
        while True:
            batch = []
            for _ in range(self.batch_size):
                i = torch.randint(0, len(iters), size=(1,)).item()
                try:
                    batch.append(next(iters[i]))
                except StopIteration:
                    continue
            if not batch:
                break
            yield batch
