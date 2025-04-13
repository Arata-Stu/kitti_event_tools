# data/utils/sharded_stream_sampler.py
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

class ShardedSequenceSampler(IterableDataset):
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        local_worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        global_worker_id = rank * num_workers + local_worker_id

        # 分割
        my_datasets = [
            ds for i, ds in enumerate(self.datasets)
            if i % (world_size * num_workers) == global_worker_id
        ]
        iters = [iter(ds) for ds in my_datasets]
        while iters:
            batch = []
            for i in range(min(self.batch_size, len(iters))):
                try:
                    batch.append(next(iters[i]))
                except StopIteration:
                    iters.pop(i)
            if batch:
                yield batch
            else:
                break
