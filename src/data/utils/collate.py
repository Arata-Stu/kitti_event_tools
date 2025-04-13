# data/utils/collate.py

from torch.utils.data._utils.collate import default_collate
import numpy as np
import torch

custom_collate_fn_map = {
    torch.Tensor: default_collate,
    np.ndarray: lambda batch: torch.stack([torch.from_numpy(b) for b in batch]),
    int: default_collate,
    float: default_collate,
    str: lambda batch: batch,
    dict: lambda batch: {k: custom_collate([d[k] for d in batch]) for k in batch[0]},
    list: lambda batch: [custom_collate(b) for b in zip(*batch)],
}

def custom_collate(batch, collate_fn_map=custom_collate_fn_map):
    elem_type = type(batch[0])
    if elem_type in collate_fn_map:
        return collate_fn_map[elem_type](batch)
    raise TypeError(f"Unsupported type: {elem_type}")

def custom_collate_rnd(batch):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = 0 if worker_info is None else worker_info.id
    return {
        'data': custom_collate(batch),
        'worker_id': worker_id,
    }

def custom_collate_streaming(batch):
    samples, worker_id = batch
    return {
        'data': custom_collate(samples),
        'worker_id': worker_id,
    }
