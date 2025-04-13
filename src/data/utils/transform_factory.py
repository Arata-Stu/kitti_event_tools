import numpy as np
from src.data.utils.transform.rotate import Rotate
from src.data.utils.transform.resize import Resize

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

def make_transform_for_sequence(seq_id: str):
    # rng = np.random.RandomState(seed=int(seq_id))
    # angle = rng.uniform(-15, 15)
    # return Compose([
    #     Rotate(angle),
    #     Resize((128, 128)),  # 必要に応じて変更
    # ])
    return None
