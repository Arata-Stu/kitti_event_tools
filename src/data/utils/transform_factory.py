import numpy as np
from omegaconf import DictConfig
from src.data.utils.transform.resize import Resize
from src.data.utils.transform.flip import Flip
from src.data.utils.transform.rotate import Rotate
from src.data.utils.transform.zoom import RandomZoom, ZoomPerSequence

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

class TransformFactory:
    def __init__(self, mode: str, transform_cfg: DictConfig):
        assert mode in ["train", "test", "val"], f"Invalid mode: {mode}"
        self.target_size = transform_cfg.get("target_size", None)
        self.rotate_range = transform_cfg.get("rotate_range", None)
        self.zoom_weight = transform_cfg.get("zoom_weight", None)
        self.mode = mode

    def build_for_random(self):
        """ ランダムアクセス用の transform を構築 """
        angle = np.random.uniform(*self.rotate_range)
        hflip = np.random.rand() < 0.5
        vflip =False

        if self.mode == "train":
            # train の場合は、resize, flip, rotate, zoom を適用
            transform = Compose([
                Resize(self.target_size),
                Flip(horizontal=hflip, vertical=vflip),
                Rotate(angle),
                RandomZoom(prob_weight=self.zoom_weight)
            ])
        else:
            # test/val の場合は、resize, flip, rotate を適用
            transform = Compose([
                Resize(self.target_size)
            ])
            
        return transform
    
    def build_for_stream(self, seq_id: str):
        """ Streaming 用に、seq_id に依存した一貫した transform を構築 """
        rng = np.random.RandomState(seed=int(seq_id))
        angle = rng.uniform(*self.rotate_range)
        hflip = rng.rand() < 0.5
        vflip = False

        if self.mode == "train":
            # train の場合は、resize, flip, rotate, zoom を適用
            transform = Compose([
                Resize(self.target_size),
                Flip(horizontal=hflip, vertical=vflip),
                Rotate(angle),
                ZoomPerSequence(prob_weight=self.zoom_weight)
            ])
        else:
            # test/val の場合は、resize, flip, rotate を適用
            transform = Compose([
                Resize(self.target_size)
            ])

        return transform