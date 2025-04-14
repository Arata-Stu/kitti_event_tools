import lightning.pytorch as pl
from omegaconf import DictConfig

from src.data.dataloader import (
    build_random_dataloader,
    build_stream_dataloader
)

class KittiDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_cfg: DictConfig,
                 dataloader_cfg: DictConfig,
                 use_streaming: bool = True):
        """
        Parameters:
            dataset_cfg: データセットの設定（data_dir, ev_repr_name, seq_len など）
            dataloader_cfg: DataLoader の設定（batch_size, num_workers など）
            use_streaming: TrueでStreamingモードを使用。FalseでRandomモード。
        """
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.use_streaming = use_streaming

    def _build_loader(self, mode: str):
        if self.use_streaming:
            return build_stream_dataloader(
                dataset_mode=mode,
                dataset_cfg=self.dataset_cfg,
                batch_size=self.dataloader_cfg.batch_size,
                num_workers=self.dataloader_cfg.num_workers,
                pin_memory=self.dataloader_cfg.get("pin_memory", True),
            )
        else:
            return build_random_dataloader(
                dataset_mode=mode,
                dataset_cfg=self.dataset_cfg,
                batch_size=self.dataloader_cfg.batch_size,
                num_workers=self.dataloader_cfg.num_workers,
                shuffle=self.dataloader_cfg.get("shuffle", True),
                drop_last=self.dataloader_cfg.get("drop_last", True),
                pin_memory=self.dataloader_cfg.get("pin_memory", True),
            )

    def train_dataloader(self):
        return self._build_loader("train")

    def val_dataloader(self):
        return self._build_loader("val")

    def test_dataloader(self):
        return self._build_loader("test")
