import sys
sys.path.append("..")  # 親ディレクトリをパスに追加

from omegaconf import OmegaConf
from src.data.dataloader import build_random_dataloader

def test_build_random_dataloader():
    config_path = "../config/test.yaml"
    cfg = OmegaConf.load(config_path)

    loader = build_random_dataloader(mode="train", cfg=cfg.dataset)

    # 1バッチだけ確認
    batch = next(iter(loader))
    assert "data" in batch
    assert "images" in batch["data"][0]
    assert "events" in batch["data"][0]
    assert batch["data"][0]["images"].shape[0] == cfg.seq_len
    assert batch["data"][0]["events"].shape[0] == cfg.seq_len

    print("✅ Random dataloader test passed.")
