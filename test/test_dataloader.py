import sys
sys.path.append("..")  # 親ディレクトリをパスに追加
from pathlib import Path
from types import SimpleNamespace
from src.data.dataloader import build_random_dataloader

def test_build_random_dataloader():
    cfg = SimpleNamespace(
        data_dir=Path("tests/mock_kitti"),  # 存在する小さなテスト用データに変更
        ev_repr_name="voxels",
        seq_len=5,
        downsample=False,
        transform=None,
        batch_size=SimpleNamespace(train=2, eval=1),
        hardware=SimpleNamespace(num_workers=0)  # シンプルな検証用
    )

    loader = build_random_dataloader(mode="train", cfg=cfg)

    # 1バッチだけ確認
    batch = next(iter(loader))
    assert "data" in batch
    assert "images" in batch["data"][0]
    assert "events" in batch["data"][0]
    assert batch["data"][0]["images"].shape[0] == cfg.seq_len
    assert batch["data"][0]["events"].shape[0] == cfg.seq_len

    print("✅ Random dataloader test passed.")
