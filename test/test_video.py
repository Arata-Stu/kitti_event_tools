import sys
sys.path.append("..")

from omegaconf import OmegaConf
import cv2
import numpy as np
import torch
from pathlib import Path
from src.data.stream_dataset import build_stream_dataset
from pathlib import Path

def save_sequence_as_video(images_list, out_path, fps=10):
    if len(images_list) == 0:
        raise ValueError("空の画像リストです")

    images_bgr = []
    for i, img in enumerate(images_list):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()  # torch → numpy に変換

        if img.ndim != 3:
            raise ValueError(f"[{i}] 画像の次元が3ではありません: shape={img.shape}")
        if img.shape[0] != 3:
            raise ValueError(f"[{i}] Cが3ではありません: shape={img.shape}")

        img_bgr = np.transpose(img, (1, 2, 0))[:, :, ::-1]  # RGB → BGR
        images_bgr.append(img_bgr.astype(np.uint8))

    height, width = images_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for frame in images_bgr:
        out.write(frame)

    out.release()
    print(f"✅ 動画保存完了: {out_path}")

def main():
    config_path = "../config/test.yaml"
    cfg = OmegaConf.load(config_path)
    seq_ids = ["0000"]

    datasets = build_stream_dataset(dataset_mode="train", seq_ids=seq_ids, dataset_cfg=cfg.dataset)
    ds = datasets[0]  # シーケンス0000だけ

    save_dir = Path("debug_videos")
    save_dir.mkdir(exist_ok=True)

    current_images = []
    for sample in ds:
        # 各sampleは images: [seq_len, C, H, W]
        # 重複なしの __iter__ を前提に、連続フレームをすべてつなげる
        current_images.extend(sample["images"])

    video_path = save_dir / "sequence_0000_full.mp4"
    save_sequence_as_video(current_images, video_path)

if __name__ == "__main__":
    main()
