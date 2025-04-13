import argparse
import cv2
from pathlib import Path

def create_mp4_from_pngs(image_dir: Path, output_path: Path, fps=10):
    png_files = sorted(image_dir.glob("*.png"))
    if not png_files:
        print(f"[{image_dir.name}] No PNG files found.")
        return

    first_image = cv2.imread(str(png_files[0]))
    if first_image is None:
        print(f"[{image_dir.name}] Failed to read the first image.")
        return
    height, width, _ = first_image.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img_path in png_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            video_writer.write(img)

    video_writer.release()
    print(f"[{image_dir.name}] Video saved to {output_path}")

def process_all_sequences(base_dir: Path, output_dir: Path, fps=10):
    sequence_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])

    for seq_dir in sequence_dirs:
        output_path = output_dir / f"{seq_dir.name}.mp4"
        create_mp4_from_pngs(seq_dir, output_path, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNG images to MP4 video.")
    parser.add_argument("--base_dir", type=Path, help="Base directory containing sequence folders.")
    parser.add_argument("--output_dir", type=Path, help="Output directory for the MP4 files.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
    args = parser.parse_args()

    process_all_sequences(args.base_dir, args.output_dir, args.fps)