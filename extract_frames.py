import os
import numpy as np  # noqa: F401
import cv2
from decord import VideoReader, gpu, cpu
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_interval=10, ctx=None):
    """
    Extract every Nth frame from a video.
    frame_interval=10 means 1 frame per ~0.33 seconds at 30fps
    """
    os.makedirs(output_dir, exist_ok=True)

    if ctx is None:
        ctx = cpu(0)

    vr = VideoReader(video_path, ctx=ctx)
    total = len(vr)
    indices = list(range(0, total, frame_interval))

    # Batch-fetch only the frames we actually need
    frames = vr.get_batch(indices).asnumpy()  # shape: (N, H, W, C), RGB

    for i, frame in enumerate(frames):
        out_path = os.path.join(output_dir, f"frame_{i:05d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return len(indices)


def process_dataset(base_dir, use_gpu=True):
    categories = {
        "real": ["original"],
        "fake": ["Deepfakes", "FaceSwap", "Face2Face", "FaceShifter", "NeuralTextures"]
    }

    ctx = gpu(0) if use_gpu else cpu(0)

    # Collect all videos first so we can show an outer progress bar
    videos = []
    for label, folders in categories.items():
        for folder in folders:
            video_dir = os.path.join(base_dir, folder)
            if not os.path.exists(video_dir):
                continue
            for video_file in os.listdir(video_dir):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    videos.append((label, folder, video_file))

    outer = tqdm(videos, desc="Videos", unit="video")
    inner = tqdm(total=0, desc="Frames", unit="frame", leave=False)

    for label, folder, video_file in outer:
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(base_dir, "..", "frames", label, folder, video_name)
        video_path = os.path.join(base_dir, folder, video_file)

        outer.set_postfix(file=f"{folder}/{video_file}")
        try:
            n = extract_frames(video_path, output_dir, frame_interval=10, ctx=ctx)
        except Exception:
            # Fall back to CPU if GPU decoding fails for a particular file
            n = extract_frames(video_path, output_dir, frame_interval=10, ctx=cpu(0))
        inner.reset(total=n)
        inner.update(n)
        outer.write(f"[{label}] {folder}/{video_file} → {n} frames saved")

    inner.close()
    outer.close()


if __name__ == "__main__":
    base_dir = os.path.expanduser("~/projects/ML/data/raw")
    process_dataset(base_dir, use_gpu=True)
