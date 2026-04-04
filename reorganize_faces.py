"""
Moves flat face crops produced by crop_faces2.py into the same
subdirectory structure used by the frames pipeline:
  faces/<label>/<method>/<video_name>/<filename>
"""

import os
import shutil
from tqdm import tqdm

KNOWN_METHODS = {"Deepfakes", "FaceSwap", "Face2Face", "FaceShifter", "NeuralTextures", "original"}


def parse_filename(name):
    """
    Split a flat filename like 'Deepfakes_000_003_frame_00000_face.jpg'
    into (method, video_name, frame_filename).
    Returns None if the filename doesn't match the expected pattern.
    """
    # Everything before the first '_frame_' is '{method}_{video_name}'
    try:
        prefix, rest = name.split("_frame_", 1)
    except ValueError:
        return None

    # Match the known method at the start of the prefix
    method = next((m for m in KNOWN_METHODS if prefix.startswith(m + "_") or prefix == m), None)
    if method is None:
        return None

    video_name = prefix[len(method) + 1:] if prefix != method else ""
    frame_filename = f"frame_{rest}"
    return method, video_name, frame_filename


def reorganize(faces_dir, dry_run=False):
    moved = 0
    skipped = 0

    all_files = [
        f for f in os.listdir(faces_dir)
        if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(faces_dir, f))
    ]

    for filename in tqdm(all_files, desc=f"  Moving", unit="file"):
        parsed = parse_filename(filename)
        if parsed is None:
            skipped += 1
            continue

        method, video_name, frame_filename = parsed
        dest_dir = os.path.join(faces_dir, method, video_name)
        src = os.path.join(faces_dir, filename)
        dst = os.path.join(dest_dir, frame_filename)

        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.move(src, dst)
        moved += 1

    return moved, skipped


def main():
    base_dir = os.path.expanduser("~/projects/ML/data")

    for label in ["real", "fake"]:
        faces_dir = os.path.join(base_dir, "faces", label)
        print(f"\n[{label.upper()}] {faces_dir}")
        moved, skipped = reorganize(faces_dir)
        print(f"  Moved:   {moved}")
        if skipped:
            print(f"  Skipped: {skipped} (couldn't parse filename)")

    print("\nDone!")


if __name__ == "__main__":
    main()
