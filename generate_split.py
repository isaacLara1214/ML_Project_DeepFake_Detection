"""
Generate a video-level 70/15/15 split for deepfake training.

Important: this split groups by video ID (parent folder of each frame)
to prevent frame-level leakage between train/val/test.
"""

import argparse
import os

import torch
from tqdm import tqdm


SEED = 42


def collect_samples(root_dir):
    """Mirror train.py dataset enumeration to keep index alignment."""
    samples = []
    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(root_dir, folder)
        print(f"Scanning {folder_path} ...")
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            for name in sorted(files):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_files.append((os.path.join(root, name), label))
        for item in tqdm(all_files, desc=f"  {folder}", unit="img"):
            samples.append(item)
    return samples


def video_id(path):
    """Use parent folder as video/group identifier."""
    return os.path.basename(os.path.dirname(path))


def main():
    parser = argparse.ArgumentParser(description="Create video-level split indices")
    parser.add_argument("--data-dir", default="~/projects/ML/data/faces",
                        help="Path to faces dir containing real/ and fake/")
    parser.add_argument("--out", default="~/projects/ML/data/split_indices.pt",
                        help="Path to write split_indices.pt")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    out_path = os.path.expanduser(args.out)

    print(f"Data dir: {data_dir}")
    samples = collect_samples(data_dir)
    if not samples:
        print(f"ERROR: no images found under {data_dir}")
        return

    print(f"Found {len(samples):,} images — grouping by video ID ...")
    groups = {}
    for idx, (path, _) in enumerate(tqdm(samples, desc="Grouping", unit="img")):
        key = video_id(path)
        groups.setdefault(key, []).append(idx)

    keys = sorted(groups.keys())
    n_groups = len(keys)
    print(f"Unique video IDs: {n_groups} — computing 70/15/15 split (seed={args.seed}) ...")
    perm = torch.randperm(n_groups, generator=torch.Generator().manual_seed(args.seed)).tolist()

    n_train = int(0.70 * n_groups)
    n_val = int(0.15 * n_groups)

    train_keys = {keys[i] for i in perm[:n_train]}
    val_keys = {keys[i] for i in perm[n_train:n_train + n_val]}
    test_keys = {keys[i] for i in perm[n_train + n_val:]}

    train_idx, val_idx, test_idx = [], [], []
    for key, idxs in groups.items():
        if key in train_keys:
            train_idx.extend(idxs)
        elif key in val_keys:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    total = len(train_idx) + len(val_idx) + len(test_idx)
    if total != len(samples):
        print("ERROR: split assignment mismatch")
        return

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving split to: {out_path} ...")
    torch.save({"train": train_idx, "val": val_idx, "test": test_idx}, out_path)

    print("\nDone.")
    print(f"Total frames : {len(samples):,}")
    print(f"Video IDs    : {n_groups}  (train {len(train_keys)} / val {len(val_keys)} / test {len(test_keys)})")
    print(f"Frames       : train {len(train_idx):,} / val {len(val_idx):,} / test {len(test_idx):,}")
    print(f"Saved        : {out_path}")


if __name__ == "__main__":
    main()
