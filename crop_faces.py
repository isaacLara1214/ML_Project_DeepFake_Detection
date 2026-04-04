import cv2
import os
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# ── GPU Setup ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version:  {torch.version.cuda}")
    print(f"  VRAM total:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  VRAM free:     {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.1f} GB")
else:
    print("⚠ No GPU found, running on CPU")

# ── Detector ─────────────────────────────────────────────────────────────────
detector = MTCNN(
    keep_all=False,        # only return the best/largest face
    device=device,         # use GPU
    min_face_size=60,      # ignore tiny faces
    thresholds=[0.6, 0.7, 0.9],  # P-Net, R-Net, O-Net confidence thresholds
    post_process=False     # return raw pixels, not normalized tensor
)

def crop_faces(frames_dir, faces_dir, padding=0.2, min_confidence=0.95):
    os.makedirs(faces_dir, exist_ok=True)
    saved = 0
    skipped = 0

    # Collect all image files first so tqdm knows the total
    all_files = []
    for root, _, files in os.walk(frames_dir):
        for filename in sorted(files):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append((root, filename))

    for root, filename in tqdm(all_files, desc="  Cropping", unit="frame"):
        img_path = os.path.join(root, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # facenet-pytorch expects RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        # Detect face and get bounding box + confidence
        boxes, probs = detector.detect(rgb)

        if boxes is None or probs is None:
            skipped += 1
            continue

        # Filter by confidence
        confident = [(box, prob) for box, prob in zip(boxes, probs) if prob >= min_confidence]
        if not confident:
            skipped += 1
            continue

        # Pick the largest face
        best_box, best_prob = max(confident, key=lambda x: (x[0][2]-x[0][0]) * (x[0][3]-x[0][1]))
        x1, y1, x2, y2 = best_box

        # Add padding
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w_img, int(x2) + pad_x)
        y2 = min(h_img, int(y2) + pad_y)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            skipped += 1
            continue

        face_crop = cv2.resize(face_crop, (224, 224))

        rel_path = os.path.relpath(root, frames_dir)
        out_dir = os.path.join(faces_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{os.path.splitext(filename)[0]}_face.jpg"
        out_path = os.path.join(out_dir, out_name)

        cv2.imwrite(out_path, face_crop)
        saved += 1

    return saved, skipped


def main():
    base_dir = os.path.expanduser("~/projects/ML/data")

    for label in ["real", "fake"]:
        frames_dir = os.path.join(base_dir, "frames", label)
        faces_dir  = os.path.join(base_dir, "faces",  label)

        print(f"\n[{label.upper()}] {frames_dir}")
        saved, skipped = crop_faces(frames_dir, faces_dir)
        print(f"  ✓ Saved:   {saved} face crops")
        print(f"  ✗ Skipped: {skipped} frames (no confident face detected)")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()