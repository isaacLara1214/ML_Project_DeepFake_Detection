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

# ── Config ───────────────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.98       # raise from 0.95 → only very confident detections
MIN_FACE_RATIO = 0.15       # face must be at least 15% of frame width/height
PADDING        = 0.2
OUTPUT_SIZE    = (224, 224)

# ── Detector ─────────────────────────────────────────────────────────────────
detector = MTCNN(
    keep_all=False,
    device=device,
    min_face_size=60,
    thresholds=[0.6, 0.7, 0.9],
    post_process=False
)

def crop_faces(frames_dir, faces_dir):
    os.makedirs(faces_dir, exist_ok=True)
    saved = 0
    skipped_no_face = 0
    skipped_low_conf = 0
    skipped_too_small = 0

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

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img.shape[:2]

        boxes, probs = detector.detect(rgb)

        if boxes is None or probs is None:
            skipped_no_face += 1
            continue

        # Filter by confidence
        confident = [(box, prob) for box, prob in zip(boxes, probs) if prob >= MIN_CONFIDENCE]
        if not confident:
            skipped_low_conf += 1
            continue

        # Filter by face size relative to frame
        def face_is_large_enough(box):
            x1, y1, x2, y2 = box
            face_w = x2 - x1
            face_h = y2 - y1
            return (face_w / w_img) >= MIN_FACE_RATIO and (face_h / h_img) >= MIN_FACE_RATIO

        confident = [(box, prob) for box, prob in confident if face_is_large_enough(box)]
        if not confident:
            skipped_too_small += 1
            continue

        # Pick the largest face
        best_box, _ = max(confident, key=lambda x: (x[0][2]-x[0][0]) * (x[0][3]-x[0][1]))
        x1, y1, x2, y2 = best_box

        # Add padding
        w = x2 - x1
        h = y2 - y1
        pad_x = int(w * PADDING)
        pad_y = int(h * PADDING)
        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w_img, int(x2) + pad_x)
        y2 = min(h_img, int(y2) + pad_y)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            skipped_no_face += 1
            continue

        face_crop = cv2.resize(face_crop, OUTPUT_SIZE)

        rel_path = os.path.relpath(root, frames_dir)
        safe_name = rel_path.replace(os.sep, "_")
        out_name = f"{safe_name}_{os.path.splitext(filename)[0]}_face.jpg"
        out_path = os.path.join(faces_dir, out_name)

        cv2.imwrite(out_path, face_crop)
        saved += 1

    return saved, skipped_no_face, skipped_low_conf, skipped_too_small


def main():
    base_dir = os.path.expanduser("~/projects/ML/data")

    for label in ["real", "fake"]:
        frames_dir = os.path.join(base_dir, "frames", label)
        faces_dir  = os.path.join(base_dir, "faces",  label)

        print(f"\n[{label.upper()}] {frames_dir}")
        saved, no_face, low_conf, too_small = crop_faces(frames_dir, faces_dir)
        print(f"  ✓ Saved:              {saved} face crops")
        print(f"  ✗ No face detected:   {no_face} frames")
        print(f"  ✗ Low confidence:     {low_conf} frames")
        print(f"  ✗ Face too small:     {too_small} frames")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()