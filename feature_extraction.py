"""
feature_extraction.py
---------------------
Extracts handcrafted features from cropped face images in data/faces/.
Features:
  - Local Binary Patterns (LBP)  — texture / micro-detail
  - FFT magnitude spectrum        — frequency artifact fingerprint
  - (optional) dlib landmarks     — geometric inconsistencies

Output: features/features.npy  (N x D float32)
        features/labels.npy    (N,) int  [0=real, 1=fake]
        features/paths.npy     (N,) str  — image paths for debugging

Usage:
    python feature_extraction.py
    python feature_extraction.py --max_per_class 5000   # quick test run
    python feature_extraction.py --use_landmarks        # requires dlib
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from skimage.feature import local_binary_pattern

# ── Optional: dlib landmarks ──────────────────────────────────────────────────
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
FACES_DIR   = os.path.expanduser("~/projects/ML/data/faces")
OUT_DIR     = "features"
LBP_RADIUS  = 3
LBP_POINTS  = 24           # 8 * radius
LBP_BINS    = LBP_POINTS + 2   # 'uniform' method → n_points + 2 bins
FFT_BINS    = 64           # radial frequency bins
LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"


# ── LBP Feature ──────────────────────────────────────────────────────────────
def extract_lbp(gray: np.ndarray) -> np.ndarray:
    """Return normalised LBP histogram (LBP_BINS,)."""
    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_BINS,
                           range=(0, LBP_BINS), density=True)
    return hist.astype(np.float32)


# ── FFT Feature ───────────────────────────────────────────────────────────────
def extract_fft(gray: np.ndarray) -> np.ndarray:
    """
    Return radially-binned FFT magnitude spectrum (FFT_BINS,).
    GAN upsampling artifacts appear as spikes at regular spatial frequencies.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    radius_map = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    max_r = np.sqrt(cx ** 2 + cy ** 2)
    bins = np.linspace(0, max_r, FFT_BINS + 1)
    hist = np.zeros(FFT_BINS, dtype=np.float32)
    for i in range(FFT_BINS):
        mask = (radius_map >= bins[i]) & (radius_map < bins[i + 1])
        vals = magnitude[mask]
        hist[i] = vals.mean() if vals.size > 0 else 0.0

    rng = hist.max() - hist.min()
    if rng > 0:
        hist = (hist - hist.min()) / rng
    return hist


# ── dlib Landmark Feature ─────────────────────────────────────────────────────
LANDMARK_DIM = 15

def build_landmark_detector():
    if not DLIB_AVAILABLE:
        print("⚠  dlib not installed. Run: pip install dlib")
        return None, None
    if not os.path.exists(LANDMARK_MODEL):
        print(f"⚠  Landmark model not found at '{LANDMARK_MODEL}'.")
        print("   Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("   Then: bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
        return None, None
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL)
    print("✓ dlib landmark detector loaded")
    return detector, predictor


def extract_landmarks(rgb: np.ndarray, detector, predictor) -> np.ndarray:
    """Return 15 facial geometry ratios, or zeros if detection fails."""
    zeros = np.zeros(LANDMARK_DIM, dtype=np.float32)
    if detector is None:
        return zeros

    dets = detector(rgb, 1)
    if len(dets) == 0:
        return zeros

    shape = predictor(rgb, dets[0])
    pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)],
                   dtype=np.float32)

    def dist(a, b):
        return float(np.linalg.norm(pts[a] - pts[b]) + 1e-6)

    face_w  = dist(0, 16)
    face_h  = dist(8, 27)
    eye_l   = dist(36, 39)
    eye_r   = dist(42, 45)
    eye_sep = dist(39, 42)
    nose_w  = dist(31, 35)
    mouth_w = dist(48, 54)
    mouth_h = dist(51, 57)

    sym_eye  = abs(eye_l - eye_r) / (face_w + 1e-6)
    sym_brow = abs(dist(17, 21) - dist(22, 26)) / (face_w + 1e-6)
    sym_cheek= abs(dist(1, 30)  - dist(15, 30)) / (face_w + 1e-6)

    return np.array([
        eye_l   / face_w,
        eye_r   / face_w,
        eye_sep / face_w,
        nose_w  / face_w,
        mouth_w / face_w,
        mouth_h / (mouth_w + 1e-6),
        face_h  / (face_w + 1e-6),
        dist(17, 26) / face_w,
        dist(36, 45) / face_w,
        dist(27, 33) / face_h,
        sym_eye,
        sym_brow,
        sym_cheek,
        dist(0, 8)   / (dist(8, 16)  + 1e-6),
        dist(36, 39) / (dist(42, 45) + 1e-6),
    ], dtype=np.float32)


# ── Image → feature vector ────────────────────────────────────────────────────
def extract_features(img_path: str, detector, predictor,
                     use_landmarks: bool) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    parts = [extract_lbp(gray), extract_fft(gray)]
    if use_landmarks:
        parts.append(extract_landmarks(rgb, detector, predictor))

    return np.concatenate(parts)


# ── Collect image paths ───────────────────────────────────────────────────────
def collect_paths(faces_dir: str, max_per_class: int):
    paths, labels = [], []
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        label_dir = os.path.join(faces_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"⚠  Missing directory: {label_dir}")
            continue
        count = 0
        for root, _, files in os.walk(label_dir):
            for f in sorted(files):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(root, f))
                    labels.append(label_id)
                    count += 1
                    if max_per_class and count >= max_per_class:
                        break
            if max_per_class and count >= max_per_class:
                break
        print(f"  [{label_name}] found {count} images")
    return paths, labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_dir', default=FACES_DIR)
    parser.add_argument('--max_per_class', type=int, default=0,
                        help='Cap per class (0 = no cap). Use 5000 for a quick test.')
    parser.add_argument('--use_landmarks', action='store_true')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n📂 Faces dir : {args.faces_dir}")
    print(f"📐 LBP bins  : {LBP_BINS}  |  FFT bins: {FFT_BINS}")
    print(f"📐 Landmarks : {'yes' if args.use_landmarks else 'no (add --use_landmarks to enable)'}")
    print(f"📐 Cap/class : {args.max_per_class if args.max_per_class else 'none'}\n")

    detector, predictor = None, None
    if args.use_landmarks:
        detector, predictor = build_landmark_detector()

    print("Scanning image directories...")
    paths, labels = collect_paths(args.faces_dir, args.max_per_class)
    print(f"\nTotal images to process: {len(paths)}\n")

    if not paths:
        print("❌ No images found. Run crop_faces.py first.")
        return

    feature_list, valid_labels, valid_paths = [], [], []
    for img_path, label in tqdm(zip(paths, labels), total=len(paths),
                                desc="Extracting features", unit="img"):
        feat = extract_features(img_path, detector, predictor, args.use_landmarks)
        if feat is not None:
            feature_list.append(feat)
            valid_labels.append(label)
            valid_paths.append(img_path)

    X = np.stack(feature_list).astype(np.float32)
    y = np.array(valid_labels, dtype=np.int32)
    p = np.array(valid_paths)

    print(f"\n✅ Feature matrix : {X.shape}   (real={( y==0).sum()}, fake={(y==1).sum()})")

    np.save(os.path.join(OUT_DIR, "features.npy"), X)
    np.save(os.path.join(OUT_DIR, "labels.npy"),   y)
    np.save(os.path.join(OUT_DIR, "paths.npy"),    p)
    print(f"💾 Saved to {OUT_DIR}/features.npy  labels.npy  paths.npy")


if __name__ == "__main__":
    main()
