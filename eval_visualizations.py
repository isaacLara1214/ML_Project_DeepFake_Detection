"""
eval_visualizations.py
----------------------
Generates all figures needed for the paper's results section:

  1. Confusion matrix grid  (all models, 2x2)    → results/confusion_matrices.png
  2. Precision-Recall curves (all models overlay) → results/pr_curves.png
  3. Grad-CAM examples       (real vs fake)       → results/gradcam_examples.png

Requirements:
    - results/classical_results.json  (from classical_ml.py)
    - results/hybrid_results.json     (from hybrid_model.py)
    - results/pr_curves_classical.json
    - results/pr_curves_hybrid.json
    - models/cnn_weights.pth          (for Grad-CAM)
    - data/faces/                     (for Grad-CAM sample images)

Usage:
    python eval_visualizations.py
    python eval_visualizations.py --skip_gradcam   # if CNN weights unavailable
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

RESULTS_DIR = "results"
MODELS_DIR  = "models"
FACES_DIR   = os.path.expanduser("~/projects/ML/data/faces")

# Colour palette — consistent across all figures
COLORS = {
    "SVM_RBF":        "#2196F3",
    "RandomForest":   "#4CAF50",
    "AdaBoost":       "#FF9800",
    "CNN_SVM_Hybrid": "#9C27B0",
    "CNN":            "#F44336",
}


# ── 1. Confusion Matrix Grid ──────────────────────────────────────────────────
def plot_confusion_matrices(all_results: list, out_path: str):
    n = len(all_results)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for i, result in enumerate(all_results):
        ax = axes[i]
        cm = np.array(result["confusion_matrix"])
        name = result["model"]

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        classes = ["Real", "Fake"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=13)
        ax.set_yticklabels(classes, fontsize=13)

        thresh = cm.max() / 2.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, f"{cm[row, col]:,}",
                        ha="center", va="center", fontsize=14,
                        color="white" if cm[row, col] > thresh else "black")

        f1  = result.get("f1", 0)
        auc = result.get("auc", 0)
        ax.set_title(f"{name}\nF1={f1:.4f}  AUC={auc:.4f}", fontsize=14, pad=12)
        ax.set_ylabel("True label",      fontsize=12)
        ax.set_xlabel("Predicted label", fontsize=12)

    # Hide any extra axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Confusion matrices → {out_path}")


# ── 2. Precision-Recall Curve ─────────────────────────────────────────────────
def plot_pr_curves(pr_data: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, data in pr_data.items():
        color = COLORS.get(name, "#607D8B")
        ap    = data.get("ap", 0)
        ax.plot(data["recall"], data["precision"],
                label=f"{name}  (AP={ap:.3f})",
                color=color, linewidth=2.2)

    ax.set_xlabel("Recall",    fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 PR curves → {out_path}")


# ── 3. Grad-CAM ───────────────────────────────────────────────────────────────
def gradcam_on_image(model, img_tensor, target_layer):
    """Return heatmap (H, W) via Grad-CAM on the given layer."""
    import torch

    activations, gradients = [], []

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    model.eval()
    out = model(img_tensor.unsqueeze(0))
    score = out[0, out.argmax()]
    model.zero_grad()
    score.backward()

    fh.remove()
    bh.remove()

    act  = activations[0].squeeze(0)   # (C, H, W)
    grad = gradients[0].squeeze(0)     # (C, H, W)
    weights = grad.mean(dim=(1, 2))    # (C,)

    cam = (weights[:, None, None] * act).sum(0)
    cam = torch.relu(cam).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay_gradcam(img_bgr, cam):
    """Overlay Grad-CAM heatmap on BGR image."""
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def sample_images(faces_dir, n=3):
    """Return n sample paths from each class."""
    samples = {"real": [], "fake": []}
    for label in ["real", "fake"]:
        d = os.path.join(faces_dir, label)
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    samples[label].append(os.path.join(root, f))
            if len(samples[label]) >= n:
                break
        samples[label] = samples[label][:n]
    return samples


def plot_gradcam(weights_path: str, faces_dir: str, arch: str, out_path: str):
    try:
        import torch
        from torchvision import models, transforms
    except ImportError:
        print("⚠  PyTorch not available, skipping Grad-CAM")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model with classification head intact
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 2)
        target_layer = model.layer4[-1]
    else:
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = torch.nn.Linear(1280, 2)
        target_layer = model.features[-1]

    if not os.path.exists(weights_path):
        print(f"⚠  No CNN weights at {weights_path} — skipping Grad-CAM")
        return

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    samples = sample_images(faces_dir, n=3)
    n_real = len(samples["real"])
    n_fake = len(samples["fake"])
    total  = n_real + n_fake

    if total == 0:
        print("⚠  No sample images found for Grad-CAM")
        return

    fig, axes = plt.subplots(2, max(n_real, n_fake),
                             figsize=(5 * max(n_real, n_fake), 10))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (label, paths) in enumerate([("real", samples["real"]),
                                               ("fake", samples["fake"])]):
        for col_idx, path in enumerate(paths):
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = transform(img_rgb).to(device)

            cam     = gradcam_on_image(model, tensor, target_layer)
            overlay = overlay_gradcam(img_bgr, cam)

            ax = axes[row_idx, col_idx]
            ax.imshow(overlay)
            ax.set_title(f"{label.upper()}", fontsize=13,
                         color="green" if label == "real" else "red")
            ax.axis('off')

    # Hide unused axes
    for row_idx in range(2):
        n_paths = len(samples["real"] if row_idx == 0 else samples["fake"])
        for col_idx in range(n_paths, axes.shape[1]):
            axes[row_idx, col_idx].set_visible(False)

    fig.suptitle("Grad-CAM: Regions influencing deepfake detection",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"💾 Grad-CAM → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',  default=RESULTS_DIR)
    parser.add_argument('--models_dir',   default=MODELS_DIR)
    parser.add_argument('--faces_dir',    default=FACES_DIR)
    parser.add_argument('--arch',         default='resnet18',
                        choices=['resnet18', 'efficientnet'])
    parser.add_argument('--skip_gradcam', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load all results
    all_results = []
    pr_data     = {}

    for fname in ["classical_results.json", "hybrid_results.json"]:
        path = os.path.join(args.results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                all_results.extend(json.load(f))
        else:
            print(f"⚠  {path} not found — run classical_ml.py / hybrid_model.py first")

    for fname in ["pr_curves_classical.json", "pr_curves_hybrid.json"]:
        path = os.path.join(args.results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                pr_data.update(json.load(f))

    if not all_results:
        print("❌ No results found. Run classical_ml.py and hybrid_model.py first.")
        return

    # 1. Confusion matrices
    plot_confusion_matrices(
        all_results,
        os.path.join(args.results_dir, "confusion_matrices.png"))

    # 2. PR curves
    if pr_data:
        plot_pr_curves(
            pr_data,
            os.path.join(args.results_dir, "pr_curves.png"))
    else:
        print("⚠  No PR curve data found, skipping PR plot")

    # 3. Grad-CAM
    if not args.skip_gradcam:
        weights_path = os.path.join(args.models_dir, "cnn_weights.pth")
        plot_gradcam(weights_path, args.faces_dir, args.arch,
                     os.path.join(args.results_dir, "gradcam_examples.png"))
    else:
        print("Skipping Grad-CAM (--skip_gradcam)")

    print(f"\n✅ All figures saved to {args.results_dir}/")
    print("   confusion_matrices.png")
    print("   pr_curves.png")
    if not args.skip_gradcam:
        print("   gradcam_examples.png")


if __name__ == "__main__":
    main()
