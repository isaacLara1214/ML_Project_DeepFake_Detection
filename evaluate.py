"""
evaluate.py — Evaluate a saved checkpoint on the held-out test set.

Usage:
    python evaluate.py --model resnet50 --checkpoint models/resnet50_best.pt
    python evaluate.py --model efficientnet_b0 --checkpoint models/efficientnet_b0_best.pt

Outputs (saved to results/):
    <model>_test_metrics.json    — accuracy, precision, recall, F1, AUC
    <model>_confusion_matrix.png
    <model>_roc_curve.png
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay,
)

from train import DeepfakeDataset, build_model

# ── Transforms (no augmentation) ─────────────────────────────────────────────
eval_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_test_loader(data_dir, batch_size, num_workers, split_override=None):
    data_dir = os.path.expanduser(data_dir)
    full_ds = DeepfakeDataset(data_dir, transform=eval_transform)

    split_path = os.path.expanduser(split_override) if split_override else os.path.join(os.path.dirname(data_dir), "split_indices.pt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"{split_path} not found. Run train.py first (it auto-generates the split)."
        )

    split = torch.load(split_path)
    test_ds = Subset(full_ds, split["test"])

    n_real = sum(1 for i in split["test"] if full_ds.samples[i][1] == 0)
    n_fake = sum(1 for i in split["test"] if full_ds.samples[i][1] == 1)
    print(f"Test set — real: {n_real}  fake: {n_fake}  total: {len(test_ds)}")

    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().astype(int))

    return np.array(all_probs), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    return roc_auc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("results", exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} from {args.checkpoint}")
    model = build_model(args.model, freeze_backbone=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # ── Test data ─────────────────────────────────────────────────────────────
    loader = get_test_loader(args.data_dir, args.batch, args.workers, args.split)

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference...")
    probs, labels = run_inference(model, loader, device)
    preds = (probs >= 0.5).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    roc_path = f"results/{args.model}_roc_curve.png"
    roc_auc  = plot_roc_curve(labels, probs, roc_path, args.model)

    metrics = {
        "model":     args.model,
        "accuracy":  round(float(acc),  4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
        "auc":       round(float(roc_auc), 4),
        "n_test":    int(len(labels)),
    }

    print("\n=== Test Results ===")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = f"results/{args.model}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm_path = f"results/{args.model}_confusion_matrix.png"
    plot_confusion_matrix(labels, preds, cm_path, args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector on test set")
    parser.add_argument("--model",      required=True,
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint from train.py")
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--data-dir",  default="~/OneDrive/Desktop/DeepFake Detection_images/data/faces",
                        help="Path to faces dir containing real/ and fake/ subfolders")
    parser.add_argument("--split",    default=None,
                        help="Path to split_indices.pt (auto-detected if omitted)")
    args = parser.parse_args()
    main(args)
