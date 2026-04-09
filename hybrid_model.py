"""
hybrid_model.py
---------------
Extracts penultimate-layer embeddings from a trained CNN (ResNet18 or
EfficientNet-B0) and feeds them into an SVM classifier.

This often outperforms both CNN-alone and classical-ML-alone because
the SVM finds a better margin-based decision boundary in the learned
embedding space than a linear softmax head.

Requirements:
    - A trained CNN saved as  models/resnet50_best.pt   (state_dict)
    - data/faces/  with real/ and fake/ subdirectories

Output:
    results/hybrid_results.json
    results/pr_curves_hybrid.json
    models/hybrid_svm.pkl

Usage:
    python hybrid_model.py
    python hybrid_model.py --arch efficientnet   # if group used EfficientNet
    python hybrid_model.py --max_per_class 5000  # quick test
"""

import os
import json
import argparse
import numpy as np
import pickle

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)

FACES_DIR    = os.path.expanduser("~/projects/ML/data/faces")
MODELS_DIR   = "models"
RESULTS_DIR  = "results"
RANDOM_STATE = 42
BATCH_SIZE   = 64
EMBED_DIM    = 512   # ResNet18 / EfficientNet-B0 both output 512 before head


# ── Dataset ───────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, faces_dir, transform=None, max_per_class=0):
        self.samples = []
        self.transform = transform
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            d = os.path.join(faces_dir, label_name)
            if not os.path.isdir(d):
                continue
            count = 0
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(root, f), label_id))
                        count += 1
                        if max_per_class and count >= max_per_class:
                            break
                if max_per_class and count >= max_per_class:
                    break
            print(f"  [{label_name}] {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── CNN backbone (no classification head) ─────────────────────────────────────
def build_backbone(arch: str, weights_path: str, device):
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Identity()   # remove final FC → output is 512-dim
    elif arch == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose 'resnet50' or 'efficientnet'")

    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        # Strip 'fc.' or 'classifier.' keys if the checkpoint includes them
        state = {k: v for k, v in state.items()
                 if not k.startswith(('fc.', 'classifier.'))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"✓ Loaded weights from {weights_path}")
        if missing:
            print(f"  Missing keys (expected if head was removed): {missing[:3]}")
    else:
        print(f"⚠  No weights found at {weights_path}.")
        print("   Using random weights — embeddings will be meaningless.")
        print("   Ask your teammate to run:  torch.save(model.state_dict(), 'models/resnet50_best.pt')")

    model.eval()
    model.to(device)
    return model


# ── Embedding extraction ──────────────────────────────────────────────────────
def extract_embeddings(model, dataloader, device):
    embeddings, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            emb = model(imgs)
            # Flatten in case model returns (B, C, 1, 1)
            emb = emb.view(emb.size(0), -1)
            embeddings.append(emb.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.vstack(embeddings), np.array(labels)


# ── Evaluate SVM ──────────────────────────────────────────────────────────────
def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "model":     "CNN_SVM_Hybrid",
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n── CNN → SVM Hybrid ──")
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"   {k:<12}: {metrics[k]:.4f}")
    print(f"   confusion:\n{np.array(metrics['confusion_matrix'])}")
    return metrics, y_prob


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_dir',     default=FACES_DIR)
    parser.add_argument('--models_dir',    default=MODELS_DIR)
    parser.add_argument('--results_dir',   default=RESULTS_DIR)
    parser.add_argument('--arch',          default='resnet50',
                        choices=['resnet50', 'efficientnet'])
    parser.add_argument('--weights',       default='models/resnet50_best.pt')
    parser.add_argument('--max_per_class', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.models_dir,  exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ImageNet normalisation — matches how the CNN was trained
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("\nScanning face images...")
    dataset = FaceDataset(args.faces_dir, transform=transform,
                          max_per_class=args.max_per_class)
    print(f"Total: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)

    model = build_backbone(args.arch, args.weights, device)

    print("\n⏳ Extracting CNN embeddings...")
    X, y = extract_embeddings(model, loader, device)
    print(f"✓ Embeddings: {X.shape}")

    # Save embeddings for potential reuse
    np.save(os.path.join(args.models_dir, "cnn_embeddings.npy"), X)
    np.save(os.path.join(args.models_dir, "cnn_labels.npy"),     y)

    # Train/val/test split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
    print(f"Split  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}")

    # Train hybrid SVM on CNN embeddings
    print("\n⏳ Training SVM on CNN embeddings...")
    hybrid_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel='rbf', C=1.0, gamma='scale',
                       probability=True, random_state=RANDOM_STATE)),
    ])
    hybrid_clf.fit(X_tr, y_tr)

    model_path = os.path.join(args.models_dir, "hybrid_svm.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(hybrid_clf, f)
    print(f"💾 Saved hybrid SVM to {model_path}")

    metrics, y_prob = evaluate(hybrid_clf, X_te, y_te)

    prec, rec, _ = precision_recall_curve(y_te, y_prob)
    ap = average_precision_score(y_te, y_prob)
    pr_data = {"CNN_SVM_Hybrid": {
        "precision": prec.tolist(), "recall": rec.tolist(), "ap": round(ap, 4)
    }}

    with open(os.path.join(args.results_dir, "hybrid_results.json"), 'w') as f:
        json.dump([metrics], f, indent=2)
    with open(os.path.join(args.results_dir, "pr_curves_hybrid.json"), 'w') as f:
        json.dump(pr_data, f, indent=2)

    print(f"\n✅ Done.  Next → python eval_visualizations.py")


if __name__ == "__main__":
    main()
