"""
train.py — Fine-tune ResNet-50 or EfficientNet-B0 for deepfake detection.

Usage:
    python train.py --model resnet50   --epochs 25 --batch 32
    python train.py --model efficientnet_b0 --epochs 25 --batch 32

Data expected at:  data/faces/real/  and  data/faces/fake/
Split indices from: data/split_indices.pt  (produced by generate_split.py)
Checkpoints saved to: models/
Training history saved to: results/
"""

import argparse
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # noqa: F401
from tqdm import tqdm

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DropoutLinear(nn.Linear):
    """Linear head with input dropout while keeping fc.weight/fc.bias keys."""

    def __init__(self, in_features, out_features=1, p=0.5):
        super().__init__(in_features, out_features)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return super().forward(self.dropout(x))


# ── Dataset ───────────────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """Walks data/faces/real and data/faces/fake, assigns labels 0/1."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        for label, folder in [(0, "real"), (1, "fake")]:
            folder_path = os.path.join(root_dir, folder)
            for root, dirs, files in os.walk(folder_path):
                dirs.sort()
                for f in sorted(files):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(root, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(name: str, freeze_backbone: bool = True) -> nn.Module:
    """Return a pretrained model with a binary classification head."""
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = DropoutLinear(in_features, 1, p=0.5)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[0] = nn.Dropout(p=0.5)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)

    else:
        raise ValueError(f"Unknown model: {name}. Choose 'resnet50' or 'efficientnet_b0'.")

    return model


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model: nn.Module, name: str, lr_backbone: float, lr_head: float):
    """Create optimizer groups for differential learning rates."""
    head = model.fc if name == "resnet50" else model.classifier
    head_param_ids = {id(p) for p in head.parameters()}

    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_param_ids
    ]
    head_params = [p for p in head.parameters() if p.requires_grad]

    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]


# ── Training helpers ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True, freeze_bn=False):
    model.train(train)
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    phase = "Train" if train else "Val  "
    with torch.set_grad_enabled(train):
        pbar = tqdm(loader, desc=f"    {phase}", leave=False, unit="batch")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs).squeeze(1)          # (B,)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.long().cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader.dataset)
    avg_loss = total_loss / n
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir  = os.path.join(script_dir, "models")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    # Build full dataset twice: once with train aug, once with val aug
    data_dir = os.path.expanduser(args.data_dir)
    full_train_ds = DeepfakeDataset(data_dir, transform=train_transform)
    full_val_ds   = DeepfakeDataset(data_dir, transform=val_transform)

    if len(full_train_ds) == 0:
        print(f"ERROR: No images found in {data_dir}/real or {data_dir}/fake")
        print("  Pass --data-dir to point to your faces directory.")
        return

    all_samples = full_train_ds.samples
    n_total = len(all_samples)

    # Load precomputed split (must be generated by generate_split.py)
    split_path = args.split if args.split else os.path.join(os.path.dirname(data_dir), "split_indices.pt")
    split_path = os.path.expanduser(split_path)
    if not os.path.exists(split_path):
        print(f"ERROR: split file not found at {split_path}")
        print("  Run: python generate_split.py --data-dir <faces_dir> --out <split_path>")
        return

    split = torch.load(split_path)
    if not all(k in split for k in ("train", "val", "test")):
        print("ERROR: split file is missing one or more keys: train, val, test")
        return

    train_idx = list(split["train"])
    val_idx = list(split["val"])
    test_idx = list(split["test"])

    if any(i < 0 or i >= n_total for i in train_idx + val_idx + test_idx):
        print("ERROR: split indices are out of range for current dataset.")
        print("  Regenerate split with the same data directory used for training.")
        return

    overlap = (set(train_idx) & set(val_idx)) | (set(train_idx) & set(test_idx)) | (set(val_idx) & set(test_idx))
    if overlap:
        print("ERROR: split file has overlapping indices between train/val/test")
        return

    if args.max_samples and args.max_samples < n_total:
        keep = set(torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))[:args.max_samples].tolist())
        train_idx = [i for i in train_idx if i in keep]
        val_idx = [i for i in val_idx if i in keep]
        test_idx = [i for i in test_idx if i in keep]
        print(f"Subsampled to {len(train_idx) + len(val_idx) + len(test_idx)} images (--max-samples)")

    if len(train_idx) == 0 or len(val_idx) == 0:
        print("ERROR: train/val split is empty after filtering; adjust split or --max-samples")
        return

    print(f"Split loaded from {split_path}  (train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})")

    train_ds = Subset(full_train_ds, train_idx)
    val_ds   = Subset(full_val_ds,   val_idx)

    # Class weights to handle imbalance
    train_labels = [all_samples[i][1] for i in train_idx]
    n_real = train_labels.count(0)
    n_fake = train_labels.count(1)
    print(f"Train — real: {n_real}  fake: {n_fake}  total: {len(train_ds)}")
    print(f"Val   — {len(val_ds)} samples")

    pos_weight = torch.tensor([n_real / max(n_fake, 1)], dtype=torch.float32).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding {args.model} (phase 1: frozen backbone)...")
    model = build_model(args.model, freeze_backbone=True).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path = os.path.join(models_dir, f"{args.model}_best.pt")

    # ── Phase 1: train head only ───────────────────────────────────────────────
    phase1_epochs = max(1, args.phase1_epochs)
    print(f"\n=== Phase 1: head only ({phase1_epochs} epochs, lr={args.lr_head}) ===")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    for epoch in tqdm(range(1, phase1_epochs + 1), desc="  Phase 1", unit="epoch"):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True, freeze_bn=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False, freeze_bn=True)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)
            flag = "  [saved]"

        tqdm.write(f"  Ep {epoch:02d}/{phase1_epochs}  "
                   f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                   f"val loss={va_loss:.4f} acc={va_acc:.4f}  "
                   f"({time.time()-t0:.0f}s){flag}")

    # ── Phase 2: full fine-tune ────────────────────────────────────────────────
    phase2_epochs = args.epochs - phase1_epochs
    if phase2_epochs > 0:
        print(f"\n=== Phase 2: full fine-tune ({phase2_epochs} epochs, backbone lr={args.lr_backbone}, head lr={args.lr_finetune}) ===")
        unfreeze_all(model)
        optimizer = torch.optim.Adam(
            get_param_groups(model, args.model, args.lr_backbone, args.lr_finetune),
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)

        for epoch in tqdm(range(1, phase2_epochs + 1), desc="  Phase 2", unit="epoch"):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            flag = ""
            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), ckpt_path)
                flag = "  [saved]"

            tqdm.write(f"  Ep {epoch:02d}/{phase2_epochs}  "
                       f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                       f"val loss={va_loss:.4f} acc={va_acc:.4f}  "
                       f"({time.time()-t0:.0f}s){flag}")

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(results_dir, f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"History:    {hist_path}")

    # ── Plot training curves ──────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs_range = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs_range, history["train_loss"], label="Train")
        ax1.plot(epochs_range, history["val_loss"],   label="Val")
        if phase1_epochs < args.epochs:
            ax1.axvline(phase1_epochs, color="gray", linestyle="--", label="Unfreeze")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs_range, history["train_acc"], label="Train")
        ax2.plot(epochs_range, history["val_acc"],   label="Val")
        if phase1_epochs < args.epochs:
            ax2.axvline(phase1_epochs, color="gray", linestyle="--", label="Unfreeze")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        fig.suptitle(f"{args.model} — Training Curves")
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"{args.model}_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot:       {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deepfake detector (ResNet-50 / EfficientNet-B0)")
    parser.add_argument("--model",         default="resnet50",
                        choices=["resnet50", "efficientnet_b0"],
                        help="Backbone architecture")
    parser.add_argument("--epochs",        type=int,   default=25,
                        help="Total training epochs (phase1 + phase2)")
    parser.add_argument("--phase1-epochs", type=int,   default=5,
                        help="Epochs to train head only (frozen backbone)")
    parser.add_argument("--batch",         type=int,   default=32)
    parser.add_argument("--lr-head",       type=float, default=1e-3,
                        help="Learning rate for phase 1 (head only)")
    parser.add_argument("--lr-finetune",   type=float, default=1e-4,
                        help="Learning rate for phase 2 head")
    parser.add_argument("--lr-backbone",   type=float, default=1e-5,
                        help="Learning rate for phase 2 backbone")
    parser.add_argument("--workers",       type=int,   default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--data-dir",      default="~/projects/ML/data/faces",
                        help="Path to faces dir containing real/ and fake/ subfolders")
    parser.add_argument("--split",         default=None,
                        help="Path to split_indices.pt (auto-detected if omitted)")
    parser.add_argument("--max-samples",  type=int, default=None,
                        help="Cap total images (e.g. 5000) for faster CPU training")

    args = parser.parse_args()
    main(args)
