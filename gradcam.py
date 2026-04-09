"""
gradcam.py — Generate Grad-CAM heatmaps for the deepfake detector.

Usage:
    python gradcam.py --model resnet50       --checkpoint models/resnet50_best.pt
    python gradcam.py --model efficientnet_b0 --checkpoint models/efficientnet_b0_best.pt

    # Visualize a specific image:
    python gradcam.py --model resnet50 --checkpoint models/resnet50_best.pt \
                      --image path/to/face.jpg

Outputs saved to results/gradcam/
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

from train import DeepfakeDataset, build_model

# ── Eval transform ────────────────────────────────────────────────────────────
eval_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Registers hooks on the target layer to capture activations and gradients,
    then computes the weighted activation map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients   = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """
        Args:
            x: (1, 3, H, W) tensor on the model's device
        Returns:
            cam: (H, W) numpy array in [0, 1]
            prob: predicted probability of fake (class 1)
        """
        self.model.zero_grad()
        logit = self.model(x).squeeze()
        prob  = torch.sigmoid(logit).item()

        # Backprop w.r.t. the output logit
        logit.backward()

        # GAP over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()    # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, prob

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    """Return the last convolutional layer for each architecture."""
    if model_name == "resnet50":
        return model.layer4[-1].conv3      # last conv in layer4
    elif model_name == "efficientnet_b0":
        return model.features[-1][0]       # last Conv2d in features
    else:
        raise ValueError(f"Unknown model: {model_name}")


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto a BGR image."""
    cam_resized = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_rgb + (1 - alpha) * image_rgb).astype(np.uint8)
    return blended


def process_image(img_path: str, label: int, gradcam: GradCAM,
                  device: torch.device, model_name: str, save_dir: str):
    """Run Grad-CAM on a single image and save the result."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  Could not read {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_224 = cv2.resize(img_rgb, (224, 224))

    # Prepare tensor
    tensor = eval_transform(img_rgb).unsqueeze(0).to(device)
    tensor.requires_grad_(False)

    cam, prob = gradcam(tensor)

    overlay = overlay_cam(img_224.astype(float), cam)
    pred_label = "Fake" if prob >= 0.5 else "Real"
    true_label = "Fake" if label == 1 else "Real"

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_224)
    axes[0].set_title(f"Input  (True: {true_label})")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM  (Pred: {pred_label}, p={prob:.3f})")
    axes[1].axis("off")

    fig.suptitle(f"{model_name}", fontsize=11)
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(save_dir, f"{base}_gradcam.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}  (true={true_label}, pred={pred_label}, p={prob:.3f})")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = f"results/gradcam/{args.model}"
    os.makedirs(save_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {args.model} from {args.checkpoint}")
    model = build_model(args.model, freeze_backbone=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    target_layer = get_target_layer(args.model, model)
    gradcam = GradCAM(model, target_layer)

    # ── Single image mode ─────────────────────────────────────────────────────
    if args.image:
        # Infer label from path (heuristic)
        label = 1 if "fake" in args.image.lower() else 0
        print(f"\nProcessing: {args.image}")
        process_image(args.image, label, gradcam, device, args.model, save_dir)
        gradcam.remove_hooks()
        return

    # ── Sample from dataset ───────────────────────────────────────────────────
    data_dir = os.path.expanduser(args.data_dir)
    print(f"\nSampling {args.n_samples} images per class from {data_dir}")
    dataset = DeepfakeDataset(data_dir, transform=None)

    real_paths = [(p, label) for p, label in dataset.samples if label == 0]
    fake_paths = [(p, label) for p, label in dataset.samples if label == 1]

    random.seed(42)
    sample_real = random.sample(real_paths, min(args.n_samples, len(real_paths)))
    sample_fake = random.sample(fake_paths, min(args.n_samples, len(fake_paths)))

    print("\n--- Real samples ---")
    for img_path, label in sample_real:
        process_image(img_path, label, gradcam, device, args.model, save_dir)

    print("\n--- Fake samples ---")
    for img_path, label in sample_fake:
        process_image(img_path, label, gradcam, device, args.model, save_dir)

    gradcam.remove_hooks()
    print(f"\nAll Grad-CAM outputs saved to {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for deepfake detector")
    parser.add_argument("--model",      required=True,
                        choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt checkpoint from train.py")
    parser.add_argument("--image",      default=None,
                        help="Path to a single image (optional). "
                             "If omitted, samples from data/faces/.")
    parser.add_argument("--n-samples",  type=int, default=5,
                        help="Number of real + fake images to visualize (default 5 each)")
    parser.add_argument("--data-dir",  default="~/OneDrive/Desktop/DeepFake Detection_images/data/faces",
                        help="Path to faces dir containing real/ and fake/ subfolders")
    parser.add_argument("--workers",   type=int, default=0,
                        help="(unused, accepted for CLI consistency)")
    args = parser.parse_args()
    main(args)
