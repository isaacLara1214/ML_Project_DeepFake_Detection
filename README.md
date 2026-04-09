## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| ResNet-50 | 99.46% | 99.82% | 99.52% | 99.67% | 99.98% |
| EfficientNet-B0 | 99.27% | 99.79% | 99.32% | 99.55% | 99.96% |
| SVM | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| CNN + SVM Hybrid | — | — | — | — | — |

> Test set: 23,777 images (17.8% real, 82.2% fake) — FaceForensics++
> SVM, Random Forest,

# 1. Train ResNet-50
python train.py \
  --model resnet50 \
  --data-dir ~/projects/ML/data/faces \
  --split ~/projects/ML/data/split_indices.pt \
  --epochs 10 \
  --phase1-epochs 3 \
  --batch 64 \
  --workers 4

# 2. Evaluate ResNet-50
python evaluate.py \
  --model resnet50 \
  --checkpoint models/resnet50_best.pt \
  --data-dir ~/projects/ML/data/faces \
  --split ~/projects/ML/data/split_indices.pt \
  --batch 64 \
  --workers 4

# 3. Train EfficientNet-B0
python train.py \
  --model efficientnet_b0 \
  --data-dir ~/projects/ML/data/faces \
  --split ~/projects/ML/data/split_indices.pt \
  --epochs 10 \
  --phase1-epochs 3 \
  --batch 64 \
  --workers 4

# 4. Evaluate EfficientNet-B0
python evaluate.py \
  --model efficientnet_b0 \
  --checkpoint models/efficientnet_b0_best.pt \
  --data-dir ~/projects/ML/data/faces \
  --split ~/projects/ML/data/split_indices.pt \
  --batch 64 \
  --workers 4

# 5. Grad-CAM
python gradcam.py \
  --model resnet50 \
  --checkpoint models/resnet50_best.pt \
  --data-dir ~/projects/ML/data/faces
