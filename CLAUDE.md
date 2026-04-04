# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CSC 4850/6850 Machine Learning — Georgia State University**  
Team: Issac Lara, Ronald Thorpe, Kashish Rikhi

Binary classifier (real vs. deepfake) trained on **FaceForensics++** facial images. The plan compares deep learning (ResNet/EfficientNet CNNs via PyTorch/TensorFlow) against traditional ML (SVM, Decision Tree, AdaBoost on handcrafted features). Evaluation: Accuracy, Precision, Recall, F1.

## Running Scripts

```bash
# Extract frames from raw videos (reads from ~/projects/ML/FaceForensics++_C23)
python extract_frames.py

# Launch Jupyter for notebook work
jupyter notebook
```

## Data Layout

```
data/
  raw/           # Source videos by manipulation method
    original/    # Real videos
    Deepfakes/   FaceSwap/   Face2Face/   FaceShifter/   NeuralTextures/   DeepFakeDetection/
    csv/         # Per-method CSVs + FF++_Metadata.csv (full dataset index)
  frames/        # Extracted JPEGs — real/ and fake/
  faces/         # Face-cropped images — real/ and fake/
```

`extract_frames.py` maps each manipulation method to a `real` or `fake` label and writes frames to `data/frames/<label>/<method>/<video_name>/frame_NNNNN.jpg` at `frame_interval=10` (every 10th frame ≈ 1 fps at 30 fps source).

## Pipeline Stages

1. **Frame extraction** — `extract_frames.py` (done)
2. **Face detection / cropping** — populate `data/faces/`
3. **Model training** — CNN baseline + traditional ML comparison
4. **Evaluation** — metrics tables and figures for the final report

## Key Dependencies

- `opencv-python` (`cv2`) — video decoding and frame extraction
- `torch` or `tensorflow` — CNN training
- `scikit-learn` — traditional ML classifiers and metrics
- `numpy`, `matplotlib`
- `jupyter`
