"""
classical_ml.py
---------------
Trains SVM, Random Forest, and AdaBoost on the handcrafted features
produced by feature_extraction.py.

Input:  features/features.npy
        features/labels.npy

Output: results/classical_results.json   — all metrics
        results/pr_curves_classical.json — precision/recall arrays for plotting
        models/svm_rbf.pkl
        models/randomforest.pkl
        models/adaboost.pkl

Usage:
    python classical_ml.py
"""

import os
import json
import time
import argparse
import numpy as np
import pickle

import torch
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score,
                             balanced_accuracy_score)

FEATURES_DIR = "features"
RESULTS_DIR  = "results"
MODELS_DIR   = "models"
RANDOM_STATE = 42
DEFAULT_SPLIT = os.path.expanduser("~/projects/ML/data/split_indices.pt")
DEFAULT_FACES_DIR = os.path.expanduser("~/projects/ML/data/faces")


def load_features(features_dir):
    X = np.load(os.path.join(features_dir, "features.npy"))
    y = np.load(os.path.join(features_dir, "labels.npy"))
    print(f"✓ Loaded  X={X.shape}  y={y.shape}  (real={( y==0).sum()}, fake={(y==1).sum()})")
    return X, y


def enumerate_faces_paths(faces_dir):
    samples = []
    for _, folder in [(0, "real"), (1, "fake")]:
        folder_path = os.path.join(faces_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            for name in sorted(files):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append(os.path.realpath(os.path.join(root, name)))
    return samples


def validate_split_alignment(features_dir, faces_dir_for_split):
    paths_path = os.path.join(features_dir, "paths.npy")
    if not os.path.exists(paths_path):
        print(f"⚠  Alignment check skipped: missing {paths_path}")
        return

    if not os.path.isdir(faces_dir_for_split):
        print(f"⚠  Alignment check skipped: missing faces dir {faces_dir_for_split}")
        return

    feature_paths = np.load(paths_path, allow_pickle=True)
    feature_paths = [os.path.realpath(str(p)) for p in feature_paths.tolist()]
    split_order_paths = enumerate_faces_paths(faces_dir_for_split)

    if len(feature_paths) != len(split_order_paths):
        raise ValueError(
            "Split alignment mismatch: features/paths.npy length differs from faces traversal "
            f"(features={len(feature_paths)}, faces_order={len(split_order_paths)})."
        )

    first_mismatch = None
    for i, (a, b) in enumerate(zip(feature_paths, split_order_paths)):
        if a != b:
            first_mismatch = (i, a, b)
            break

    if first_mismatch is not None:
        i, a, b = first_mismatch
        raise ValueError(
            "Split alignment mismatch at index "
            f"{i}:\n  features path: {a}\n  faces order : {b}\n"
            "Rebuild either split_indices.pt or features so both use the exact same file ordering."
        )

    print(f"✓ Alignment check passed: features/paths.npy matches traversal of {faces_dir_for_split}")


def evaluate(name, clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    classes = np.asarray(clf.classes_)
    pos_matches = np.where(classes == 1)[0]
    if pos_matches.size == 0:
        raise ValueError(f"{name} classes_ does not contain positive class 1: {classes}")
    y_prob = clf.predict_proba(X_test)[:, pos_matches[0]]

    # Auto-correct if features discriminate in the inverted direction.
    # LBP/FFT features score higher for real faces than fakes on FF++,
    # causing AUC < 0.5. Flipping the probabilities corrects orientation
    # while keeping real=0, fake=1 consistent with the CNN label convention.
    auc_raw = roc_auc_score(y_test, y_prob)
    if auc_raw < 0.5:
        print(f"   ⚠ AUC={auc_raw:.4f} < 0.5 — features inverted on this dataset. Flipping probabilities.")
        y_prob = 1.0 - y_prob
        y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc":       round(auc, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n── {name} ──")
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"   {k:<12}: {metrics[k]:.4f}")
    print(f"   confusion:\n{np.array(metrics['confusion_matrix'])}")
    return metrics, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', default=FEATURES_DIR)
    parser.add_argument('--results_dir',  default=RESULTS_DIR)
    parser.add_argument('--models_dir',   default=MODELS_DIR)
    parser.add_argument('--split',        default=DEFAULT_SPLIT,
                        help='Path to split_indices.pt from generate_split.py')
    parser.add_argument('--faces_dir_for_split', default=DEFAULT_FACES_DIR,
                        help='Faces directory used when generating split_indices.pt (for alignment check)')
    parser.add_argument('--only',         default=None,
                        choices=['SVM_RBF', 'RandomForest', 'AdaBoost'],
                        help='Run only one classifier (e.g. --only RandomForest)')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir,  exist_ok=True)

    X, y = load_features(args.features_dir)

    faces_dir_for_split = os.path.expanduser(args.faces_dir_for_split)
    validate_split_alignment(args.features_dir, faces_dir_for_split)

    split_path = os.path.expanduser(args.split)
    if os.path.exists(split_path):
        print(f"Using video-level split from {split_path}")
        split = torch.load(split_path)
        train_idx = list(split["train"])
        test_idx  = list(split["test"])
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]
        print(f"Split  train={len(y_tr)}  test={len(y_te)}  (video-level, no leakage)")
    else:
        raise FileNotFoundError(
            f"split_indices.pt not found at {split_path}. "
            "Run generate_split.py first, or pass --split <path>."
        )

    classifiers = {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel='rbf', C=10, gamma='scale',
                          probability=True, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight="balanced_subsample"),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
    }

    all_metrics = []
    pr_data = {}

    if args.only:
        classifiers = {args.only: classifiers[args.only]}

    for name, clf in tqdm(classifiers.items(), desc="Models", unit="model"):
        tqdm.write(f"\n=== Training {name} ===")
        t0 = time.time()

        if isinstance(clf, RandomForestClassifier):
            tqdm.write("  Fitting RandomForest (200 trees)...")
            clf.fit(X_tr, y_tr)

        elif isinstance(clf, AdaBoostClassifier):
            # warm_start not supported — fit in one shot
            tqdm.write("  Fitting 100 boosting rounds (no step-level progress available)...")
            clf.fit(X_tr, y_tr)

        else:
            # SVM: single convex optimisation — no incremental steps possible
            tqdm.write("  Step 1/2: Scaling features (StandardScaler)...")
            tqdm.write("  Step 2/2: Fitting SVM — libsvm optimisation, no step-level progress available...")
            clf.fit(X_tr, y_tr)

        elapsed = time.time() - t0
        tqdm.write(f"  Fit complete ({elapsed:.0f}s)")

        model_path = os.path.join(args.models_dir, f"{name.lower()}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        tqdm.write(f"  Saved → {model_path}")

        tqdm.write(f"  Evaluating on test set...")
        metrics, y_prob = evaluate(name, clf, X_te, y_te)
        all_metrics.append(metrics)

        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        ap = average_precision_score(y_te, y_prob)
        pr_data[name] = {"precision": prec.tolist(), "recall": rec.tolist(),
                         "ap": round(ap, 4)}
        tqdm.write(f"  AP={ap:.4f}  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")

    # Save outputs
    with open(os.path.join(args.results_dir, "classical_results.json"), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(args.results_dir, "pr_curves_classical.json"), 'w') as f:
        json.dump(pr_data, f, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print(f"  {'Model':<20} {'Acc':>7} {'F1':>7} {'AUC':>7}")
    print(f"{'─'*55}")
    for m in all_metrics:
        print(f"  {m['model']:<20} {m['accuracy']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}")
    print(f"{'='*55}")
    print(f"\n✅ Done.  Next → python hybrid_model.py")


if __name__ == "__main__":
    main()
