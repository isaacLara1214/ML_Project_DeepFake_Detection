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
import argparse
import numpy as np
import pickle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score)

FEATURES_DIR = "features"
RESULTS_DIR  = "results"
MODELS_DIR   = "models"
RANDOM_STATE = 42


def load_features(features_dir):
    X = np.load(os.path.join(features_dir, "features.npy"))
    y = np.load(os.path.join(features_dir, "labels.npy"))
    print(f"✓ Loaded  X={X.shape}  y={y.shape}  (real={( y==0).sum()}, fake={(y==1).sum()})")
    return X, y


def evaluate(name, clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_test, y_prob), 4),
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
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir,  exist_ok=True)

    X, y = load_features(args.features_dir)

    # 70 / 15 / 15 split — consistent with generate_split.py strategy
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE)
    print(f"Split  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}")

    classifiers = {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel='rbf', C=10, gamma='scale',
                          probability=True, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
    }

    all_metrics = []
    pr_data = {}

    for name, clf in classifiers.items():
        print(f"\n⏳ Training {name}...")
        clf.fit(X_tr, y_tr)

        model_path = os.path.join(args.models_dir, f"{name.lower()}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"   💾 {model_path}")

        metrics, y_prob = evaluate(name, clf, X_te, y_te)
        all_metrics.append(metrics)

        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        ap = average_precision_score(y_te, y_prob)
        pr_data[name] = {"precision": prec.tolist(), "recall": rec.tolist(),
                         "ap": round(ap, 4)}

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
