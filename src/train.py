"""SECOM defect detection: cross-validation, ROC plot and submission.

Usage (from the repo root):
    python src/train.py            # full run (a few minutes)
    python src/train.py --quick    # fast smoke test (1 repeat, fewer trees)

Expects data/train.csv and data/test.csv (download from the Kaggle page).
Outputs: reports/figures/roc_curve.png and outputs/submission.csv
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def load_data():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    y = (train["target"] == 1).astype(int).values  # raw labels are -1 (pass) / 1 (fail)
    return train, test, y


def add_time_feature(df, feats):
    X = df[feats].copy()
    t = pd.to_datetime(df["time"])
    X["t_num"] = (t - pd.Timestamp("2008-01-01")).dt.total_seconds() / 86400
    return X


def select_columns(X):
    """Drop constant columns and columns with more than 50% missing values."""
    return [c for c in X.columns if X[c].nunique() > 1 and X[c].isna().mean() < 0.5]


# --------------------------------------------------------------------------
# 2. Models (imputation lives inside each Pipeline -> no leakage across folds)
# --------------------------------------------------------------------------
def make_models(n_trees):
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "et_tuned": lambda s=0: make_pipeline(imp(), ExtraTreesClassifier(
            n_trees, min_samples_leaf=8, max_features=0.05, max_depth=10, n_jobs=-1, random_state=s)),
        "et_base": lambda s=0: make_pipeline(imp(), ExtraTreesClassifier(
            n_trees, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=s)),
        "rf": lambda s=0: make_pipeline(imp(), RandomForestClassifier(
            n_trees, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=s)),
        "logreg": lambda s=0: make_pipeline(
            imp(), QuantileTransformer(n_quantiles=200, output_distribution="normal"),
            LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000)),
    }


BLEND_WEIGHTS = {"et_tuned": 2, "et_base": 1, "rf": 1, "logreg": 1}


def rank01(p):
    """Rank-normalise scores so models on different scales can be averaged."""
    return rankdata(p) / len(p)


def blend(scores):
    return sum(w * scores[k] for k, w in BLEND_WEIGHTS.items()) / sum(BLEND_WEIGHTS.values())


# --------------------------------------------------------------------------
# 3. Cross-validation
# --------------------------------------------------------------------------
def cross_validate(X, y, models, repeats):
    names = list(models) + ["blend"]
    oof = {k: np.zeros((repeats, len(y))) for k in names}
    fold_auc = {k: [] for k in names}
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100 + r)
        for tr, va in skf.split(X, y):
            scores = {k: rank01(f().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1])
                      for k, f in models.items()}
            scores["blend"] = blend(scores)
            for k in names:
                oof[k][r, va] = scores[k]
                fold_auc[k].append(roc_auc_score(y[va], scores[k]))
    print(f"\nStratified 5-fold CV x{repeats} (mean per-fold ROC-AUC)")
    for k in names:
        print(f"  {k:10s} {np.mean(fold_auc[k]):.4f} +- {np.std(fold_auc[k]):.4f}")
    return oof


# --------------------------------------------------------------------------
# 4. ROC figure
# --------------------------------------------------------------------------
def plot_roc(oof, y, path):
    grid = np.linspace(0, 1, 201)

    def curve(k):
        tprs, aucs = [], []
        for r in range(oof[k].shape[0]):
            f, t, _ = roc_curve(y, oof[k][r])
            tprs.append(np.interp(grid, f, t))
            aucs.append(roc_auc_score(y, oof[k][r]))
        return np.mean(tprs, 0), np.min(tprs, 0), np.max(tprs, 0), np.mean(aucs)

    spec = {"blend": ("#2a78d6", "-", 2.4, "Final ensemble"),
            "et_tuned": ("#eb6834", "--", 1.8, "ExtraTrees (tuned)"),
            "logreg": ("#1baf7a", ":", 2.0, "Logistic Regression")}
    fig, ax = plt.subplots(figsize=(7, 6.2), dpi=170)
    ax.plot([0, 1], [0, 1], color="#9a9893", lw=1.2, ls=(0, (2, 3)))
    ax.text(0.60, 0.545, "Random (AUC 0.50)", color="#52514e", fontsize=9, rotation=39,
            rotation_mode="anchor", ha="center", va="top")
    _, lo, hi, _ = curve("blend")
    ax.fill_between(grid, lo, hi, color="#2a78d6", alpha=0.12, lw=0)
    for k in ("logreg", "et_tuned", "blend"):
        mean, _, _, auc = curve(k)
        col, ls, lw, name = spec[k]
        ax.plot(grid, mean, color=col, ls=ls, lw=lw, label=f"{name}  (AUC {auc:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (pass flagged as defect)")
    ax.set_ylabel("True positive rate (defects caught)")
    ax.grid(color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title("SECOM defect detection: ROC curve (out-of-fold)", loc="left", fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"Saved {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------
# 5. Final fit + submission
# --------------------------------------------------------------------------
def make_submission(X, y, Xt, ids, models, seeds):
    total = 0
    for name, w in BLEND_WEIGHTS.items():
        n = 1 if name == "logreg" else seeds
        preds = [rank01(models[name](s).fit(X, y).predict_proba(Xt)[:, 1]) for s in range(n)]
        total = total + w * np.mean(preds, axis=0)
    sub = pd.DataFrame({"id": ids, "target": total / sum(BLEND_WEIGHTS.values())})
    out = ROOT / "outputs" / "submission.csv"
    out.parent.mkdir(exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"Saved {out.relative_to(ROOT)} ({len(sub)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    args = ap.parse_args()
    repeats, n_trees, seeds = (1, 150, 1) if args.quick else (3, 600, 4)

    train, test, y = load_data()
    feats = [c for c in test.columns if c not in ("id", "time")]
    X = add_time_feature(train, feats)
    keep = select_columns(X)
    X = X[keep]
    Xt = add_time_feature(test, feats)[keep]
    print(f"Samples: {len(y)}  defects: {y.sum()} ({y.mean():.2%})  features kept: {len(keep)}")

    models = make_models(n_trees)
    oof = cross_validate(X, y, models, repeats)
    plot_roc(oof, y, ROOT / "reports" / "figures" / ("roc_curve_quick.png" if args.quick else "roc_curve.png"))
    make_submission(X, y, Xt, test["id"], models, seeds)


if __name__ == "__main__":
    main()
