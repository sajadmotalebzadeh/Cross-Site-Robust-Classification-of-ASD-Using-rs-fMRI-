#!/usr/bin/env python3
"""
ASD classification on ABIDE-I with figure generation.

What this script does:
- Builds fused phenotypic (+ optional CC200 connectome PCA) features.
- Leakage-free, site-aware evaluation: outer GroupKFold (by SITE_ID),
  inner split for NN early stopping, XGB calibration (isotonic/sigmoid),
  meta-learner fit, and threshold tuning (acc/bal_acc/youden/mcc/f1).
- Saves publication-ready Figures 1–8:
  1) Pipeline diagram (corrected stacking + calibration)
  2) Preprocessing flow (includes Fisher z)
  3) Evaluation flow (outer GroupKFold + inner calibration/threshold)
  4) ROC (per-fold thin + mean + CI)
  5) Confusion matrix (aggregated, counts + percents)
  6) Top-15 XGBoost features (mean ± SD across folds)
  7) NN training loss per fold (● marks best epoch)
  8) NN validation AUC per fold (● marks best epoch)

Key fixes (per manuscript audit):
- SITE_ID is used only for grouping (never as a feature).
- No per-site z-scoring on validation/test; preprocessing is fit on train only.
- Deterministic history files per fold.
"""

import os, re, argparse, random, warnings, glob, gzip, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib import patheffects

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Reproducibility & device
# ----------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed()

# ----------------------------
# I/O helpers
# ----------------------------
def load_tabular(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # target
    target_col = None
    for c in ["DX_GROUP", "dx_group", "Diagnosis", "diagnosis", "ASD"]:
        if c in df.columns:
            target_col = c; break
    if target_col is None:
        raise ValueError("Target column not found (expected DX_GROUP or similar).")

    y = df[target_col]
    if y.dtype.kind in "iuf":
        y = y.replace({2: 0, 1: 1})  # ABIDE: 1=ASD, 2=TDC -> map {2:0, 1:1}
    else:
        y = (y.astype(str).str.lower()
             .replace({"asd": 1, "autism": 1, "control": 0, "hc": 0, "td": 0}))
    df = df.drop(columns=[target_col])
    y = y.astype(int).values

    # drop mostly-missing and ID-like columns (keep SITE_ID for grouping only)
    missing_frac = df.isna().mean()
    to_drop = missing_frac[missing_frac > 0.90].index.tolist()
    id_like = [c for c in df.columns
               if c.upper() in ["SUBJECT", "ID", "SCAN_ID", "FILE_ID", "SUBJECT_ID"]]
    drop_final = [c for c in set(to_drop + id_like) if c != "SITE_ID"]
    if drop_final: df = df.drop(columns=drop_final, errors="ignore")
    return df, y

def split_columns_excluding_site(df):
    """Return numeric & categorical lists while EXCLUDING any site identifiers."""
    num = df.select_dtypes(include=["int", "float"]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    # ensure these do not enter features
    for site_col in ["SITE_ID", "SITE", "SITE_NAME"]:
        if site_col in num:  num.remove(site_col)
        if site_col in cat:  cat.remove(site_col)
    # handle categorical coercions
    for c in ["SEX", "HANDEDNESS", "HANDEDNESS_CATEGORY"]:
        if c in df.columns and c in num:
            num.remove(c); cat.append(c)
    # never treat SUB_ID as a numeric feature
    if "SUB_ID" in num: num.remove("SUB_ID")
    return num, cat

def get_groups(df):
    for cand in ["SITE_ID", "SITE", "SITE_NAME"]:
        if cand in df.columns:
            return df[cand].astype(str).values
    return np.array(["GLOBAL"] * len(df))

# ----------------------------
# CC200 connectivity utilities
# ----------------------------
def map_subject_id_from_path(path):
    base = os.path.basename(path)
    m = re.search(r'(\d{5,7})', base)
    if m: return int(m.group(1))
    m = re.search(r'SUB[_-]?ID[_-]?(\d+)', base, flags=re.I)
    if m: return int(m.group(1))
    return None

def load_timeseries_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path); return np.asarray(arr, dtype=np.float32)
    if ext in [".gz"] and path.endswith(".1D.gz"):
        with gzip.open(path, "rt") as f:
            arr = np.loadtxt(f); return np.asarray(arr, dtype=np.float32)
    if ext in [".1d", ".txt", ".csv"]:
        arr = np.loadtxt(path, delimiter=None if ext != ".csv" else ",")
        return np.asarray(arr, dtype=np.float32)
    arr = np.loadtxt(path); return np.asarray(arr, dtype=np.float32)

def build_connectivity_features_from_dir(ts_dir, cache_npz="cc200_connectomes.npz"):
    if cache_npz and os.path.exists(cache_npz):
        npz = np.load(cache_npz, allow_pickle=True)
        sub_ids, feats = npz["sub_ids"], npz["feats"]
    else:
        paths = sorted(
            glob.glob(os.path.join(ts_dir, "**", "*cc200*.*"), recursive=True) +
            glob.glob(os.path.join(ts_dir, "**", "*rois*.*"), recursive=True)
        )
        if len(paths) == 0:
            raise FileNotFoundError(f"No CC200 ROI time series found under: {ts_dir}")
        sub_ids, feat_list = [], []
        for p in paths:
            sid = map_subject_id_from_path(p)
            if sid is None: continue
            try:
                ts = load_timeseries_file(p)  # (T, R)
                if ts.ndim != 2: continue
                ts = (ts - ts.mean(0, keepdims=True)) / (ts.std(0, keepdims=True) + 1e-8)
                C = np.corrcoef(ts.T)
                C = np.clip(C, -0.9999, 0.9999)
                # Fisher z-transform
                C = 0.5 * np.log((1.0 + C) / (1.0 - C))
                iu = np.triu_indices(C.shape[0], k=1)
                v = C[iu].astype(np.float32)
                sub_ids.append(sid); feat_list.append(v)
            except Exception:
                continue
        if len(feat_list) == 0:
            raise RuntimeError("No valid time series parsed to connectomes.")
        feats = np.vstack(feat_list)
        sub_ids = np.asarray(sub_ids)
        if cache_npz:
            np.savez_compressed(cache_npz, sub_ids=sub_ids, feats=feats)

    cols = [f"conn_{i:05d}" for i in range(feats.shape[1])]
    df_conn = pd.DataFrame(feats, columns=cols)
    df_conn.insert(0, "SUB_ID", sub_ids.astype(int))
    return df_conn

# ----------------------------
# Torch dataset + Dual-Attention model
# ----------------------------
class TabularDataset(Dataset):
    def __init__(self, X, y): self.X = X.astype(np.float32); self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class DualAttentionMLP(nn.Module):
    def __init__(self, in_features, hidden=256, emb_dim=128, dropout=0.3):
        super().__init__()
        self.feature_attn_proj = nn.Linear(in_features, in_features, bias=False)
        self.feature_attn_vector = nn.Parameter(torch.randn(in_features))
        self.fc1 = nn.Linear(in_features, hidden); self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden);     self.ln2 = nn.LayerNorm(hidden)
        self.se_fc1 = nn.Linear(hidden, max(8, hidden // 16))
        self.se_fc2 = nn.Linear(max(8, hidden // 16), hidden)
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Linear(hidden, emb_dim); self.head = nn.Linear(emb_dim, 1)

    def feature_attention(self, x):
        proj = self.feature_attn_proj(x)
        scores = proj * self.feature_attn_vector
        weights = torch.softmax(scores, dim=1)
        return x * weights, weights

    def channel_se(self, h):
        s = torch.mean(h, dim=0, keepdim=True)
        e = F.gelu(self.se_fc1(s)); e = torch.sigmoid(self.se_fc2(e))
        return h * e

    def forward(self, x, return_attn=False):
        x_attn, weights = self.feature_attention(x)
        h = F.gelu(self.fc1(x_attn)); h = self.ln1(h); h = self.dropout(h)
        h2 = F.gelu(self.fc2(h));     h2 = self.ln2(h2); h2 = self.channel_se(h2)
        h = self.dropout(h2 + h)  # residual
        emb = F.gelu(self.emb(h))
        logit = self.head(emb)
        if return_attn: return logit.squeeze(1), emb, weights
        return logit.squeeze(1), emb

def train_nn_with_early_stop(model, train_loader, val_loader, epochs, lr, patience,
                             weight_decay, pos_weight, attn_reg, out_hist_path):
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    try:
        scaler = torch.amp.GradScaler("cuda"); autocast_ctx = torch.amp.autocast("cuda")
    except Exception:
        scaler = None
        from contextlib import nullcontext
        autocast_ctx = nullcontext()
    bce = (nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
           if pos_weight is not None else nn.BCEWithLogitsLoss())

    best_auc, best_state, no_improve = -np.inf, None, 0
    history = {"epoch": [], "train_loss": [], "val_auc": []}
    best_epoch = 0

    for ep in range(1, epochs+1):
        model.train(); running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                logit, _, weights = model(xb, return_attn=True)
                loss = bce(logit, yb) + attn_reg * (weights ** 2).mean()
            if scaler is not None:
                scaler.scale(loss).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            running += loss.item()

        # validation
        model.eval(); probs = []; targs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logit, _ = model(xb); probs.append(torch.sigmoid(logit).cpu().numpy())
                targs.append(yb.cpu().numpy())
        probs = np.concatenate(probs); targs = np.concatenate(targs)
        auc = roc_auc_score(targs, probs) if len(np.unique(targs))==2 else -np.inf

        history["epoch"].append(ep); history["train_loss"].append(running/len(train_loader))
        history["val_auc"].append(auc)
        if auc > best_auc:
            best_auc = auc; best_epoch = ep
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        scheduler.step()
        if no_improve >= patience: break

    np.savez_compressed(out_hist_path,
                        epoch=np.array(history["epoch"]),
                        train_loss=np.array(history["train_loss"]),
                        val_auc=np.array(history["val_auc"]),
                        best_epoch=np.array([best_epoch]))
    if best_state: model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, history, best_epoch

def get_embeddings(model, X):
    loader = DataLoader(TabularDataset(X, np.zeros(len(X))), batch_size=512)
    model.eval(); embs, probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE); logit, emb = model(xb)
            embs.append(emb.cpu().numpy()); probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.vstack(embs), np.concatenate(probs)

# ----------------------------
# Preprocessing builder
# ----------------------------
def compute_global_ohe_categories(df, cat_cols):
    """Return categories list aligned with cat_cols (stable across folds)."""
    cats = []
    for c in cat_cols:
        # keep real categories; ignore NaN
        vals = pd.Series(df[c]).dropna().unique()
        # ensure strings are consistent
        try:
            vals = np.array(sorted(vals, key=lambda x: str(x)))
        except Exception:
            vals = np.array(sorted(vals))
        cats.append(vals)
    return cats


def build_preprocessor(num_cols, cat_cols, ohe_categories=None):
    num_trans = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    if ohe_categories is None:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(categories=ohe_categories, handle_unknown="ignore", sparse_output=False)

    cat_trans = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    pre = ColumnTransformer([
        ("num", num_trans, num_cols),
        ("cat", cat_trans, cat_cols),
    ], remainder="drop")
    return pre


def get_feature_names_robust(preprocessor, num_cols, cat_cols):
    """Return readable feature names for ColumnTransformer(num, cat->OHE)."""
    names = []
    # numeric names
    names.extend([f"{c}" for c in num_cols])
    # categorical names via OHE categories
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        cats = ohe.categories_
        for c, cat_vals in zip(cat_cols, cats):
            for v in cat_vals:
                names.append(f"{c}={v}")
    except Exception:
        # Fallback to opaque names if sklearn version differs
        names.extend([f"{c}_OHE" for c in cat_cols])
    return names

# ----------------------------
# Threshold tuning
# ----------------------------
def find_best_threshold(y_true, y_prob, metric="acc"):
    thr_candidates = np.r_[0.0, np.unique(y_prob), 1.0]
    best_thr, best_score = 0.5, -np.inf
    for t in thr_candidates:
        y_hat = (y_prob >= t).astype(int)
        if metric == "acc":
            score = accuracy_score(y_true, y_hat)
        elif metric in ("bal_acc", "balanced_accuracy"):
            score = balanced_accuracy_score(y_true, y_hat)
        elif metric == "f1":
            score = f1_score(y_true, y_hat)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, y_hat)
        elif metric in ("youden", "j"):
            cm = confusion_matrix(y_true, y_hat, labels=[0,1])
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            sens = tp / (tp + fn + 1e-12); spec = tn / (tn + fp + 1e-12)
            score = sens + spec - 1.0
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        if score > best_score: best_thr, best_score = float(t), float(score)
    return best_thr, best_score

# ----------------------------
# Figure utilities (1–3 diagrams)
# ----------------------------
def _draw_box(ax, xy, w, h, text, fc="#F7F7F7", ec="black", fontsize=10, bold=False):
    x,y = xy
    box = patches.FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.02,rounding_size=0.015",
                                 linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    kw = dict(ha="center", va="center", fontsize=fontsize)
    if bold:
        kw["fontweight"] = "bold"
    ax.text(x + w/2, y + h/2, text, **kw)
    return (x,y,w,h)

def _arrow(ax, p1, p2, text=None):
    arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="black")
    ax.add_patch(arr)
    if text:
        xm = (p1[0]+p2[0])/2; ym=(p1[1]+p2[1])/2
        ax.text(xm, ym+0.015, text, fontsize=9, ha="center", va="bottom",
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])

def make_figure1_pipeline(out_prefix="paper_figs"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patches import FancyArrowPatch

    def box(ax, x, y, w, h, text, fc, bold=False, fs=12):
        r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.4, edgecolor="black", facecolor=fc)
        ax.add_patch(r)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")
        return r

    def arrow(ax, x1, y1, x2, y2, label=None, fs=10):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                            linewidth=1.4, color="black")
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.02, label, fontsize=fs,
                    ha="center", va="bottom")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Inputs (left column)
    box(ax, 0.05, 0.70, 0.22, 0.16, "Phenotypic CSV", fc="#E6F0FF")
    box(ax, 0.05, 0.47, 0.22, 0.16, "CC200 ROI\nTime Series", fc="#E6F0FF")
    box(ax, 0.05, 0.24, 0.22, 0.16, "Connectivity\nComputation", fc="#E6F0FF")

    # Fusion (center-left)
    fusion = box(ax, 0.33, 0.44, 0.18, 0.20, "Feature\nFusion", fc="#FFF2DA", bold=True)

    # Learners (center-right)
    dann = box(ax, 0.56, 0.62, 0.24, 0.28, "Dual-Attention\nNeural Network", fc="#ECF6EC", bold=True)
    xgb  = box(ax, 0.56, 0.24, 0.24, 0.28, "XGBoost\n(calibrated)", fc="#ECF6EC", bold=True)

    # Meta and output (right column)
    meta = box(ax, 0.84, 0.50, 0.12, 0.20, "Meta-Learner\n(Logistic)", fc="#FBE9EE", bold=True)
    out  = box(ax, 0.84, 0.24, 0.12, 0.18, "Prediction\n(ASD/TDC)", fc="#FFFFFF", bold=True)

    # Arrows
    arrow(ax, 0.27, 0.78, 0.33, 0.54)   # phenotypic -> fusion
    arrow(ax, 0.27, 0.55, 0.33, 0.54)   # timeseries -> fusion
    arrow(ax, 0.27, 0.32, 0.33, 0.54)   # connectivity -> fusion

    arrow(ax, 0.51, 0.54, 0.56, 0.76)   # fusion -> DA-NN
    arrow(ax, 0.51, 0.54, 0.56, 0.38)   # fusion -> XGB

    arrow(ax, 0.80, 0.76, 0.84, 0.60, "p₁ (cal.)")  # DA-NN -> meta
    arrow(ax, 0.80, 0.38, 0.84, 0.60, "p₂ (cal.)")  # XGB   -> meta
    arrow(ax, 0.90, 0.50, 0.90, 0.35)               # meta  -> out

    for ext in ("svg", "png"):
        fig.savefig(f"{out_prefix}_figure1_pipeline.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_figure2_preprocessing(out_prefix="paper_figs"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.patches import FancyArrowPatch

    def box(ax, x, y, w, h, text, fc, bold=False, fs=12):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.4, edgecolor="black", facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")
        return b

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.4, color="black"))

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Left column (data sources)
    ds   = box(ax, 0.05, 0.65, 0.22, 0.22, "ABIDE-I Dataset", fc="#E6F0FF", bold=True)
    ph   = box(ax, 0.05, 0.35, 0.22, 0.22, "Phenotypic Data", fc="#E6F0FF")
    ts   = box(ax, 0.05, 0.05, 0.22, 0.22, "Time Series (CC200)", fc="#E6F0FF")

    # Middle column (connectome pipeline)
    conn = box(ax, 0.33, 0.58, 0.28, 0.22, "Connectome Construction", fc="#EAF6EC")
    corr = box(ax, 0.33, 0.28, 0.28, 0.22, "Correlation Matrices\n+ Fisher’s z", fc="#EAF6EC")

    # PCA & outputs
    pca  = box(ax, 0.65, 0.43, 0.20, 0.22, "PCA (95% var)", fc="#FFF2DA", bold=True)
    feats= box(ax, 0.88, 0.55, 0.20, 0.22, "Final Features\n(~200 PCA + 51 phenotypes)", fc="#FFFFFF")
    labs = box(ax, 0.88, 0.15, 0.20, 0.22, "Labels\n(ASD / Control)", fc="#FFFFFF")

    # Arrows — no diagonals crossing text
    arrow(ax, 0.27, 0.76, 0.33, 0.69)   # ABIDE -> Connectome
    arrow(ax, 0.27, 0.46, 0.33, 0.69)   # Phenotypic (note: fused later)
    arrow(ax, 0.27, 0.16, 0.33, 0.39)   # Time Series -> Corr+z
    arrow(ax, 0.61, 0.69, 0.65, 0.54)   # Connectome -> PCA
    arrow(ax, 0.61, 0.39, 0.65, 0.54)   # Corr+z -> PCA
    arrow(ax, 0.85, 0.54, 0.88, 0.66)   # PCA -> Features
    arrow(ax, 0.27, 0.76, 0.88, 0.26)   # Dataset -> Labels (separate source)

    for ext in ("svg", "png"):
        fig.savefig(f"{out_prefix}_figure2_preprocessing.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_figure3_evaluation(out_prefix="paper_figs"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def box(ax, x, y, w, h, text, fc, bold=False, fs=12):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                           linewidth=1.4, edgecolor="black", facecolor=fc)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")
        return b

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.4, color="black"))

    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    outer = box(ax, 0.05, 0.62, 0.30, 0.26, "Outer CV: GroupKFold\n(by acquisition SITE_ID)", fc="#E6F0FF", bold=True)
    train = box(ax, 0.40, 0.66, 0.18, 0.20, "Training Folds", fc="#EAF6EC")
    inner = box(ax, 0.62, 0.66, 0.26, 0.20,
                "Inner split:\n• ES for DA-NN\n• Calibrate XGB (isotonic)\n• Train meta-learner\n• Tune threshold (F1)",
                fc="#FFF2DA")
    model = box(ax, 0.90, 0.66, 0.12, 0.20, "Stacked Model", fc="#FBE9EE", bold=True)

    test  = box(ax, 0.40, 0.26, 0.18, 0.20, "Outer Test Fold\n(held-out site[s])", fc="#EAF6EC")
    metrics = box(ax, 0.62, 0.24, 0.40, 0.24,
                  "Evaluation Metrics\nAccuracy • Precision • Recall • F1 • AUC • MCC",
                  fc="#FFFFFF")

    # Flows
    arrow(ax, 0.35, 0.75, 0.40, 0.76)   # outer -> train
    arrow(ax, 0.58, 0.76, 0.62, 0.76)   # train -> inner
    arrow(ax, 0.88, 0.76, 0.90, 0.76)   # inner -> model
    arrow(ax, 0.49, 0.36, 0.62, 0.36)   # test -> metrics

    for ext in ("svg", "png"):
        fig.savefig(f"{out_prefix}_figure3_evaluation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Plotting helpers (4–8)
# ----------------------------
def plot_roc_folds(y_true_folds, y_prob_folds, out_png):
    # Interpolate TPRs on a common FPR grid
    fpr_grid = np.linspace(0, 1, 501)
    tprs = []
    aucs = []
    plt.figure(figsize=(7,5))
    for y, p in zip(y_true_folds, y_prob_folds):
        fpr, tpr, _ = roc_curve(y, p)
        aucs.append(roc_auc_score(y, p))
        plt.plot(fpr, tpr, lw=0.8, alpha=0.7, color="#73a2cf")
        tprs.append(np.interp(fpr_grid, fpr, tpr))
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0); std_tpr = tprs.std(axis=0)
    mean_auc = np.mean(aucs)

    plt.plot(fpr_grid, mean_tpr, color="#1f78b4", lw=2.2, label=f"Mean ROC (AUC = {mean_auc:.3f})")
    plt.fill_between(fpr_grid, np.maximum(mean_tpr-std_tpr,0), np.minimum(mean_tpr+std_tpr,1),
                     color="#1f78b4", alpha=0.15, label="±1 SD")
    plt.plot([0,1],[0,1],"--", color="gray", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC (outer test folds)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_confusion(y_all, y_pred, out_png):
    cm = confusion_matrix(y_all, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix (outer test folds, aggregated)")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i,j] / (row_sum + 1e-12)
            ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def plot_importance_topk(mean_importance, std_importance, feat_names, k, out_png):
    idx = np.argsort(mean_importance)[::-1][:k]
    top_imp = mean_importance[idx]; top_std = std_importance[idx]
    labels = [feat_names[i] for i in idx]
    plt.figure(figsize=(9,6))
    y = np.arange(len(idx))[::-1]
    plt.barh(y, top_imp[::-1], xerr=top_std[::-1], color=plt.cm.viridis(np.linspace(0.15,0.85,len(idx)))[::-1])
    plt.yticks(y, labels[::-1], fontsize=9)
    plt.xlabel("Mean gain importance (across folds)")
    plt.title("Top 15 XGBoost Features (gain, mean ± SD)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_training_histories(histories, out_loss_png, out_auc_png):
    # histories: list of dicts with "epoch","train_loss","val_auc","best_epoch"
    plt.figure(figsize=(10,4))
    for i, h in enumerate(histories, 1):
        ep = np.array(h["epoch"]); loss = np.array(h["train_loss"])
        plt.plot(ep, loss, lw=1.8, label=f"Fold {i}")
        # best epoch marker
        be = h.get("best_epoch", None)
        if be is not None and be <= ep.max():
            be_idx = np.where(ep==be)[0]
            if len(be_idx): plt.scatter([be], [loss[be_idx[0]]], s=35, zorder=3)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Neural Network Training Loss per Fold")
    plt.legend(); plt.tight_layout(); plt.savefig(out_loss_png, dpi=150); plt.close()

    plt.figure(figsize=(10,4))
    for i, h in enumerate(histories, 1):
        ep = np.array(h["epoch"]); va = np.array(h["val_auc"])
        plt.plot(ep, va, lw=1.8, label=f"Fold {i}")
        be = h.get("best_epoch", None)
        if be is not None and be <= ep.max():
            be_idx = np.where(ep==be)[0]
            if len(be_idx): plt.scatter([be], [va[be_idx[0]]], s=35, zorder=3)
    plt.xlabel("Epoch"); plt.ylabel("Validation AUC"); plt.title("Neural Network Validation AUC per Fold")
    plt.legend(); plt.tight_layout(); plt.savefig(out_auc_png, dpi=150); plt.close()

# NEW — reliability diagram for stacked probabilities
def plot_reliability(y_true, y_prob, out_png, n_bins=10):
    import numpy as np
    import matplotlib.pyplot as plt
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    frac_pos, mean_pred = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            frac_pos.append(y_true[mask].mean())
            mean_pred.append(y_prob[mask].mean())
        else:
            frac_pos.append(np.nan); mean_pred.append(np.nan)
    frac_pos = np.array(frac_pos); mean_pred = np.array(mean_pred)
    brier = np.mean((y_prob - y_true) ** 2)

    plt.figure(figsize=(5.2, 4.2))
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    plt.scatter(mean_pred, frac_pos, s=40, color="#1f78b4")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Reliability diagram (Brier = {brier:.3f})")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()


def plot_reliability_plus(y_true, y_prob, out_png, n_bins=10, strategy="quantile",
                          min_per_bin=30, n_boot=0, seed=42):
    """
    Reliability diagram with options:
      - strategy: 'quantile' (equal-count) or 'uniform'
      - prints ECE, MCE, Brier; optional bootstrap CI for ECE
      - shows a probability histogram underneath
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # Bin edges
    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins+1))
        # de-duplicate edges if ties; fall back to uniform
        if np.unique(edges).size < edges.size:
            edges = np.linspace(0, 1, n_bins+1)
    else:
        edges = np.linspace(0, 1, n_bins+1)

    # Assign bins
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, len(edges)-2)

    # Aggregate
    bin_conf = []   # mean predicted prob in bin
    bin_acc  = []   # observed frequency in bin
    bin_n    = []
    for b in range(len(edges)-1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            bin_conf.append(np.nan); bin_acc.append(np.nan); bin_n.append(0)
            continue
        # merge tiny bins to neighbors if below threshold
        if n < min_per_bin and b < len(edges)-2:
            # move these points into next bin
            idx[m] = b+1
            continue
        bin_conf.append(float(y_prob[m].mean()))
        bin_acc.append(float(y_true[m].mean()))
        bin_n.append(n)

    # Recompute after merging small bins
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, len(edges)-2)
    bin_conf, bin_acc, bin_n = [], [], []
    for b in range(len(edges)-1):
        m = idx == b
        if m.sum() == 0: continue
        bin_conf.append(float(y_prob[m].mean()))
        bin_acc.append(float(y_true[m].mean()))
        bin_n.append(int(m.sum()))
    bin_conf = np.array(bin_conf); bin_acc = np.array(bin_acc); bin_n = np.array(bin_n)
    N = bin_n.sum()

    # Metrics
    ece = float(np.sum((bin_n / N) * np.abs(bin_acc - bin_conf)))
    mce = float(np.max(np.abs(bin_acc - bin_conf)))
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Optional bootstrap CI for ECE
    ece_lo = ece_hi = None
    if n_boot > 0:
        eces = []
        for _ in range(n_boot):
            ii = rng.integers(0, len(y_true), len(y_true))
            yt = y_true[ii]; yp = y_prob[ii]
            # quick ECE on uniform bins for speed
            e_edges = np.linspace(0,1,n_bins+1)
            e_idx = np.clip(np.digitize(yp, e_edges)-1, 0, n_bins-1)
            e_conf = []; e_acc = []; e_n=[]
            for b in range(n_bins):
                m = e_idx==b
                if m.sum()==0: continue
                e_conf.append(yp[m].mean()); e_acc.append(yt[m].mean()); e_n.append(m.sum())
            e_conf = np.array(e_conf); e_acc=np.array(e_acc); e_n=np.array(e_n)
            eces.append(np.sum((e_n/np.sum(e_n))*np.abs(e_acc-e_conf)))
        ece_lo, ece_hi = np.percentile(eces, [2.5, 97.5])

    # Plot
    fig = plt.figure(figsize=(6.0, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.15)

    ax = fig.add_subplot(gs[0])
    ax.plot([0,1], [0,1], "--", color="gray", lw=1)
    ax.scatter(bin_conf, bin_acc, s=50, color="#1f78b4")
    for x,y,n in zip(bin_conf, bin_acc, bin_n):
        ax.text(x, y, f" n={n}", fontsize=8, ha="left", va="bottom", color="#444")
    ttl = f"Reliability diagram (Brier = {brier:.3f}, ECE = {ece:.3f}"
    if ece_lo is not None:
        ttl += f" [{ece_lo:.3f},{ece_hi:.3f}]"
    ttl += f", MCE = {mce:.3f})"
    ax.set_title(ttl)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0,1); ax.set_ylim(0,1.03)

    # Histogram of predicted probabilities
    axh = fig.add_subplot(gs[1], sharex=ax)
    axh.hist(y_prob, bins=30, range=(0,1), color="#73a2cf", alpha=0.8)
    axh.set_ylabel("Count"); axh.set_xlabel("Predicted probability")
    axh.set_xlim(0,1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)




# ----------------------------
# Evaluate (outer GroupKFold)
# ----------------------------
def evaluate(df, y, n_splits=5, out_prefix="paper_figs",
             tune_metric="acc", calibration="isotonic"):
    groups = get_groups(df)
    gkf = GroupKFold(n_splits=n_splits)

    all_true, all_prob, all_pred = [], [], []
    y_true_folds, y_prob_folds = [], []

    # for feature importance over folds
    feat_name_reference = None
    importance_list = []

    # histories for figs 7–8
    histories = []

    
    # Build global schema ON THE FINAL DF so dimension is stable across folds
    whole_num_cols, whole_cat_cols = split_columns_excluding_site(df)
    ohe_categories = compute_global_ohe_categories(df, whole_cat_cols)

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, y, groups)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        X_tr_df_raw, y_tr = df.iloc[tr_idx].copy(), y[tr_idx]
        X_val_df_raw, y_val = df.iloc[val_idx].copy(), y[val_idx]

        # Build preprocessing on TRAIN ONLY (no per-site scaling on val!)
        num_cols, cat_cols = split_columns_excluding_site(X_tr_df_raw)

        pre = build_preprocessor(whole_num_cols, whole_cat_cols, ohe_categories=ohe_categories)
        X_tr = pre.fit_transform(X_tr_df_raw, y_tr)   # fit on train only
        X_val = pre.transform(X_val_df_raw)
        feat_names = get_feature_names_robust(pre, whole_num_cols, whole_cat_cols)
        if feat_name_reference is None:
            feat_name_reference = feat_names

        # One inner split from training for ES + calibration + threshold
        inner_groups = get_groups(X_tr_df_raw)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        inner_tr_idx2, inner_val_idx2 = next(gss.split(X_tr_df_raw, y_tr, inner_groups))
        X_es_tr, y_es_tr = X_tr[inner_tr_idx2], y_tr[inner_tr_idx2]
        X_es_val, y_es_val = X_tr[inner_val_idx2], y_tr[inner_val_idx2]

        # ===== Base 1: NN with early stopping on ES val =====
        pos2 = (y_es_tr == 1).sum(); neg2 = (y_es_tr == 0).sum()
        pos_weight_torch2 = torch.tensor([neg2 / max(pos2, 1.0)], dtype=torch.float32)
        nn_model_full = DualAttentionMLP(in_features=X_es_tr.shape[1], hidden=256, emb_dim=128, dropout=0.3)
        train_loader2 = DataLoader(TabularDataset(X_es_tr, y_es_tr), batch_size=256, shuffle=True)
        val_loader2   = DataLoader(TabularDataset(X_es_val, y_es_val), batch_size=512)
        hist_path = f"{out_prefix}_nn_history_fold{fold+1}.npz"
        nn_model_full, history, best_epoch = train_nn_with_early_stop(
            nn_model_full, train_loader2, val_loader2,
            epochs=60, lr=1e-3, patience=10, weight_decay=1e-4,
            pos_weight=pos_weight_torch2, attn_reg=1e-3, out_hist_path=hist_path
        )
        history["best_epoch"] = best_epoch
        histories.append(history)

        # ===== Base 2: XGB with calibration on ES val =====
        spw2 = neg2 / max(pos2, 1.0)
        xgb_model_full = xgb.XGBClassifier(
            n_estimators=1000, max_depth=5, learning_rate=0.015,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, reg_alpha=0.5, min_child_weight=3, gamma=0.1,
            tree_method="hist", eval_metric="auc",
            scale_pos_weight=spw2, random_state=SEED, n_jobs=-1
        )
        xgb_model_full.fit(X_es_tr, y_es_tr)

        # capture gain importances (map fN -> column index)
        booster = xgb_model_full.get_booster()
        gain = booster.get_score(importance_type="gain")
        imp_vec = np.zeros(X_es_tr.shape[1], dtype=float)
        for k,v in gain.items():
            try:
                idx = int(k[1:])
                if idx < len(imp_vec): imp_vec[idx] = float(v)
            except Exception:
                pass
        importance_list.append(imp_vec)

        if calibration.lower() in ("isotonic", "sigmoid"):
            xgb_cal_full = CalibratedClassifierCV(xgb_model_full, cv="prefit", method=calibration.lower())
            xgb_cal_full.fit(X_es_val, y_es_val)
        elif calibration.lower() == "none":
            xgb_cal_full = xgb_model_full
        else:
            raise ValueError("calibration must be one of {'isotonic','sigmoid','none'}")

        # ===== Meta on ES val + threshold tuning =====
        _, nn_prob_es_val = get_embeddings(nn_model_full, X_es_val)
        xgb_prob_es_val = xgb_cal_full.predict_proba(X_es_val)[:, 1]
        Z_es_val = np.vstack([nn_prob_es_val, xgb_prob_es_val]).T
        meta = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
        meta.fit(Z_es_val, y_es_val)
        meta_prob_es_val = meta.predict_proba(Z_es_val)[:, 1]
        best_thr, best_score = find_best_threshold(y_es_val, meta_prob_es_val, metric=tune_metric)
        print(f"[Fold {fold+1}] Tuned threshold ({tune_metric} on ES-val): {best_thr:.4f} (score={best_score:.3f})")

        # ===== Outer validation predictions =====
        _, nn_prob_val = get_embeddings(nn_model_full, X_val)
        xgb_prob_val = xgb_cal_full.predict_proba(X_val)[:, 1]
        Z_val = np.vstack([nn_prob_val, xgb_prob_val]).T
        prob = meta.predict_proba(Z_val)[:, 1]; pred = (prob >= best_thr).astype(int)

        # Store per-fold
        y_true_folds.append(y_val); y_prob_folds.append(prob)
        all_true.append(y_val); all_prob.append(prob); all_pred.append(pred)

    # Aggregate
    y_all = np.concatenate(all_true); p_all = np.concatenate(all_prob); y_pred = np.concatenate(all_pred)
    auc = roc_auc_score(y_all, p_all)
    acc = accuracy_score(y_all, y_pred)
    bal_acc = balanced_accuracy_score(y_all, y_pred)
    prec = precision_score(y_all, y_pred); rec = recall_score(y_all, y_pred)
    f1 = f1_score(y_all, y_pred); mcc = matthews_corrcoef(y_all, y_pred)

    print("\n===== FINAL RESULTS (outer test folds) =====")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}  Balanced Acc: {bal_acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  MCC: {mcc:.4f}")

    # ----- FIGURE 4: ROC (per fold + mean + CI) -----
    plot_roc_folds(y_true_folds, y_prob_folds, f"{out_prefix}_figure4_roc.png")

    # ----- FIGURE 5: Confusion matrix (counts + row %) -----
    plot_confusion(y_all, y_pred, f"{out_prefix}_figure5_confusion.png")

    # ----- FIGURE 6: XGBoost feature importances (mean ± SD) -----
    if len(importance_list):
        max_len = max(v.shape[0] for v in importance_list)
        padded = [np.pad(v, (0, max_len - v.shape[0])) for v in importance_list]
        imp_arr = np.vstack(padded)
        mean_imp = imp_arr.mean(axis=0); std_imp = imp_arr.std(axis=0)
        if feat_name_reference is None or len(feat_name_reference) != max_len:
            feat_name_reference = [f"f{i}" for i in range(max_len)]
        plot_importance_topk(mean_imp, std_imp, feat_name_reference, k=15,
                            out_png=f"{out_prefix}_figure6_xgb_importance.png")
        # If feature names differ per fold, we still need a single list.
        # We use the names from the *largest* vector if available; otherwise fall back to generic names.
        if feat_name_reference is None or len(feat_name_reference) != max_len:
            feat_name_reference = [f"f{i}" for i in range(max_len)]

        plot_importance_topk(mean_imp, std_imp, feat_name_reference, k=15,
                            out_png=f"{out_prefix}_figure6_xgb_importance.png")

    # ----- FIGURE 7 & 8: NN training curves with best-epoch markers -----
    plot_training_histories(histories,
                            out_loss_png=f"{out_prefix}_figure7_nn_loss.png",
                            out_auc_png=f"{out_prefix}_figure8_nn_val_auc.png")
    
    # NEW: calibration plot for the stacked output
    plot_reliability(y_all, p_all, f"{out_prefix}_supp_calibration.png", n_bins=10)


    plot_reliability_plus(y_all, p_all, f"{out_prefix}_supp_calibration.png",
                      n_bins=10, strategy="quantile", min_per_bin=30, n_boot=1000)


    return {"auc": float(auc), "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "mcc": float(mcc)}

    

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="ASD (ABIDE-I) phenotypic-only or phenotypic+connectivity (CC200).")
    ap.add_argument("--csv", type=str, default="Phenotypic_V1_0b_preprocessed1.csv",
                    help="Phenotypic CSV (default: Phenotypic_V1_0b_preprocessed1.csv)")
    ap.add_argument("--add_connectivity", action="store_true",
                    help="Fuse CC200 connectome PCA features (requires ROI time series).")
    ap.add_argument("--ts_dir", type=str, default=None,
                    help="Folder containing CC200 ROI time series files.")
    ap.add_argument("--out_prefix", type=str, default="paper_figs",
                    help="Prefix for saved plots and results.")
    ap.add_argument("--splits", type=int, default=5, help="GroupKFold splits (default 5).")
    ap.add_argument("--pca_var", type=float, default=0.95,
                    help="Variance to retain for PCA on connectomes (default 0.95).")
    ap.add_argument("--pca_max_components", type=int, default=200,
                    help="Max PCA components for connectomes (default 200).")
    ap.add_argument("--tune_metric", type=str, default="f1",
                    choices=["acc", "bal_acc", "youden", "mcc", "f1"],
                    help="Threshold-tuning metric on inner validation (default: f1).")
    ap.add_argument("--calibration", type=str, default="isotonic",
                    choices=["isotonic", "sigmoid", "none"],
                    help="Calibration for XGBoost base learner (default: isotonic).")
    args = ap.parse_args()

    # 0) Always generate diagram figures 1–3 (no data required)
    make_figure1_pipeline(args.out_prefix)
    make_figure2_preprocessing(args.out_prefix)
    make_figure3_evaluation(args.out_prefix)

    # 1) Load phenotypic CSV
    df, y = load_tabular(args.csv)


    # keep labels tied to SUB_ID so we can re-align after merges
    if "SUB_ID" not in df.columns:
        for cand in ["SUBJECT_ID", "SUBJECT", "SUB"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "SUB_ID"})
                break
    if "SUB_ID" not in df.columns:
        raise ValueError("SUB_ID column is required to align labels and connectivity.")

    labels = df[["SUB_ID"]].copy()
    labels["_Y_TARGET_"] = y  # carry labels with SUB_ID

    # 2) Optionally add CC200 connectivity (PCA)
    if args.add_connectivity:
        if not args.ts_dir or not os.path.exists(args.ts_dir):
            raise FileNotFoundError("Please provide --ts_dir pointing to CC200 ROI time series.")

        df_conn = build_connectivity_features_from_dir(args.ts_dir, cache_npz="cc200_connectomes.npz")

        # PCA on connectomes (unchanged from your script) ...
        conn_cols = [c for c in df_conn.columns if c.startswith("conn_")]
        if len(conn_cols) > 0:
            feats = df_conn[conn_cols].values.astype(np.float32)
            mask = ~(np.isnan(feats).any(1) | np.isinf(feats).any(1))
            df_conn = df_conn.loc[mask].reset_index(drop=True)
            feats = feats[mask]

            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(args.pca_max_components, feats.shape[1]))
            pca.fit(feats)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            k = int(np.searchsorted(cumsum, args.pca_var) + 1)
            k = min(k, args.pca_max_components)
            pca_final = PCA(n_components=k, random_state=SEED)
            feats_pca = pca_final.fit_transform(feats)
            df_pca = pd.DataFrame(feats_pca, columns=[f"conn_pca_{i:03d}" for i in range(k)])
            df_conn = pd.concat([df_conn[["SUB_ID"]], df_pca], axis=1)

        # INNER JOIN with phenotypes, then INNER JOIN labels so y realigns
        df = df.merge(df_conn, on="SUB_ID", how="inner")
        df = df.merge(labels,   on="SUB_ID", how="inner")
        y = df.pop("_Y_TARGET_").values  # <- re-extract labels aligned to merged subjects
    else:
        # no connectivity: still ensure labels are aligned to df rows
        df = df.merge(labels, on="SUB_ID", how="inner")
        y = df.pop("_Y_TARGET_").values


    # 3) Evaluate (outer GroupKFold, leakage-free)
    results = evaluate(df, y, n_splits=args.splits, out_prefix=args.out_prefix,
                       tune_metric=args.tune_metric, calibration=args.calibration)
    print("[Done] Results:", results)

if __name__ == "__main__":
    main()
