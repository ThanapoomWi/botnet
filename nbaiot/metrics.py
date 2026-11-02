"""วัดผล (ข้อ 8–9 ในโน้ตบุ๊ก) และเตรียมข้อมูลกราฟเป็น JSON ให้หน้าเว็บ"""
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

from .config import BASE, METHODS


def evaluate(y, S, Pred, methods=METHODS):
    rows = []
    two = len(np.unique(y)) > 1
    for m in methods:
        tn, fp, fn, tp = confusion_matrix(y, Pred[m], labels=[0, 1]).ravel()
        rows.append({
            "Model": m,
            "Accuracy": (tp + tn) / len(y),
            "Precision": tp / max(tp + fp, 1),
            "Recall": tp / max(tp + fn, 1),
            "F1": f1_score(y, Pred[m], zero_division=0),
            "ROC-AUC": roc_auc_score(y, S[m]) if two else np.nan,
            "PR-AUC": average_precision_score(y, S[m]) if two else np.nan,
            "FPR": fp / max(fp + tn, 1),
            "FNR": fn / max(fn + tp, 1),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "TN": int(tn),
        })
    return pd.DataFrame(rows).set_index("Model")


def ensemble_gain(results, P_test, y_test, best_single):
    rows = []
    single_wrong = P_test[best_single].to_numpy() != y_test
    for m in ["Mean", "Weighted", "Stacking", "Vote"]:
        ens_wrong = P_test[m].to_numpy() != y_test
        rows.append({
            "Ensemble": m,
            "wrong_single": int(single_wrong.sum()),
            "wrong_ensemble": int(ens_wrong.sum()),
            "fixed": int((single_wrong & ~ens_wrong).sum()),
            "broken": int((~single_wrong & ens_wrong).sum()),
            "dFN": int(results.loc[m, "FN"] - results.loc[best_single, "FN"]),
            "dFP": int(results.loc[m, "FP"] - results.loc[best_single, "FP"]),
        })
    return pd.DataFrame(rows).set_index("Ensemble")


FPR_GRID = np.concatenate([[0.0], np.logspace(-5, 0, 160)])


def roc_payload(y, S, methods):
    out = {}
    for m in methods:
        fpr, tpr, _ = roc_curve(y, S[m])
        out[m] = {"fpr": FPR_GRID.tolist(), "tpr": np.interp(FPR_GRID, fpr, tpr).tolist()}
    return out


def hist_payload(y, S, methods, bins=40):
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for m in methods:
        b, _ = np.histogram(np.clip(S.loc[y == 0, m], 0, 1), bins=edges)
        a, _ = np.histogram(np.clip(S.loc[y == 1, m], 0, 1), bins=edges)
        out[m] = {"benign": b.tolist(), "attack": a.tolist()}
    return {"edges": edges.tolist(), "by_method": out}


def confusion_payload(results):
    return {m: [[int(r.TN), int(r.FP)], [int(r.FN), int(r.TP)]] for m, r in results.iterrows()}


def frame_records(df, index_name):
    d = df.reset_index().rename(columns={df.index.name or "index": index_name})
    return d.to_dict(orient="records")


def clean_json(o):
    """แปลง numpy / NaN ให้เป็นค่าที่ JSON รับได้"""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean_json(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o
