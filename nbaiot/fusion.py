"""ปรับสเกลคะแนน + ตัวรวมผล (ข้อ 7 ในโน้ตบุ๊ก) และระดับแจ้งเตือน"""
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import BASE, ENSEMBLES

EPS = 1e-6
WEIGHT_GRID = [w for w in itertools.product(np.round(np.arange(0, 1.01, 0.1), 1), repeat=3)
               if abs(sum(w) - 1) < 1e-9]

ALERT_LABELS = {0: "ปกติ", 1: "เฝ้าระวัง", 2: "แจ้งเตือน"}


def to_features(raw):
    p = lambda s: np.clip(np.asarray(s, dtype=np.float64), EPS, 1 - EPS)
    return pd.DataFrame({
        "RF": np.log(p(raw["RF"]) / (1 - p(raw["RF"]))),
        "XGB": np.log(p(raw["XGB"]) / (1 - p(raw["XGB"]))),
        "AE": np.log(np.asarray(raw["AE"], dtype=np.float64) + 1e-12),
    })


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def best_threshold(y, s, mode="f1", target_fpr=0.01):
    """คืน (threshold, คะแนนเป้าหมาย) โดยทำนายว่าโจมตีเมื่อ s >= threshold"""
    y = np.asarray(y)
    s = np.asarray(s)
    if mode == "fpr":
        b = np.sort(s[y == 0])[::-1]
        k = int(np.floor(target_fpr * len(b)))
        thr = np.nextafter(b[min(k, len(b) - 1)], np.inf)
        return float(thr), float(np.mean(s[y == 1] >= thr))
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def _objective_of_predictions(y, pred, mode):
    y = np.asarray(y)
    pred = np.asarray(pred)
    if mode == "fpr":
        return float(pred[y == 1].mean()) if (y == 1).any() else 0.0
    return float(f1_score(y, pred, zero_division=0))


class ScoreFusion:
    def __init__(self, threshold_mode="f1", target_fpr=0.01, seed=42):
        self.mode, self.target_fpr, self.seed = threshold_mode, target_fpr, seed

    def _bt(self, y, s):
        return best_threshold(y, s, self.mode, self.target_fpr)

    def fit(self, raw_cal, y_cal):
        y = np.asarray(y_cal)
        F = to_features(raw_cal)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        # (ก) คะแนน out-of-fold สำหรับเลือกน้ำหนัก/threshold
        oof = pd.DataFrame({m: cross_val_predict(make_lr(), F[[m]], y, cv=cv, method="predict_proba")[:, 1]
                            for m in BASE})
        oof_stack = cross_val_predict(make_lr(), F[BASE], y, cv=cv, method="predict_proba")[:, 1]

        self.thr = {m: self._bt(y, oof[m])[0] for m in BASE}
        best = (-1, None)
        for w in WEIGHT_GRID:
            obj = self._bt(y, oof[BASE].to_numpy() @ np.array(w))[1]
            if obj > best[0] + 1e-9:
                best = (obj, w)
        self.weights = np.array(best[1])

        self.thr["Mean"], obj_mean = self._bt(y, oof[BASE].mean(axis=1))
        self.thr["Weighted"], obj_w = self._bt(y, oof[BASE].to_numpy() @ self.weights)
        self.thr["Stacking"], obj_s = self._bt(y, oof_stack)
        self.thr["Vote"] = 2 / 3 - 1e-9
        votes = np.mean([oof[m] >= self.thr[m] for m in BASE], axis=0)
        self.cal_objective = {m: self._bt(y, oof[m])[1] for m in BASE}
        self.cal_objective.update({"Mean": obj_mean, "Weighted": obj_w, "Stacking": obj_s,
                                   "Vote": _objective_of_predictions(y, votes >= self.thr["Vote"], self.mode)})

        # (ข) fit ตัวจริงด้วยชุด calibration ทั้งหมด
        self.calibrators = {m: make_lr().fit(F[[m]], y) for m in BASE}
        self.stacker = make_lr().fit(F[BASE], y)
        return self

    def scores(self, raw):
        F = to_features(raw)
        P = pd.DataFrame({m: self.calibrators[m].predict_proba(F[[m]])[:, 1] for m in BASE})
        S = P.copy()
        S["Mean"] = P[BASE].mean(axis=1)
        S["Weighted"] = P[BASE].to_numpy() @ self.weights
        S["Stacking"] = self.stacker.predict_proba(F[BASE])[:, 1]
        S["Vote"] = np.mean([P[m] >= self.thr[m] for m in BASE], axis=0)
        return S

    def predict(self, S):
        return pd.DataFrame({c: (S[c] >= self.thr[c]).astype(np.int8) for c in S.columns})

    def default_method(self):
        """ensemble ที่ดีที่สุดบนชุด calibration (ไม่ได้ดูชุด test)"""
        obj = getattr(self, "cal_objective", None) or {}
        cands = [m for m in ["Weighted", "Stacking", "Mean", "Vote"] if m in obj]
        return max(cands, key=lambda m: obj[m]) if cands else "Weighted"

    # ---- บันทึก/โหลด: dict ของออบเจ็กต์ scikit-learn ล้วน ๆ (รูปแบบเดียวกับ fusion.joblib ในโน้ตบุ๊ก) ----
    def export(self, features, ae_scaler):
        return {"features": list(features), "ae_scaler": ae_scaler,
                "calibrators": self.calibrators, "stacker": self.stacker,
                "weights": self.weights, "thresholds": self.thr,
                "cal_objective": self.cal_objective, "threshold_mode": self.mode,
                "target_fpr": self.target_fpr}

    @classmethod
    def from_export(cls, d):
        f = cls(d.get("threshold_mode", "f1"), d.get("target_fpr", 0.01))
        f.calibrators, f.stacker = d["calibrators"], d["stacker"]
        f.weights = np.asarray(d["weights"], dtype=float)
        f.thr = {k: float(v) for k, v in d["thresholds"].items()}
        f.cal_objective = d.get("cal_objective")
        return f


def alert_levels(S, thr, method):
    """ระดับแจ้งเตือน
    2 = แจ้งเตือน : ensemble ที่เลือกบอกว่าโจมตี และโมเดลเดี่ยวเห็นด้วยอย่างน้อย 2 ใน 3
    1 = เฝ้าระวัง : ensemble บอกว่าโจมตีแต่โมเดลเดี่ยวเห็นด้วยไม่ถึง 2 ตัว หรือ ensemble บอกว่าปกติแต่มีโมเดลเดี่ยวเตือน
    0 = ปกติ
    """
    ens = (S[method].to_numpy() >= thr[method])
    n_single = np.sum([S[m].to_numpy() >= thr[m] for m in BASE], axis=0)
    return np.where(ens & (n_single >= 2), 2, np.where(ens | (n_single >= 1), 1, 0)).astype(int)


def ae_only_flags(S, thr):
    """AE เตือนแต่ RF/XGB ไม่เตือน — อาจเป็นรูปแบบการโจมตีที่โมเดลแบบมี label ไม่เคยเห็น"""
    return ((S["AE"] >= thr["AE"]) & (S["RF"] < thr["RF"]) & (S["XGB"] < thr["XGB"])).to_numpy()


__all__ = ["ScoreFusion", "alert_levels", "ae_only_flags", "best_threshold", "ALERT_LABELS", "ENSEMBLES"]
