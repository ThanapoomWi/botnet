"""ชั้นบริการที่ API เรียกใช้: โหลดโมเดลครั้งเดียว แล้วให้คะแนนข้อมูลจากหลายแหล่ง"""
import io
import json
import re
import threading
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import BASE, METHODS, find_default_data_dir
from .data import DEVICE_NAMES, clean_frame, discover_files, read_csv_fast, resolve_data_root
from .fusion import ALERT_LABELS, ScoreFusion, ae_only_flags, alert_levels
from .metrics import clean_json
from .models import load_models, raw_scores

MAX_UPLOAD_ROWS = 200_000
MAX_SAMPLE_ROWS = 20_000
MAX_RETURN_ROWS = 300


class ServiceError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


class DetectorService:
    def __init__(self, artifacts_dir, data_dir=None):
        self.art = Path(artifacts_dir)
        self.data_dir_override = data_dir
        self.lock = threading.Lock()
        self.reload()

    # ------------------------------------------------------------------ โหลด
    def reload(self):
        self.error = None
        self.models = self.fusion = self.features = None
        self.metrics = self.demo = None
        self.files = pd.DataFrame()
        self.files_error = None
        self._file_cache = OrderedDict()

        cfg = {}
        if (self.art / "config.json").exists():
            cfg = json.loads((self.art / "config.json").read_text(encoding="utf-8"))
        self.train_config = cfg

        # ลำดับ: NBAIOT_DATA -> โฟลเดอร์ data/ ในโปรเจกต์ -> โฟลเดอร์ที่ใช้ตอน train
        data_dir = None
        if self.data_dir_override:
            data_dir = resolve_data_root(self.data_dir_override) or Path(self.data_dir_override).expanduser()
        data_dir = data_dir or find_default_data_dir()
        if data_dir is None and cfg.get("data_dir"):
            data_dir = resolve_data_root(cfg["data_dir"])
        self.data_dir = str(Path(data_dir)) if data_dir else None

        try:
            d = joblib.load(self.art / "fusion.joblib")
            self.fusion = ScoreFusion.from_export(d)
            self.features = list(d["features"])
            self.models = load_models(self.art, d["ae_scaler"])
        except FileNotFoundError as e:
            self.error = (f"ยังไม่มีโมเดลในโฟลเดอร์ {self.art} — รัน `python train.py` ก่อน "
                          f"(หรือคัดลอกไฟล์ผลลัพธ์จาก Colab มาไว้ที่นี่) [{Path(e.filename).name if e.filename else e}]")
        except Exception as e:  # noqa: BLE001
            self.error = f"โหลดโมเดลไม่สำเร็จ: {type(e).__name__}: {e}"

        if (self.art / "metrics.json").exists():
            self.metrics = json.loads((self.art / "metrics.json").read_text(encoding="utf-8"))
        if (self.art / "demo_pool.joblib").exists():
            self.demo = joblib.load(self.art / "demo_pool.joblib")

        if self.data_dir:
            try:
                self.files = discover_files(self.data_dir)
            except Exception as e:  # noqa: BLE001
                self.files_error = str(e)

    def _require_models(self):
        if self.error or self.models is None:
            raise ServiceError(503, self.error or "โมเดลยังไม่พร้อม")

    def _check_method(self, method):
        method = method or self.fusion.default_method()
        if method not in METHODS:
            raise ServiceError(400, f"method ต้องเป็นหนึ่งใน {METHODS}")
        return method

    # ------------------------------------------------------------------ ข้อมูลสถานะ
    def status(self):
        f = self.fusion
        return clean_json({
            "ready": self.error is None and self.models is not None,
            "error": self.error,
            "artifacts_dir": str(self.art.resolve()),
            "data_dir": self.data_dir,
            "data_files": int(len(self.files)),
            "data_error": self.files_error,
            "has_metrics": self.metrics is not None,
            "has_demo": self.demo is not None,
            "methods": METHODS,
            "base": BASE,
            "default_method": f.default_method() if f else None,
            "thresholds": f.thr if f else None,
            "weights": dict(zip(BASE, f.weights.tolist())) if f else None,
            "n_features": len(self.features) if self.features else None,
            "alert_labels": ALERT_LABELS,
            "trained_at": self.train_config.get("trained_at"),
        })

    def get_metrics(self):
        if self.metrics is None:
            raise ServiceError(404, "ยังไม่มี metrics.json — รัน `python train.py` เพื่อสร้างผลวัดและกราฟ")
        return self.metrics

    def catalog(self):
        devices = {}

        def dev_entry(dev, name=None):
            dev = int(dev)
            if dev not in devices:
                devices[dev] = {"device": dev, "name": name or DEVICE_NAMES.get(dev, f"device_{dev}"),
                                "attacks": {}}
            return devices[dev]

        for r in self.files.itertuples(index=False):
            e = dev_entry(r.device, r.device_name)
            e["attacks"].setdefault(r.attack, {"attack": r.attack, "family": r.family,
                                               "file_id": None, "demo_rows": 0})["file_id"] = r.file_id
        if self.demo is not None:
            meta = self.demo["meta"]
            for (dev, att), n in meta.groupby(["device", "attack"]).size().items():
                fam = "benign" if att == "benign" else att.split(".")[0]
                e = dev_entry(dev)
                e["attacks"].setdefault(att, {"attack": att, "family": fam, "file_id": None,
                                              "demo_rows": 0})["demo_rows"] = int(n)
        out = []
        for dev in sorted(devices):
            e = devices[dev]
            e["attacks"] = sorted(e["attacks"].values(), key=lambda a: (a["family"] != "benign", a["attack"]))
            out.append(e)
        return {"devices": out, "data_dir": self.data_dir}

    # ------------------------------------------------------------------ การให้คะแนน
    def _score(self, X, method, truth=None, meta=None, notes=None):
        X = np.asarray(X, dtype=np.float32)
        finite = np.isfinite(X).all(axis=1)
        notes = list(notes or [])
        if not finite.all():
            notes.append(f"ตัดแถวที่มีค่า NaN/inf ออก {int((~finite).sum()):,} แถว")
            X = X[finite]
            truth = None if truth is None else np.asarray(truth)[finite]
            meta = None if meta is None else meta.loc[finite].reset_index(drop=True)
        if len(X) == 0:
            raise ServiceError(400, "ไม่มีแถวข้อมูลให้ประเมิน")

        with self.lock:
            raw = raw_scores(self.models, X)
        S = self.fusion.scores(raw)
        P = self.fusion.predict(S)
        thr = self.fusion.thr
        level = alert_levels(S, thr, method)
        ae_only = ae_only_flags(S, thr)
        n = len(S)

        summary = {
            "n": n, "method": method, "threshold": thr[method],
            "levels": {str(k): int((level == k).sum()) for k in ALERT_LABELS},
            "flagged": {m: int(P[m].sum()) for m in METHODS},
            "flag_rate": {m: float(P[m].mean()) for m in METHODS},
            "ae_only": int(ae_only.sum()),
            "notes": notes,
        }
        edges = np.linspace(0, 1, 21)
        hist = {"edges": edges.tolist()}
        if truth is not None:
            y = np.asarray(truth).astype(int)
            hist["benign"] = np.histogram(S[method][y == 0], bins=edges)[0].tolist()
            hist["attack"] = np.histogram(S[method][y == 1], bins=edges)[0].tolist()
            ev = {}
            for m in METHODS:
                p = P[m].to_numpy()
                tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
                fp = int(((p == 1) & (y == 0)).sum()); tn = int(((p == 0) & (y == 0)).sum())
                ev[m] = {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
                         "accuracy": (tp + tn) / n,
                         "detection_rate": tp / (tp + fn) if tp + fn else None,
                         "fpr": fp / (fp + tn) if fp + tn else None}
            summary["truth"] = {"benign": int((y == 0).sum()), "attack": int((y == 1).sum())}
            summary["eval"] = ev
        else:
            hist["all"] = np.histogram(S[method], bins=edges)[0].tolist()
        summary["hist"] = hist

        k = min(n, MAX_RETURN_ROWS)
        rows = []
        for i in range(k):
            r = {"i": i,
                 "scores": {m: float(S[m].iat[i]) for m in METHODS},
                 "flags": {m: int(P[m].iat[i]) for m in METHODS},
                 "ae_mse": float(raw["AE"].iat[i]),
                 "level": int(level[i]), "ae_only": bool(ae_only[i])}
            if truth is not None:
                r["truth"] = int(truth[i])
            if meta is not None:
                for c in ("device", "attack"):
                    if c in meta:
                        v = meta[c].iat[i]
                        r[c] = int(v) if c == "device" else str(v)
            rows.append(r)
        return clean_json({"summary": summary, "rows": rows, "rows_returned": k})

    def _features_from_frame(self, df):
        """เลือกคอลัมน์ features ตามที่โมเดลฝึกมา"""
        notes = []
        cols = [str(c) for c in df.columns]
        df.columns = cols
        missing = [c for c in self.features if c not in cols]
        if not missing:
            return df[self.features].to_numpy(np.float32), notes
        num = df.select_dtypes(include=[np.number])
        num = num.loc[:, [c for c in num.columns if c.lower() not in ("label", "y", "target")
                          and not c.startswith("Unnamed")]]
        if num.shape[1] == len(self.features):
            notes.append("ชื่อคอลัมน์ไม่ตรงกับตอนฝึก — ใช้คอลัมน์ตัวเลขตามลำดับแทน")
            return num.to_numpy(np.float32), notes
        raise ServiceError(400, f"ไฟล์ขาดคอลัมน์ {len(missing)} จาก {len(self.features)} คอลัมน์ "
                                f"เช่น {', '.join(missing[:5])} (ต้องเป็น features 115 ตัวแบบ N-BaIoT)")

    # แหล่งที่ 1: ชุด test ที่ไม่เคยถูกใช้ฝึก (บันทึกตอน train)
    def score_demo(self, device=None, attack=None, n=200, method=None, seed=None):
        self._require_models()
        method = self._check_method(method)
        if self.demo is None:
            raise ServiceError(404, "ไม่มี demo_pool.joblib — รัน `python train.py` ใหม่")
        meta = self.demo["meta"]
        mask = np.ones(len(meta), dtype=bool)
        if device is not None:
            mask &= meta["device"].to_numpy() == int(device)
        if attack:
            mask &= meta["attack"].to_numpy() == attack
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            raise ServiceError(404, "ไม่มีข้อมูลตัวอย่างสำหรับตัวเลือกนี้")
        rng = np.random.default_rng(seed)
        n = int(max(1, min(n, len(idx), MAX_SAMPLE_ROWS)))
        pick = np.sort(rng.choice(idx, size=n, replace=False))
        sub = meta.iloc[pick].reset_index(drop=True)
        out = self._score(self.demo["X"][pick], method, truth=sub["label"].to_numpy(), meta=sub)
        out["source"] = {"type": "demo", "available": int(len(idx))}
        return out

    # แหล่งที่ 2: ไฟล์ CSV ในโฟลเดอร์ข้อมูลของเรา
    def _read_data_file(self, path):
        if path in self._file_cache:
            self._file_cache.move_to_end(path)
            return self._file_cache[path]
        df = clean_frame(read_csv_fast(path))
        self._file_cache[path] = df
        while len(self._file_cache) > 3:
            self._file_cache.popitem(last=False)
        return df

    def score_dataset_file(self, file_id, n=500, method=None, seed=None):
        self._require_models()
        method = self._check_method(method)
        if self.files.empty:
            raise ServiceError(404, f"ไม่พบไฟล์ข้อมูลในโฟลเดอร์ {self.data_dir} {self.files_error or ''}".strip())
        hit = self.files[self.files["file_id"] == file_id]
        if hit.empty:
            raise ServiceError(404, f"ไม่พบไฟล์ {file_id}")
        r = hit.iloc[0]
        df = self._read_data_file(r.path)
        X, notes = self._features_from_frame(df.copy())
        rng = np.random.default_rng(seed)
        n = int(max(1, min(n, len(X), MAX_SAMPLE_ROWS)))
        pick = np.sort(rng.choice(len(X), size=n, replace=False))
        truth = np.full(n, 0 if r.family == "benign" else 1)
        meta = pd.DataFrame({"device": np.full(n, int(r.device)), "attack": np.full(n, r.attack)})
        notes.append("แถวจากไฟล์ต้นฉบับอาจรวมแถวที่เคยถูกสุ่มไปฝึกโมเดล — ถ้าต้องการผลที่ยุติธรรมให้ใช้ชุดทดสอบ")
        out = self._score(X[pick], method, truth=truth, meta=meta, notes=notes)
        out["source"] = {"type": "file", "file": str(Path(r.path).relative_to(self.data_dir))
                         if self.data_dir and str(r.path).startswith(self.data_dir) else r.path,
                         "rows_in_file": int(len(X))}
        return out

    # แหล่งที่ 3: ไฟล์ที่อัปโหลดจากหน้าเว็บ
    def score_csv_bytes(self, content, filename="upload.csv", method=None, truth_mode="auto", seed=0):
        self._require_models()
        method = self._check_method(method)
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:  # noqa: BLE001
            raise ServiceError(400, f"อ่านไฟล์ CSV ไม่ได้: {e}") from e
        if df.empty:
            raise ServiceError(400, "ไฟล์ว่าง")
        notes = []
        if len(df) > MAX_UPLOAD_ROWS:
            df = df.sample(n=MAX_UPLOAD_ROWS, random_state=seed).sort_index().reset_index(drop=True)
            notes.append(f"ไฟล์ใหญ่เกิน — สุ่มมา {MAX_UPLOAD_ROWS:,} แถว")
        truth = None
        label_col = next((c for c in df.columns if str(c).lower() in ("label", "is_attack")), None)
        if label_col is not None:
            truth = (pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy() > 0).astype(int)
        elif truth_mode in ("benign", "attack"):
            truth = np.full(len(df), 0 if truth_mode == "benign" else 1)
        elif truth_mode == "auto":
            name = filename.lower()
            if "benign" in name:
                truth = np.zeros(len(df), dtype=int)
            elif re.search(r"gafgyt|mirai|attack|combo|junk|scan|tcp|udp|ack|syn", name):
                truth = np.ones(len(df), dtype=int)
            if truth is not None:
                notes.append(f"เดาประเภทจริงจากชื่อไฟล์: {'ปกติ' if truth[0] == 0 else 'โจมตี'}")
        X, more = self._features_from_frame(df)
        out = self._score(X, method, truth=truth, notes=notes + more)
        out["source"] = {"type": "upload", "file": filename, "rows_in_file": int(len(df))}
        return out

    # แหล่งที่ 4: JSON (สำหรับระบบอื่นเรียก API)
    def score_rows(self, rows, method=None):
        self._require_models()
        method = self._check_method(method)
        if not rows:
            raise ServiceError(400, "rows ว่าง")
        if isinstance(rows[0], dict):
            df = pd.DataFrame(rows)
        else:
            arr = np.asarray(rows, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != len(self.features):
                raise ServiceError(400, f"แต่ละแถวต้องมี {len(self.features)} ค่า")
            df = pd.DataFrame(arr, columns=self.features)
        X, notes = self._features_from_frame(df)
        out = self._score(X, method, notes=notes)
        out["source"] = {"type": "api"}
        return out
