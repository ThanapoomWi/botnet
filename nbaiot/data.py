"""อ่านชุดข้อมูล N-BaIoT จากโฟลเดอร์ในเครื่อง

รองรับ 2 รูปแบบ
1) แบบ UCI (โฟลเดอร์ที่ดาวน์โหลดจาก UCI):
       <root>/Danmini_Doorbell/benign_traffic.csv
       <root>/Danmini_Doorbell/gafgyt_attacks/combo.csv ...   (หลังแตกไฟล์ .rar)
       <root>/Danmini_Doorbell/mirai_attacks/syn.csv ...
2) แบบ Kaggle: ไฟล์ชื่อ 1.benign.csv, 1.gafgyt.combo.csv, 1.mirai.syn.csv
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# หมายเลขอุปกรณ์ตามลำดับของ Kaggle (อุปกรณ์ 3 และ 7 ไม่มีข้อมูล mirai)
DEVICE_NAMES = {
    1: "Danmini_Doorbell",
    2: "Ecobee_Thermostat",
    3: "Ennio_Doorbell",
    4: "Philips_B120N10_Baby_Monitor",
    5: "Provision_PT_737E_Security_Camera",
    6: "Provision_PT_838_Security_Camera",
    7: "Samsung_SNH_1011_N_Webcam",
    8: "SimpleHome_XCS7_1002_WHT_Security_Camera",
    9: "SimpleHome_XCS7_1003_WHT_Security_Camera",
}
_NAME_TO_ID = {v.lower(): k for k, v in DEVICE_NAMES.items()}
FLAT_RE = re.compile(r"^(\d+)\.(benign|gafgyt|mirai)(?:\.([a-z_]+))?\.csv$", re.IGNORECASE)


def _family_from_path(rel_parts, stem):
    s = stem.lower()
    if s.startswith("benign"):
        return "benign"
    joined = "/".join(p.lower() for p in rel_parts)
    if "gafgyt" in joined or "bashlite" in joined:
        return "gafgyt"
    if "mirai" in joined:
        return "mirai"
    return None


def _looks_like_root(d):
    """โฟลเดอร์นี้มีโฟลเดอร์อุปกรณ์ (แบบ UCI) หรือไฟล์ชื่อแบบ Kaggle อยู่ข้างในโดยตรงหรือไม่"""
    try:
        for c in d.iterdir():
            if c.is_dir() and c.name.lower() in _NAME_TO_ID:
                return True
            if c.is_file() and FLAT_RE.match(c.name):
                return True
    except OSError:
        pass
    return False


def resolve_data_root(path, max_depth=4):
    """หาโฟลเดอร์รากของชุดข้อมูลภายใน path (ค้นลงไปได้หลายชั้น) — ไม่เจอคืน None"""
    path = Path(path).expanduser()
    if not path.is_dir():
        return None
    level = [path]
    for _ in range(max_depth + 1):
        for d in level:
            if _looks_like_root(d):
                return d
        level = sorted(c for d in level for c in d.iterdir()
                       if c.is_dir() and not c.name.startswith((".", "__")))
        if not level:
            break
    return None


def discover_files(root):
    """คืนตารางไฟล์ข้อมูล: path, device, device_name, family, attack, file_id"""
    given = Path(root).expanduser()
    if not given.is_dir():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ข้อมูล: {given}")
    root = resolve_data_root(given) or given
    rows = []

    # แบบ Kaggle (ไฟล์ชื่อมีหมายเลขอุปกรณ์)
    for p in root.rglob("*.csv"):
        m = FLAT_RE.match(p.name)
        if m:
            dev, fam, typ = m.groups()
            fam = fam.lower()
            rows.append({"path": str(p), "device": int(dev), "family": fam,
                         "attack": "benign" if fam == "benign" else f"{fam}.{typ.lower()}"})

    # แบบ UCI (โฟลเดอร์ละอุปกรณ์)
    extra_id = 100
    for d in sorted(x for x in root.iterdir() if x.is_dir() and not x.name.startswith((".", "__"))):
        dev = _NAME_TO_ID.get(d.name.lower())
        if dev is None:
            has_csv = any(True for _ in d.rglob("*.csv"))
            if not has_csv:
                continue
            dev, extra_id = extra_id, extra_id + 1
        for p in sorted(d.rglob("*.csv")):
            rel = p.relative_to(d).parts
            if any(part.startswith((".", "__MACOSX")) for part in rel):
                continue
            fam = _family_from_path(rel[:-1], p.stem)
            if fam is None:
                continue
            attack = "benign" if fam == "benign" else f"{fam}.{p.stem.lower()}"
            rows.append({"path": str(p), "device": dev, "family": fam, "attack": attack,
                         "_dev_name": d.name})

    if not rows:
        return pd.DataFrame(columns=["path", "device", "device_name", "family", "attack", "file_id"])
    files = pd.DataFrame(rows)
    names = dict(DEVICE_NAMES)
    if "_dev_name" in files:
        for dev, nm in files.dropna(subset=["_dev_name"]).groupby("device")["_dev_name"].first().items():
            names.setdefault(int(dev), nm)
        files = files.drop(columns="_dev_name")
    files["device_name"] = files["device"].map(lambda d: names.get(int(d), f"device_{d}"))
    files = (files.drop_duplicates(subset=["device", "attack"], keep="first")
                  .sort_values(["device", "attack"]).reset_index(drop=True))
    files["file_id"] = files["device"].astype(str) + ":" + files["attack"]
    return files[["path", "device", "device_name", "family", "attack", "file_id"]]


def find_unextracted_archives(root, files=None):
    """ไฟล์ .rar/.zip ของอุปกรณ์ที่ยังไม่มี csv ตระกูลนั้น (เช่น ยังไม่ได้แตก gafgyt_attacks.rar)"""
    root = Path(root).expanduser()
    files = discover_files(root) if files is None else files
    have = set(zip(files["device"], files["family"])) if len(files) else set()
    out = []
    for arc in list(root.rglob("*.rar")) + list(root.rglob("*.zip")):
        fam = _family_from_path([], arc.stem) or _family_from_path([arc.stem], "x")
        dev = _NAME_TO_ID.get(arc.parent.name.lower())
        if fam and dev is not None and (dev, fam) not in have:
            out.append(arc)
    return out


def extract_archives(archives, log=print):
    """แตกไฟล์ .rar/.zip ไว้ในโฟลเดอร์ชื่อเดียวกัน (เช่น gafgyt_attacks.rar -> gafgyt_attacks/)
    macOS: ใช้คำสั่ง tar ที่มากับเครื่องได้เลย ถ้าไม่ได้ให้ลง `brew install unar` หรือ `brew install rar`"""
    ok = True
    for arc in archives:
        dest = arc.parent / arc.stem
        dest.mkdir(exist_ok=True)
        cmds = []
        if arc.suffix.lower() == ".zip":
            cmds.append(["unzip", "-o", "-q", str(arc), "-d", str(dest)])
        cmds += [["tar", "-xf", str(arc), "-C", str(dest)],
                 ["unar", "-q", "-f", "-o", str(dest), str(arc)],
                 ["unrar", "x", "-o+", "-inul", str(arc), str(dest) + os.sep],
                 ["7z", "x", "-y", f"-o{dest}", str(arc)]]
        done = False
        for cmd in cmds:
            if shutil.which(cmd[0]) is None:
                continue
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and any(dest.rglob("*.csv")):
                log(f"  แตกไฟล์ {arc.name} ด้วย {cmd[0]} -> {dest}")
                done = True
                break
        if not done:
            ok = False
            log(f"  ⚠️ แตกไฟล์ {arc} ไม่สำเร็จ — ลองดับเบิลคลิกใน Finder หรือ `brew install unar`")
    return ok


def read_csv_fast(path, nrows=None):
    try:
        if nrows is None:
            return pd.read_csv(path, engine="pyarrow")
    except Exception:
        pass
    return pd.read_csv(path, nrows=nrows)


def clean_frame(df, feature_cols=None):
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.select_dtypes(include=[np.number]).astype(np.float32)
    if feature_cols is not None:
        df = df[feature_cols]
    return df


def load_sample(files, cfg, log=print):
    """สุ่มแถวจากแต่ละไฟล์ (เหมือนข้อ 3 ในโน้ตบุ๊ก)"""
    rng = np.random.RandomState(cfg.seed)
    parts, feature_cols = [], None
    t0 = time.time()
    for i, r in enumerate(files.itertuples(index=False)):
        df = read_csv_fast(r.path)
        df = clean_frame(df)
        if feature_cols is None:
            feature_cols = list(df.columns)
        df = df[feature_cols]
        cap = cfg.benign_per_file if r.family == "benign" else cfg.attack_per_file
        if len(df) > cap:
            df = df.sample(n=cap, random_state=rng)
        df = df.reset_index(drop=True)
        meta = pd.DataFrame({"device": np.full(len(df), r.device, dtype=np.int16),
                             "family": r.family, "attack": r.attack,
                             "label": np.int8(r.family != "benign")}, index=df.index)
        parts.append(pd.concat([df, meta], axis=1))
        if (i + 1) % 10 == 0 or i + 1 == len(files):
            log(f"  อ่านแล้ว {i + 1}/{len(files)} ไฟล์ ({time.time() - t0:.0f}s)")
    data = pd.concat(parts, ignore_index=True)
    data["label"] = data["label"].astype(np.int8)

    bad = ~np.isfinite(data[feature_cols].to_numpy()).all(axis=1)
    if bad.any():
        log(f"  ตัดแถวที่มี inf/NaN ออก {int(bad.sum()):,} แถว")
        data = data.loc[~bad].reset_index(drop=True)
    return data, feature_cols


def split_data(df, cfg):
    """Train / Calibration / Test แบบ stratified ตามชนิดการโจมตี"""
    idx = np.arange(len(df))
    strat = df["attack"].values
    idx_rest, idx_test = train_test_split(idx, test_size=cfg.test_size, stratify=strat,
                                          random_state=cfg.seed)
    cal_frac = cfg.cal_size / (1 - cfg.test_size)
    idx_train, idx_cal = train_test_split(idx_rest, test_size=cal_frac, stratify=strat[idx_rest],
                                          random_state=cfg.seed)
    return (df.iloc[idx_train].reset_index(drop=True),
            df.iloc[idx_cal].reset_index(drop=True),
            df.iloc[idx_test].reset_index(drop=True))
