"""ฝึกโมเดลแผน G จากโฟลเดอร์ข้อมูลในเครื่อง แล้วบันทึกทุกอย่างที่เว็บแอปต้องใช้ลง artifacts/

ตัวอย่าง
    python train.py                                  # ใช้ชุดข้อมูลที่วางไว้ในโฟลเดอร์ data/
    python train.py --data "/path/to/detection+of+iot+botnet+attacks+n+baiot"
    python train.py --fast                           # ลองรันเร็ว ๆ ด้วยข้อมูลน้อย (~2–4 นาที)
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nbaiot.config import BASE, DEFAULT_ARTIFACTS, DEFAULT_DATA_DIR, METHODS, TrainConfig, find_default_data_dir
from nbaiot.data import (DEVICE_NAMES, discover_files, extract_archives, find_unextracted_archives,
                         load_sample, resolve_data_root, split_data)
from nbaiot.fusion import ScoreFusion
from nbaiot.metrics import (clean_json, confusion_payload, ensemble_gain, evaluate, frame_records,
                            hist_payload, roc_payload)
from nbaiot.models import raw_scores, save_models, train_base_models

_T0 = time.time()


def tick(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def parse_args():
    d = TrainConfig()
    ap = argparse.ArgumentParser(description="ฝึกโมเดลตรวจจับ IoT botnet (RF + XGBoost + Autoencoder)")
    ap.add_argument("--data", help="โฟลเดอร์ชุดข้อมูล N-BaIoT (ค่าเริ่มต้น: โฟลเดอร์ data/ ในโปรเจกต์)")
    ap.add_argument("--out", default=str(DEFAULT_ARTIFACTS), help="โฟลเดอร์เก็บโมเดล (artifacts)")
    ap.add_argument("--benign-per-file", type=int, default=d.benign_per_file)
    ap.add_argument("--attack-per-file", type=int, default=d.attack_per_file)
    ap.add_argument("--threshold-mode", choices=["f1", "fpr"], default=d.threshold_mode)
    ap.add_argument("--target-fpr", type=float, default=d.target_fpr)
    ap.add_argument("--rf-trees", type=int, default=d.rf_trees)
    ap.add_argument("--xgb-trees", type=int, default=d.xgb_trees)
    ap.add_argument("--ae-epochs", type=int, default=d.ae_epochs)
    ap.add_argument("--no-unseen", action="store_true", help="ข้ามการทดลองการโจมตีที่ไม่เคยเห็น (เร็วขึ้นเกือบครึ่ง)")
    ap.add_argument("--holdout-family", choices=["mirai", "gafgyt"], default=d.holdout_family)
    ap.add_argument("--demo-per-attack", type=int, default=d.demo_per_attack)
    ap.add_argument("--fast", action="store_true", help="ข้อมูลน้อย + โมเดลเล็ก สำหรับลองระบบ")
    ap.add_argument("--extract", action="store_true", help="แตกไฟล์ .rar ในโฟลเดอร์ข้อมูลอัตโนมัติ")
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = TrainConfig(benign_per_file=a.benign_per_file, attack_per_file=a.attack_per_file,
                      threshold_mode=a.threshold_mode, target_fpr=a.target_fpr,
                      rf_trees=a.rf_trees, xgb_trees=a.xgb_trees, ae_epochs=a.ae_epochs,
                      run_unseen=not a.no_unseen, holdout_family=a.holdout_family,
                      demo_per_attack=a.demo_per_attack)
    if a.fast:
        cfg.benign_per_file, cfg.attack_per_file = 5_000, 500
        cfg.rf_trees, cfg.xgb_trees, cfg.ae_epochs = 50, 100, 10

    if a.data:
        data_dir = resolve_data_root(a.data) or Path(a.data).expanduser()
    else:
        data_dir = find_default_data_dir()
    if data_dir is None or not Path(data_dir).is_dir():
        sys.exit(f"ไม่พบชุดข้อมูล — วางโฟลเดอร์ N-BaIoT ไว้ใน {DEFAULT_DATA_DIR} หรือระบุด้วย --data \"/path/to/folder\"")
    data_dir = Path(data_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tick(f"โฟลเดอร์ข้อมูล: {data_dir}")

    # ---------- 1) หาไฟล์ ----------
    files = discover_files(data_dir)
    pending = find_unextracted_archives(data_dir, files)
    if pending:
        if a.extract:
            tick(f"พบไฟล์บีบอัดที่ยังไม่แตก {len(pending)} ไฟล์ — กำลังแตกไฟล์")
            extract_archives(pending, log=tick)
            files = discover_files(data_dir)
        else:
            print(f"\n⚠️ พบไฟล์ .rar ที่ยังไม่ได้แตก {len(pending)} ไฟล์ เช่น {pending[0].relative_to(data_dir)}")
            print("   เพิ่ม --extract เพื่อให้สคริปต์แตกให้ หรือแตกเองก่อน (ถ้าไม่แตก จะฝึกได้แค่ข้อมูลที่มีอยู่)\n")
    if files.empty:
        sys.exit("ไม่พบไฟล์ CSV ของ N-BaIoT ในโฟลเดอร์นี้")
    fam_counts = files.groupby("family").size().to_dict()
    tick(f"พบไฟล์ข้อมูล {len(files)} ไฟล์ จาก {files.device.nunique()} อุปกรณ์ {fam_counts}")
    if len(files) != 89:
        print("   (ชุดข้อมูลเต็มควรมี 89 ไฟล์ = 9 benign + 80 attack)")
    if (files["family"] == "benign").sum() == 0 or (files["family"] != "benign").sum() == 0:
        sys.exit("ต้องมีทั้งไฟล์ปกติ (benign) และไฟล์โจมตี")

    # ---------- 2) โหลด + สุ่ม (มี cache) ----------
    key = hashlib.md5(f"{data_dir.resolve()}|{len(files)}".encode()).hexdigest()[:8]
    cache = out / "cache" / f"sample_b{cfg.benign_per_file}_a{cfg.attack_per_file}_s{cfg.seed}_{key}.joblib"
    if cache.exists():
        data, features = joblib.load(cache)
        tick(f"โหลดข้อมูลจาก cache {cache.name}")
    else:
        tick("กำลังอ่านไฟล์ CSV (ครั้งแรกใช้เวลาหลายนาที)")
        data, features = load_sample(files, cfg, log=tick)
        cache.parent.mkdir(exist_ok=True)
        joblib.dump((data, features), cache)
    tick(f"ข้อมูลที่ใช้: {len(data):,} แถว × {len(features)} features | โจมตี {data.label.mean():.1%}")

    # ---------- 3) แบ่งข้อมูล ----------
    train_df, cal_df, test_df = split_data(data, cfg)
    for name, d in [("Train", train_df), ("Calibration", cal_df), ("Test", test_df)]:
        tick(f"{name:12s}: {len(d):>9,} แถว | โจมตี {d.label.mean():.1%}")

    # ---------- 4) ฝึกโมเดลพื้นฐาน ----------
    tick("ฝึกโมเดลพื้นฐาน 3 ตัว")
    models = train_base_models(train_df, features, cfg, log=tick)

    # ---------- 5) ตัวรวมผล (ชุด calibration) ----------
    raw_cal = raw_scores(models, cal_df[features].to_numpy(np.float32))
    fusion = ScoreFusion(cfg.threshold_mode, cfg.target_fpr, cfg.seed).fit(raw_cal, cal_df["label"])
    tick(f"ตัวรวมผลพร้อม | น้ำหนัก {dict(zip(BASE, fusion.weights.tolist()))} | ค่าเริ่มต้น = {fusion.default_method()}")

    # ---------- 6) วัดผลบนชุด test ครั้งเดียว ----------
    y_test = test_df["label"].to_numpy()
    raw_test = raw_scores(models, test_df[features].to_numpy(np.float32))
    S_test = fusion.scores(raw_test)
    P_test = fusion.predict(S_test)
    results = evaluate(y_test, S_test, P_test)
    print(results[["F1", "ROC-AUC", "FP", "FN"]].to_string(float_format=lambda v: f"{v:.6f}"))

    best_single = max(BASE, key=lambda m: fusion.cal_objective[m])
    gain = ensemble_gain(results, P_test, y_test, best_single)
    per_attack = P_test.assign(attack=test_df["attack"].values).groupby("attack")[METHODS].mean()
    per_attack.insert(0, "n", test_df["attack"].value_counts().reindex(per_attack.index).values)
    dev_rows = []
    for dev, idx in test_df.groupby("device").groups.items():
        g, yy = P_test.loc[idx], y_test[idx]
        r = {"device": int(dev), "n": len(g)}
        for m in METHODS:
            r[f"{m}_recall"] = float(g[m][yy == 1].mean()) if (yy == 1).any() else None
            r[f"{m}_fpr"] = float(g[m][yy == 0].mean()) if (yy == 0).any() else None
        dev_rows.append(r)
    per_device = pd.DataFrame(dev_rows).set_index("device")

    # ---------- 7) การทดลองเสริม: การโจมตีที่ไม่เคยเห็น ----------
    unseen = None
    if cfg.run_unseen and (data["family"] == cfg.holdout_family).any():
        tick(f"การทดลองเสริม: ซ่อน '{cfg.holdout_family}' ตอนฝึก")
        keep = lambda d: d[d["family"] != cfg.holdout_family].reset_index(drop=True)
        tr_u, cal_u = keep(train_df), keep(cal_df)
        te_u = test_df[test_df["family"].isin(["benign", cfg.holdout_family])].reset_index(drop=True)
        models_u = train_base_models(tr_u, features, cfg, log=tick, verbose=False)
        fusion_u = ScoreFusion(cfg.threshold_mode, cfg.target_fpr, cfg.seed).fit(
            raw_scores(models_u, cal_u[features].to_numpy(np.float32)), cal_u["label"])
        S_u = fusion_u.scores(raw_scores(models_u, te_u[features].to_numpy(np.float32)))
        P_u = fusion_u.predict(S_u)
        res_u = evaluate(te_u["label"].to_numpy(), S_u, P_u)
        by_type = P_u.assign(attack=te_u["attack"].values).groupby("attack")[METHODS].mean()
        unseen = {"holdout_family": cfg.holdout_family,
                  "rows": {"train": len(tr_u), "cal": len(cal_u), "test": len(te_u),
                           "unseen_attacks": int(te_u.label.sum())},
                  "weights": dict(zip(BASE, fusion_u.weights.tolist())),
                  "results": frame_records(res_u, "Model"),
                  "by_type": frame_records(by_type, "attack")}
        res_u.to_csv(out / f"results_unseen_{cfg.holdout_family}.csv")
        tick("การทดลองเสริมเสร็จ")

    # ---------- 8) บันทึก ----------
    save_models(models, out)
    joblib.dump(fusion.export(features, models["ae_scaler"]), out / "fusion.joblib")

    # ตัวอย่างจากชุด test (ไม่เคยถูกใช้ฝึก) ให้หน้าเว็บสุ่มทดสอบ
    shuffled = test_df.sample(frac=1.0, random_state=cfg.seed)
    demo = shuffled[shuffled.groupby(["device", "attack"]).cumcount() < cfg.demo_per_attack]
    demo = demo.sort_values(["device", "attack"], kind="stable").reset_index(drop=True)
    joblib.dump({"X": demo[features].to_numpy(np.float32),
                 "meta": demo[["device", "family", "attack", "label"]].reset_index(drop=True)},
                out / "demo_pool.joblib", compress=3)

    stk = fusion.stacker[-1].coef_[0]
    metrics = {
        "config": cfg.to_dict(),
        "dataset": {
            "data_dir": str(data_dir),
            "files": len(files),
            "devices": {int(k): DEVICE_NAMES.get(int(k), f"device_{k}") for k in sorted(data.device.unique())},
            "rows": len(data), "features": len(features),
            "split": {"train": len(train_df), "cal": len(cal_df), "test": len(test_df)},
            "attack_rate": float(data.label.mean()),
            "by_attack": {k: int(v) for k, v in data["attack"].value_counts().sort_index().items()},
            "by_device_family": {int(dev): {k: int(v) for k, v in g.value_counts().items()}
                                 for dev, g in data.groupby("device")["family"]},
        },
        "fusion": {"weights": dict(zip(BASE, fusion.weights.tolist())),
                   "stacking_coef": dict(zip(BASE, stk.tolist())),
                   "thresholds": fusion.thr, "cal_objective": fusion.cal_objective,
                   "default_method": fusion.default_method(), "best_single": best_single},
        "results": frame_records(results, "Model"),
        "gain": frame_records(gain, "Ensemble"),
        "per_attack": frame_records(per_attack, "attack"),
        "per_device": frame_records(per_device, "device"),
        "roc": roc_payload(y_test, S_test, BASE + ["Mean", "Weighted", "Stacking"]),
        "hist": hist_payload(y_test, S_test, METHODS),
        "confusion": confusion_payload(results),
        "ae_history": models["ae_history"],
        "unseen": unseen,
    }
    (out / "metrics.json").write_text(json.dumps(clean_json(metrics), ensure_ascii=False), encoding="utf-8")
    results.to_csv(out / "results_test.csv")
    per_attack.to_csv(out / "results_per_attack.csv")
    gain.to_csv(out / "ensemble_vs_best_single.csv")

    import sklearn
    import xgboost
    import tensorflow as tf
    (out / "config.json").write_text(json.dumps({
        **cfg.to_dict(), "data_dir": str(data_dir.resolve()),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "tensorflow": tf.__version__},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    tick(f"บันทึกไว้ที่ {out.resolve()} — เปิดเว็บด้วย: uvicorn app:app --reload")


if __name__ == "__main__":
    main()
