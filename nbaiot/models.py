"""โมเดลพื้นฐาน 3 ตัว (ข้อ 6 ในโน้ตบุ๊ก) + บันทึก/โหลด"""
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _keras():
    from tensorflow import keras   # import ช้า จึงโหลดเมื่อจำเป็นเท่านั้น
    return keras


def build_autoencoder(n_features):
    keras = _keras()
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),     # bottleneck
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(n_features, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def train_base_models(tr, features, cfg, log=print, verbose=True):
    import tensorflow as tf
    from xgboost import XGBClassifier
    keras = _keras()
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    X = tr[features].to_numpy(np.float32)
    y = tr["label"].to_numpy()
    models = {}

    rf = RandomForestClassifier(n_estimators=cfg.rf_trees, n_jobs=-1, random_state=cfg.seed)
    rf.fit(X, y)
    models["rf"] = rf
    if verbose:
        log("  Random Forest เสร็จ")

    xgbm = XGBClassifier(n_estimators=cfg.xgb_trees, max_depth=6, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         eval_metric="logloss", n_jobs=-1, random_state=cfg.seed)
    xgbm.fit(X, y)
    models["xgb"] = xgbm
    if verbose:
        log("  XGBoost เสร็จ")

    Xb = X[y == 0]
    scaler = StandardScaler().fit(Xb)
    Zb = scaler.transform(Xb).astype(np.float32)
    ae = build_autoencoder(len(features))
    hist = ae.fit(Zb, Zb, epochs=cfg.ae_epochs, batch_size=cfg.ae_batch, shuffle=True,
                  validation_split=0.1, verbose=2 if verbose else 0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                           restore_best_weights=True)])
    models["ae"], models["ae_scaler"] = ae, scaler
    models["ae_history"] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    if verbose:
        log(f"  Autoencoder เสร็จ (ฝึกด้วยข้อมูลปกติ {len(Xb):,} แถว)")
    return models


def raw_scores(models, X):
    """คะแนนดิบ: RF/XGB = P(attack), AE = reconstruction MSE  (X = numpy float32, คอลัมน์ตาม FEATURES)"""
    import pandas as pd
    X = np.asarray(X, dtype=np.float32)
    Z = models["ae_scaler"].transform(X).astype(np.float32)
    R = np.asarray(models["ae"].predict(Z, batch_size=4096, verbose=0))
    return pd.DataFrame({
        "RF": models["rf"].predict_proba(X)[:, 1],
        "XGB": models["xgb"].predict_proba(X)[:, 1],
        "AE": np.mean((Z.astype(np.float64) - R) ** 2, axis=1),
    })


def save_models(models, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["rf"], out / "rf.joblib", compress=3)
    models["xgb"].save_model(str(out / "xgb.json"))
    models["ae"].save(str(out / "autoencoder.keras"))


def load_models(art_dir, ae_scaler):
    """โหลดโมเดลที่บันทึกจาก train.py หรือจากข้อ 10 ของโน้ตบุ๊ก (ไฟล์ชื่อเดียวกัน)"""
    from xgboost import XGBClassifier
    keras = _keras()
    art = Path(art_dir)
    rf = joblib.load(art / "rf.joblib")
    xgbm = XGBClassifier()
    xgbm.load_model(str(art / "xgb.json"))
    ae = keras.models.load_model(str(art / "autoencoder.keras"), compile=False)
    return {"rf": rf, "xgb": xgbm, "ae": ae, "ae_scaler": ae_scaler}
