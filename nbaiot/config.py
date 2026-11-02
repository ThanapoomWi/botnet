"""ค่าตั้งต้นทั้งหมด (ย้ายมาจากเซลล์ "ตั้งค่าหลัก" ในโน้ตบุ๊ก)"""
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS = PROJECT_DIR / "artifacts"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"      # วางชุดข้อมูล N-BaIoT ไว้ในโฟลเดอร์นี้

BASE = ["RF", "XGB", "AE"]
ENSEMBLES = ["Mean", "Weighted", "Stacking", "Vote"]
METHODS = BASE + ENSEMBLES


@dataclass
class TrainConfig:
    seed: int = 42
    benign_per_file: int = 30_000      # ไฟล์ปกติ 9 ไฟล์ (1 ไฟล์ต่ออุปกรณ์)
    attack_per_file: int = 3_000       # ไฟล์โจมตี 80 ไฟล์
    test_size: float = 0.20
    cal_size: float = 0.20
    threshold_mode: str = "f1"         # "f1" หรือ "fpr"
    target_fpr: float = 0.01
    rf_trees: int = 100
    xgb_trees: int = 300
    ae_epochs: int = 30
    ae_batch: int = 512
    run_unseen: bool = True
    holdout_family: str = "mirai"
    demo_per_attack: int = 200         # แถวจากชุด test ต่อ (อุปกรณ์, ชนิด) ที่เก็บไว้ให้หน้าเว็บสุ่มทดสอบ

    def to_dict(self):
        return asdict(self)


def find_default_data_dir():
    """คืนโฟลเดอร์ชุดข้อมูลภายใน data/ (วางโฟลเดอร์ที่ดาวน์โหลดมาทั้งก้อน หรือวางโฟลเดอร์อุปกรณ์ตรง ๆ ก็ได้)"""
    from .data import resolve_data_root
    return resolve_data_root(DEFAULT_DATA_DIR)
