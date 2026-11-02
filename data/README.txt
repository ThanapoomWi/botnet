วางชุดข้อมูล N-BaIoT ไว้ในโฟลเดอร์นี้ (วางทั้งโฟลเดอร์ที่ดาวน์โหลดมาได้เลย) เช่น

data/
└── detection+of+iot+botnet+attacks+n+baiot/
    ├── Danmini_Doorbell/
    │   ├── benign_traffic.csv
    │   ├── gafgyt_attacks/   combo.csv junk.csv scan.csv tcp.csv udp.csv
    │   └── mirai_attacks/    ack.csv scan.csv syn.csv udp.csv udpplain.csv
    ├── Ecobee_Thermostat/
    └── ... (9 อุปกรณ์)

หรือวางโฟลเดอร์อุปกรณ์ไว้ตรง ๆ ใน data/ ก็ได้ หรือไฟล์แบบ Kaggle (1.benign.csv, 1.mirai.syn.csv, ...)
ถ้ายังเป็นไฟล์ .rar ให้แตกก่อน หรือรัน: python train.py --extract
