# IoT Botnet Detection Web App (แผน G)

เว็บแอป PlanG — backend เป็น **Python (FastAPI)** ใช้โมเดล Random Forest + XGBoost + Autoencoder
และตัวรวมผล (Mean / Weighted / Stacking / Vote) แบบเดียวกับโน้ตบุ๊ก ส่วน frontend เป็น HTML/JS ล้วน (ไม่ต้อง build, ใช้ได้แบบออฟไลน์)

```
iot-botnet-webapp/
├── app.py              FastAPI: API + เสิร์ฟหน้าเว็บ
├── train.py            ฝึกโมเดลจากโฟลเดอร์ข้อมูลในเครื่อง → artifacts/
├── requirements.txt
├── data/              
├── nbaiot/            
│   ├── config.py       
│   ├── data.py         อ่านโฟลเดอร์ UCI/Kaggle, สุ่มข้อมูล, แบ่ง train/cal/test
│   ├── models.py       RF / XGBoost / Autoencoder
│   ├── fusion.py       ปรับสเกลคะแนน, ตัวรวมผล, threshold, ระดับแจ้งเตือน
│   ├── metrics.py      วัดผล + ข้อมูลกราฟ
│   └── service.py      ให้คะแนนข้อมูลจากชุด test / ไฟล์ในโฟลเดอร์ / ไฟล์อัปโหลด / JSON
└── static/             index.html, app.js, styles.css
```

## เปิดเว็บ

```bash
uvicorn app:app --reload
