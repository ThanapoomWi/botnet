"""FastAPI backend ของเว็บแอปตรวจจับ IoT Botnet

รัน:  uvicorn app:app --reload        แล้วเปิด http://127.0.0.1:8000
ตั้งค่าได้ด้วย environment variable
    NBAIOT_ARTIFACTS  โฟลเดอร์โมเดล (ค่าเริ่มต้น ./artifacts)
    NBAIOT_DATA       โฟลเดอร์ชุดข้อมูล (ค่าเริ่มต้น ./data)
"""
import os
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nbaiot.config import DEFAULT_ARTIFACTS
from nbaiot.service import DetectorService, ServiceError

HERE = Path(__file__).resolve().parent
service = DetectorService(os.environ.get("NBAIOT_ARTIFACTS", DEFAULT_ARTIFACTS),
                          os.environ.get("NBAIOT_DATA"))

app = FastAPI(title="IoT Botnet Detection — แผน G",
              description="RF + XGBoost + Autoencoder ensemble บนชุดข้อมูล N-BaIoT")


@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


class PredictBody(BaseModel):
    rows: List[Union[dict, List[float]]]
    method: Optional[str] = None


@app.get("/api/status")
def status():
    return service.status()


@app.post("/api/reload")
def reload():
    service.reload()
    return service.status()


@app.get("/api/metrics")
def metrics():
    return service.get_metrics()


@app.get("/api/catalog")
def catalog():
    return service.catalog()


@app.get("/api/score/demo")
def score_demo(device: Optional[int] = None, attack: Optional[str] = None, n: int = 200,
               method: Optional[str] = None, seed: Optional[int] = None):
    return service.score_demo(device=device, attack=attack or None, n=n, method=method, seed=seed)


@app.get("/api/score/file")
def score_file(file_id: str, n: int = 500, method: Optional[str] = None, seed: Optional[int] = None):
    return service.score_dataset_file(file_id, n=n, method=method, seed=seed)


@app.post("/api/score/upload")
async def score_upload(file: UploadFile = File(...), method: Optional[str] = Form(None),
                       truth: str = Form("auto")):
    content = await file.read()
    return await run_in_threadpool(service.score_csv_bytes, content, file.filename or "upload.csv",
                                   method or None, truth)


@app.post("/api/predict")
def predict(body: PredictBody):
    """ให้ระบบอื่นส่งข้อมูลมาตรวจ: rows = list ของ dict {ชื่อ feature: ค่า} หรือ list ของตัวเลข 115 ค่า"""
    return service.score_rows(body.rows, method=body.method)


# หน้าเว็บ (ต้อง mount ไว้ท้ายสุด ไม่งั้นจะทับเส้นทาง /api)
app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")
