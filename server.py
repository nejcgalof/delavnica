import os
import time
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
import uvicorn

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

MODEL_NAME = os.getenv("YOLO_MODEL", "yolo26s.pt")
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
CONF_DEFAULT = float(os.getenv("YOLO_CONF", "0.35"))
PORT = int(os.getenv("PORT", "8000"))
YOLO_THREADS = int(os.getenv("YOLO_THREADS", os.cpu_count() or 4))

torch.set_num_threads(YOLO_THREADS)

model = YOLO(MODEL_NAME, task="detect")
inference_lock = threading.Lock()

def warm_up():
    blank = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model.predict(blank, imgsz=IMGSZ, conf=0.5, verbose=False)

try:
    warm_up()
except Exception as e:
    print(f"Warm-up failed: {e}")

def infer(img: np.ndarray, conf: float):
    with inference_lock:
        start = time.perf_counter()
        results = model.predict(img, imgsz=IMGSZ, conf=conf, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    if results and len(results) > 0:
        result = results[0]
        h, w = result.orig_shape if hasattr(result, 'orig_shape') else (img.shape[0], img.shape[1])

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = x1 / w, y1 / h, x2 / w, y2 / h
            label = result.names[int(box.cls[0])]
            conf_val = float(box.conf[0])
            detections.append({
                "label": label,
                "conf": round(conf_val, 3),
                "box": [x1, y1, x2, y2]
            })

    h, w = img.shape[:2]
    return {
        "width": w,
        "height": h,
        "inference_ms": round(elapsed_ms, 1),
        "detections": detections
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def serve_index():
    return FileResponse("index.html", media_type="text/html")

@app.get("/info")
async def info():
    return {
        "model": MODEL_NAME,
        "imgsz": IMGSZ,
        "device": "cpu",
        "threads": YOLO_THREADS
    }

@app.post("/detect")
async def detect(request: Request, conf: float = CONF_DEFAULT):
    conf = max(0.01, min(0.99, conf))

    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="Body must be a JPEG or PNG image")

    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    import asyncio
    result = await asyncio.to_thread(infer, img, conf)
    return result

if __name__ == "__main__":
    cert_file = "cert.pem"
    key_file = "key.pem"

    if os.path.exists(cert_file) and os.path.exists(key_file):
        uvicorn.run(app, host="0.0.0.0", port=PORT, ssl_certfile=cert_file, ssl_keyfile=key_file)
    else:
        uvicorn.run(app, host="0.0.0.0", port=PORT)
