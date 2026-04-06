import os
import io
import time
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2

# ❗ УБРАЛИ загрузку модели отсюда
from ultralytics import YOLO

MODEL_NAME = "yolov8n-seg.pt"
model = None  # ← теперь глобально пусто

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Lazy load модели
# -------------------------
def get_model():
    global model
    if model is None:
        print("🔥 Loading YOLO model...")
        model = YOLO(MODEL_NAME)
    return model

# -------------------------
# Health
# -------------------------
@app.get("/")
def root():
    return {"message": "Привет"}

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Utils
# -------------------------
def pil_to_bgr(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def bgr_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode()

# -------------------------
# Main endpoint
# -------------------------
@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    try:
        model = get_model()  # ← ЗАГРУЗКА ТОЛЬКО ЗДЕСЬ

        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        img = pil_to_bgr(pil_img)

        results = model.predict(img, imgsz=640, conf=0.25)
        result = results[0]

        if result.masks is None:
            return {"success": True, "message": "No objects found"}

        output_img = result.plot()

        return {
            "success": True,
            "image": bgr_to_base64(output_img)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
