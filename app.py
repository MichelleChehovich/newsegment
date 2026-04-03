# app.py

import os
import io
import time
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2

MODEL_NAME = "yolov8n-seg.pt"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 ВАЖНО: модель НЕ загружаем сразу
model = None
MODEL_CLASS_NAMES = None


def get_model():
    global model, MODEL_CLASS_NAMES
    if model is None:
        print("Loading YOLO model...")
        model = YOLO(MODEL_NAME)
        MODEL_CLASS_NAMES = model.names
    return model


def pil_to_bgr(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_base64_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("encode error")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    classes: str = Form("")
):
    try:
        model = get_model()

        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        image_bgr = pil_to_bgr(pil_img)

        results = model.predict(
            source=image_bgr,
            retina_masks=True,
            verbose=False
        )

        result = results[0]

        if result.masks is None:
            img_b64 = bgr_to_base64_png(image_bgr)
            return {
                "success": True,
                "result_image_base64": img_b64
            }

        masks = result.masks.data.cpu().numpy()

        overlay = image_bgr.copy()
        overlay[masks[0].astype(bool)] = (0, 255, 0)

        result_img = cv2.addWeighted(overlay, 0.5, image_bgr, 0.5, 0)

        return {
            "success": True,
            "result_image_base64": bgr_to_base64_png(result_img)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
