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

# -------------------------
# Config
# -------------------------
MODEL_NAME = "yolov8n-seg.pt"
DEFAULT_CLASSES = ["person", "car", "bus", "truck", "bicycle", "motorcycle"]

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

app = FastAPI(title="YOLOv8 Segmentation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Lazy model loading (ВАЖНО!)
# -------------------------
model = None
MODEL_CLASS_NAMES = None

def get_model():
    global model, MODEL_CLASS_NAMES
    if model is None:
        print(f"Loading model: {MODEL_NAME}")
        model = YOLO(MODEL_NAME)
        MODEL_CLASS_NAMES = model.names
    return model

# -------------------------
# Helpers
# -------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_base64_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def normalize_class_list(classes_str: str):
    if not classes_str or not classes_str.strip():
        requested = DEFAULT_CLASSES
    else:
        requested = [x.strip().lower() for x in classes_str.split(",") if x.strip()]

    seen = set()
    requested_unique = []
    for c in requested:
        if c not in seen:
            requested_unique.append(c)
            seen.add(c)

    model = get_model()
    name_to_id = {str(v).lower(): int(k) for k, v in model.names.items()}

    valid_names, valid_ids, invalid_names = [], [], []

    for cls_name in requested_unique:
        if cls_name in name_to_id:
            valid_names.append(cls_name)
            valid_ids.append(name_to_id[cls_name])
        else:
            invalid_names.append(cls_name)

    return requested_unique, valid_names, valid_ids, invalid_names

# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------
# Main endpoint
# -------------------------
@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    classes: str = Form("person, car, bus, truck, bicycle, motorcycle")
):
    start_time = time.time()

    try:
        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        image_bgr = pil_to_bgr(pil_img)

        requested, valid_names, valid_ids, invalid = normalize_class_list(classes)

        if not valid_ids:
            return {
                "success": False,
                "error": "No valid classes",
                "requested_classes": requested,
                "valid_classes": valid_names,
                "invalid_classes": invalid
            }

        model = get_model()

        results = model.predict(
            source=image_bgr,
            classes=valid_ids,
            retina_masks=True,
            verbose=False
        )

        result = results[0]

        if result.masks is None:
            img_b64 = bgr_to_base64_png(image_bgr)
            return {
                "success": True,
                "message": "No objects found",
                "image": img_b64
            }

        masks = result.masks.data.cpu().numpy()
        overlay = image_bgr.copy()

        for i, mask in enumerate(masks):
            color = PALETTE[i % len(PALETTE)]
            overlay[mask.astype(bool)] = color

        output = cv2.addWeighted(overlay, 0.5, image_bgr, 0.5, 0)

        result_b64 = bgr_to_base64_png(output)

        return {
            "success": True,
            "result_image_base64": result_b64,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
