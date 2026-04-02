# =========================
# app.py
# =========================

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
# Model load
# -------------------------
print(f"Loading model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
MODEL_CLASS_NAMES = model.names  # dict: id -> class_name

# -------------------------
# Helper functions
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

    name_to_id = {str(v).lower(): int(k) for k, v in MODEL_CLASS_NAMES.items()}
    valid_names, valid_ids, invalid_names = [], [], []

    for cls_name in requested_unique:
        if cls_name in name_to_id:
            valid_names.append(cls_name)
            valid_ids.append(name_to_id[cls_name])
        else:
            invalid_names.append(cls_name)

    return requested_unique, valid_names, valid_ids, invalid_names

def overlay_masks_and_collect_stats(image_bgr, masks_xy, masks_data, boxes_cls, boxes_conf, class_names_map):
    h, w = image_bgr.shape[:2]
    image_area = h * w
    overlay = image_bgr.copy()
    output = image_bgr.copy()
    class_union_masks, class_instance_counts, class_max_conf = {}, {}, {}
    detections = []

    if masks_data is None or len(masks_data) == 0:
        return output, {
            "found_classes": [], "objects_count": 0,
            "image_width": w, "image_height": h, "detections": []
        }

    for i in range(len(masks_data)):
        cls_id = int(boxes_cls[i])
        conf = float(boxes_conf[i])
        cls_name = str(class_names_map[cls_id])
        color = PALETTE[cls_id % len(PALETTE)]
        mask = masks_data[i].astype(bool)

        if cls_name not in class_union_masks:
            class_union_masks[cls_name] = mask.copy()
            class_instance_counts[cls_name] = 1
            class_max_conf[cls_name] = conf
        else:
            class_union_masks[cls_name] |= mask
            class_instance_counts[cls_name] += 1
            class_max_conf[cls_name] = max(class_max_conf[cls_name], conf)

        overlay[mask] = (int(color[0]), int(color[1]), int(color[2]))

        if masks_xy is not None and i < len(masks_xy) and masks_xy[i] is not None and len(masks_xy[i]) > 0:
            contour = np.array(masks_xy[i], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [contour], isClosed=True, color=color, thickness=2)
            x0, y0 = contour[0][0]
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(output, (x0, max(0, y0 - 22)), (x0 + 180, y0), color, -1)
            cv2.putText(output, label, (x0 + 4, max(12, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({"class_name": cls_name, "confidence": round(conf, 4)})

    alpha = 0.45
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

    found_classes = []
    for cls_name, union_mask in class_union_masks.items():
        pixels = int(union_mask.sum())
        area_percent = round(100.0 * pixels / image_area, 2)
        found_classes.append({
            "class_name": cls_name,
            "instances": int(class_instance_counts[cls_name]),
            "area_pixels": pixels,
            "area_percent": area_percent,
            "max_confidence": round(float(class_max_conf[cls_name]), 4)
        })
    found_classes = sorted(found_classes, key=lambda x: x["area_percent"], reverse=True)

    return output, {
        "found_classes": found_classes,
        "objects_count": len(detections),
        "image_width": w,
        "image_height": h,
        "detections": detections
    }

# -------------------------
# Health endpoint
# -------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "default_classes": DEFAULT_CLASSES,
        "available_classes_count": len(MODEL_CLASS_NAMES)
    }

# -------------------------
# Segmentation endpoint
# -------------------------
@app.post("/analyze")
async def segment_image(
    image: UploadFile = File(...),
    classes: str = Form("person, car, bus, truck, bicycle, motorcycle")
):
    request_start_time = time.time()
    try:
        process_start_time = time.time()
        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        image_bgr = pil_to_bgr(pil_img)

        requested_unique, valid_names, valid_ids, invalid_names = normalize_class_list(classes)

        if not valid_ids:
            return {
                "success": False,
                "error": "No valid classes provided.",
                "requested_classes": requested_unique,
                "valid_classes": valid_names,
                "invalid_classes": invalid_names,
                "processing_time_ms": round((time.time() - process_start_time) * 1000, 2)
            }

        inference_start_time = time.time()
        results = model.predict(
            source=image_bgr,
            classes=valid_ids,
            retina_masks=True,
            verbose=False,
            imgsz=1024,
            conf=0.25
        )
        inference_time_ms = round((time.time() - inference_start_time) * 1000, 2)
        result = results[0]

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            empty_img_b64 = bgr_to_base64_png(image_bgr)
            total_processing_time_ms = round((time.time() - process_start_time) * 1000, 2)
            return {
                "success": True,
                "message": "No objects found for selected classes.",
                "requested_classes": requested_unique,
                "valid_classes": valid_names,
                "invalid_classes": invalid_names,
                "original_image_base64": empty_img_b64,
                "result_image_base64": empty_img_b64,
                "stats": {
                    "found_classes": [],
                    "objects_count": 0,
                    "image_width": image_bgr.shape[1],
                    "image_height": image_bgr.shape[0],
                    "detections": []
                },
                "processing_time": {
                    "total_ms": total_processing_time_ms,
                    "inference_ms": inference_time_ms,
                    "pre_post_ms": round(total_processing_time_ms - inference_time_ms, 2)
                }
            }

        masks_data = result.masks.data.cpu().numpy()
        masks_xy = result.masks.xy
        boxes_cls = result.boxes.cls.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()

        output_bgr, stats = overlay_masks_and_collect_stats(
            image_bgr=image_bgr,
            masks_xy=masks_xy,
            masks_data=masks_data,
            boxes_cls=boxes_cls,
            boxes_conf=boxes_conf,
            class_names_map=MODEL_CLASS_NAMES
        )

        original_b64 = bgr_to_base64_png(image_bgr)
        result_b64 = bgr_to_base64_png(output_bgr)
        total_processing_time_ms = round((time.time() - process_start_time) * 1000, 2)

        return {
            "success": True,
            "requested_classes": requested_unique,
            "valid_classes": valid_names,
            "invalid_classes": invalid_names,
            "original_image_base64": original_b64,
            "result_image_base64": result_b64,
            "stats": stats,
            "processing_time": {
                "total_ms": total_processing_time_ms,
                "inference_ms": inference_time_ms,
                "pre_post_ms": round(total_processing_time_ms - inference_time_ms, 2)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time_ms": round((time.time() - request_start_time) * 1000, 2)
        }
