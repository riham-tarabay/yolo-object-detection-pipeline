# 🔍 YOLO Object Detection Pipeline

> End-to-end YOLOv8 detection system with evaluation, FastAPI serving, and Docker packaging.

## Results (COCO pretrained — YOLOv8n)

| Metric | Value |
|---|---|
| mAP@50 | 52.9% |
| mAP@50-95 | 37.3% |
| Inference speed (CPU) | ~25ms/frame |
| Parameters | 3.2M |

## Architecture
```
Input (640×640)
  → Backbone: CSPDarknet (feature extraction)
  → Neck: PANet (multi-scale feature fusion: P3, P4, P5)
  → Head: Decoupled detection heads (3 scales)
  → NMS (IoU threshold: 0.45)
  → [class, confidence, bbox] per detection
```

## Quick Start
```bash
pip install -r requirements.txt

# Single image
python yolo_detection.py

# API server
uvicorn yolo_detection:app --host 0.0.0.0 --port 8000

# Docker
docker build -t yolo-api .
docker run -p 8000:8000 yolo-api
```

## Fine-tuning on Custom Dataset
```python
detector = YOLODetector("yolov8n.pt")
detector.fine_tune("data.yaml", epochs=100)
detector.evaluate("data.yaml")   # → mAP50, mAP50-95, P, R
detector.export_onnx()           # → model.onnx for deployment
```

## What I Learned
- YOLO's single-pass architecture vs two-stage detectors (R-CNN family)
- Anchor-free detection in YOLOv8 simplifies training and improves small object detection
- mAP@50-95 is a stricter metric than mAP@50 — key for production use cases
- IoU thresholds balance precision vs recall in NMS

## Tech Stack
`Ultralytics YOLOv8` · `OpenCV` · `FastAPI` · `Docker` · `NumPy`
