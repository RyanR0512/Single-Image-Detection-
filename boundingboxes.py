import os
import io
import cv2
import torch
import numpy as np
import requests
import zipfile
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import tensorflow as tf


# ==============================
# 1. MODEL URLS AND PATHS
# ==============================
MODEL_URL = "https://huggingface.co/RyanR0512/Yolov5m-tflite/resolve/main/yolov5m-fp16.tflite"
MODEL_PATH = "yolov5m-fp16.tflite"

# ==============================
# 2. COCO CLASSES (original + alphabetized)
# ==============================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
COCO_CLASSES_ALPHABETIZED = sorted(COCO_CLASSES)


# ==============================
# 3. DOWNLOAD YOLOV5 TFLITE MODEL IF NEEDED
# ==============================
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📦 Downloading YOLOv5 TFLite model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded:", MODEL_PATH)


# ==============================
# 4. BASIC NMS UTILITIES
# ==============================
def compute_iou(box1, boxes):
    """Compute IoU between one box and an array of boxes."""
    x1, y1, x2, y2 = box1
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])

    inter_w = np.maximum(0, xx2 - xx1)
    inter_h = np.maximum(0, yy2 - yy1)
    inter_area = inter_w * inter_h
    box_area = (x2 - x1) * (y2 - y1)
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)


def non_max_suppression(detections, iou_threshold=0.5):
    """Apply non-max suppression per class."""
    if not detections:
        return []
    boxes = np.array([d["bbox"] for d in detections])
    scores = np.array([d["score"] for d in detections])
    class_ids = np.array([d["class_id"] for d in detections])
    keep = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.argsort(-cls_scores)
        while len(cls_indices) > 0:
            best = cls_indices[0]
            keep.append(np.where(cls_mask)[0][best])
            if len(cls_indices) == 1:
                break
            ious = compute_iou(cls_boxes[best], cls_boxes[cls_indices[1:]])
            cls_indices = cls_indices[1:][ious < iou_threshold]
    return [detections[i] for i in keep]


# ==============================
# 5. LOAD AI IMAGE DETECTOR MODEL
# ==============================
print("🧠 Loading AI-vs-Real classifier (umm-maybe/AI-image-detector)...")
feature_extractor = AutoFeatureExtractor.from_pretrained("umm-maybe/AI-image-detector")
ai_model = AutoModelForImageClassification.from_pretrained("umm-maybe/AI-image-detector")
ai_model.eval()
print("✅ Loaded AI detector model.")


# ==============================
# 6. PER-CROP AI DETECTION
# ==============================
def ai_detector_from_bytes(image_bytes, threshold=0.5):
    """Classify a crop as AI-generated or human-made using umm-maybe/AI-image-detector."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = ai_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    class_id = torch.argmax(probs).item()
    confidence = probs[class_id].item()
    label = ai_model.config.id2label.get(class_id, "Unknown")

    ai_like = (label.lower().startswith("ai") or confidence > threshold)
    return {"label": label, "confidence": confidence, "ai_like": ai_like}


# ==============================
# 7. YOLO DETECTION + AI CLASSIFICATION
# ==============================
def run_detection(img_path):
    """Run YOLO object detection and AI classification per detected crop."""
    download_model()

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Failed to load image: {img_path}")
    height, width, _ = img.shape

    # Preprocess for YOLOv5
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Run inference with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    detections = output[0]

    # Parse detections
    detections_list = []
    for i, det in enumerate(detections):
        cx, cy, w, h = det[:4]
        confidence = det[4]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        score = confidence * class_probs[class_id]
        if score > 0.6:
            cx *= width
            cy *= height
            w *= width
            h *= height
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
            detections_list.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": class_name,
                "score": score
            })

    # Apply NMS
    detections_list = non_max_suppression(detections_list)

    # Draw boxes and extract crops
    crops = []
    for det in detections_list:
        x1, y1, x2, y2 = det["bbox"]
        cropped = img_resized[max(0, y1):min(y2, 640), max(0, x1):min(x2, 640)]
        is_success, buffer = cv2.imencode(".jpg", cropped)
        crop_bytes = buffer.tobytes()
        crops.append(crop_bytes)

        # AI classification
        ai_result = ai_detector_from_bytes(crop_bytes)
        det["ai_label"] = ai_result["label"]
        det["ai_score"] = ai_result["confidence"]
        det["ai_like"] = ai_result["ai_like"]

        # Draw bounding box
        color = (0, 255, 0) if not det["ai_like"] else (0, 0, 255)
        cv2.rectangle(img_resized, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det['class_name']} ({'AI' if det['ai_like'] else 'Human'})"
        cv2.putText(img_resized, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated image
    output_path = "annotated_output.jpg"
    cv2.imwrite(output_path, img_resized)

    return detections_list, output_path, None, crops
