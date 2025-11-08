import os
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests

# COCO CLASSES
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "dog",
    "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# YOLOv5 TFLite model info
MODEL_URL = "https://huggingface.co/RyanR0512/Yolov5m-tflite/resolve/main/yolov5m-fp16.tflite"
MODEL_PATH = "yolov5m-fp16.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading YOLOv5 model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded:", MODEL_PATH)

# Load AI/Real classifier
processor = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
classifier = AutoModelForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")

# ----------------- Helper Functions -----------------
def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, target_size)
    input_data = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)
    return image, input_data

def compute_iou(box1, boxes):
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

def classify_crop_ai_vs_real(crop_img):
    image_pil = Image.fromarray(crop_img)
    inputs = processor(image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = classifier(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = classifier.config.id2label[predicted_class_idx]
        ai_score = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_class_idx].item()
    return predicted_label, ai_score

# ----------------- Main Detection -----------------
def run_detection(image_path):
    download_model()  # ensure TFLite model exists

    image, input_data = preprocess_image(image_path)

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    detections_list = []
    h, w, _ = image.shape
    for det in output_data:
        score = det[4]
        if score < 0.3:
            continue
        class_id = int(det[5])
        x_center, y_center, box_w, box_h = det[0:4]
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)
        detections_list.append({
            "class_id": class_id,
            "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown",
            "bbox": (x1, y1, x2, y2),
            "score": float(score)
        })

    # Apply NMS
    detections_list = non_max_suppression(detections_list, iou_threshold=0.5)

    annotated_image = image.copy()
    crops = []

    for det in detections_list:
        x1, y1, x2, y2 = det["bbox"]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        predicted_label, ai_score = classify_crop_ai_vs_real(crop)
        det["ai_score"] = ai_score
        det["ai_like"] = "ai" in predicted_label.lower()

        color = (0, 0, 255) if det["ai_like"] else (0, 255, 0)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image,
                    f"{det['class_name']} ({'AI' if det['ai_like'] else 'Real'}) {ai_score:.2f}",
                    (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        buf = io.BytesIO()
        Image.fromarray(crop).save(buf, format="JPEG")
        crops.append(buf.getvalue())

    output_path = "output_annotated.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    return detections_list, output_path, True, crops
