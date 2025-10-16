import tensorflow as tf
import cv2
import numpy as np
import zipfile
import os
import requests

# COCO dataset class names
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

MODEL_URL = "https://huggingface.co/RyanR0512/Yolov5m-tflite/resolve/main/yolov5m-fp16.tflite"
MODEL_PATH = "yolov5m-fp16.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from cloud...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded:", MODEL_PATH)

# ---------------- NMS HELPERS ----------------
def compute_iou(box1, boxes):
    """Compute IoU between one box and many boxes."""
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
    """Filter overlapping boxes using NMS per class."""
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

# ---------------- MAIN DETECTION ----------------
def run_detection(img_path, model_path=MODEL_PATH):
    download_model()

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to load image at path: " + img_path)

    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    detections = output[0]
    detections_list = []
    height, width, _ = img_resized.shape

    # Collect raw detections
    for i, det in enumerate(detections):
        cx, cy, w, h = det[0:4]
        confidence = det[4]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        score = confidence * class_probs[class_id]

        if score > 0.8:
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
                "score": score,
                "index": i
            })

    # 🔹 Apply NMS here
    detections_list = non_max_suppression(detections_list, iou_threshold=0.5)

    # Save crops into ZIP
    zip_filename = "cropped_detections.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zip_buffer:
        for det in detections_list:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']}: {det['score']:.2f}"

            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cropped = img_resized[max(0, y1):min(y2, height), max(0, x1):min(x2, width)]
            is_success, buffer = cv2.imencode(".jpg", cropped)
            img_bytes = buffer.tobytes()

            zip_name = f"crop_{det['index']}_{det['class_name']}_{int(det['score']*100)}.jpg"
            zip_buffer.writestr(zip_name, img_bytes)
            det["zip_name"] = zip_name

    # AI detection on crops
    def ai_detector_from_bytes(image_bytes, threshold=100):
        image_array = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        _, encoded = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, 90])
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        ela = cv2.absdiff(original, recompressed)
        ela = cv2.multiply(ela, np.array([10.0]))
        score = np.mean(ela)
        return {"score": score, "ai_like": score > threshold}

    ai_results = []
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for det in detections_list:
            with zipf.open(det["zip_name"]) as f:
                img_bytes = f.read()
                result = ai_detector_from_bytes(img_bytes)
                det["ai_score"] = result["score"]
                det["ai_like"] = result["ai_like"]
                ai_results.append(result)

    crops = []
    for det in detections_list:
        x1, y1, x2, y2 = det["bbox"]
        cropped = img_resized[max(0, y1):min(y2, height), max(0, x1):min(x2, width)]
        _, buffer = cv2.imencode(".jpg", cropped)
        det["crop_bytes"] = buffer.tobytes()  # Add raw bytes to each detection
        crops.append(det["crop_bytes"])

    output_path = "annotated_output.jpg"
    cv2.imwrite(output_path, img_resized)

    return detections_list, output_path, ai_results, crops
