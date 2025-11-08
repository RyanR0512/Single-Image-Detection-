import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load Hugging Face model for AI vs Real detection
processor = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
classifier = AutoModelForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")

# COCO class list (simplified; add all if needed)
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

# Load YOLOv5m TFLite model
TFLITE_MODEL_PATH = "yolov5m-fp16.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image_path, target_size=(640, 640)):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, target_size)
    input_data = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)
    return image, input_data


def run_yolo_detection(image_path, conf_threshold=0.3):
    image, input_data = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    detections = []

    h, w, _ = image.shape
    for det in output_data:
        score = det[4]
        if score < conf_threshold:
            continue
        class_id = int(det[5])
        x_center, y_center, box_w, box_h = det[0:4]
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)
        detections.append({
            "class_id": class_id,
            "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown",
            "bbox": (x1, y1, x2, y2),
            "score": float(score)
        })
    return image, detections


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


def run_detection(image_path):
    image, detections = run_yolo_detection(image_path)
    annotated_image = image.copy()
    crops = []

    for det in detections:
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

        crop_pil = Image.fromarray(crop)
        buf = io.BytesIO()
        crop_pil.save(buf, format="JPEG")
        crops.append(buf.getvalue())

    output_path = "output_annotated.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    return detections, output_path, True, crops
