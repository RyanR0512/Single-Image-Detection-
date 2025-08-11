import tensorflow as tf
import cv2
import numpy as np
import zipfile
import os
import gdown

model_path = "yolov5m-fp16.tflite"

if not os.path.exists(model_path):
    file_id = "YOUR_FILE_ID_HERE"
    url = f"https://drive.google.com/file/d/1WgchUqXf1mrLJ8pl3l0qwgRuwcCgi2S_/view?usp=sharing"
    gdown.download(url, model_path, quiet=False)

def run_detection(img_path, model_path=r"C:\Users\rivar\PycharmProjects\sinlgeItemDetection4.0\yolov5\yolov5m-fp16.tflite"):
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

    zip_filename = "cropped_detections.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zip_buffer:
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

                label = f"Class {class_id}: {score:.2f}"
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cropped = img_resized[max(0, y1):min(y2, height), max(0, x1):min(x2, width)]
                is_success, buffer = cv2.imencode(".jpg", cropped)
                img_bytes = buffer.tobytes()

                zip_name = f"crop_{i}_class{class_id}_{int(score*100)}.jpg"
                zip_buffer.writestr(zip_name, img_bytes)

                detections_list.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "score": score,
                    "zip_name": zip_name
                })

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

    output_path = "annotated_output.jpg"
    cv2.imwrite(output_path, img_resized)
    return detections_list, output_path, ai_results
