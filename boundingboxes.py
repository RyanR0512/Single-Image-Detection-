import os
import zipfile
from io import BytesIO
import requests
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import hashlib

# ----------------------
# Model Validation Helpers
# ----------------------
def validate_tflite_model(path: str) -> bool:
    """
    Validate that a TFLite model can be loaded and tensors allocated.
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()  # fails if corrupted
        return True
    except Exception as e:
        st.error(f"Model validation failed: {e}")
        return False


def verify_checksum(path: str, expected_sha256: str) -> bool:
    """
    Verify SHA256 checksum of file against expected hash.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected_sha256


# ----------------------
# Public URL Model Download
# ----------------------
def download_model_url(url, local_path, expected_sha256=None):
    """
    Download TFLite model from a public URL if not already downloaded.
    Validates with interpreter load test and optional SHA256 checksum.
    """
    if os.path.exists(local_path):
        if validate_tflite_model(local_path):
            return  # already valid
        else:
            st.warning("Existing model corrupted, redownloading...")

    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        raise

    # Optional checksum check
    if expected_sha256 and not verify_checksum(local_path, expected_sha256):
        raise RuntimeError("Downloaded file checksum mismatch")

    # Final interpreter validation
    if not validate_tflite_model(local_path):
        raise RuntimeError("Downloaded file is not a valid TFLite model")


# ----------------------
# Detection Function
# ----------------------
def run_detection(img_path, model_path, conf_threshold=0.5, iou_threshold=0.4):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Original image size
    height, width, _ = img.shape

    # Preprocess for model
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Scale factors for mapping 640x640 -> original image
    scale_x = width / 640
    scale_y = height / 640

    # Extract boxes
    boxes, scores, classes = [], [], []
    for det in output:
        cx, cy, w, h = det[0:4]
        conf = det[4]
        class_probs = det[5:]
        class_id = int(np.argmax(class_probs))
        score = conf * class_probs[class_id]

        if score >= conf_threshold:
            # Convert back to original image coordinates
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            classes.append(class_id)

    # Non-Max Suppression
    """if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        indices = [i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices]
        boxes = [boxes[i] for i in indices]
        scores = [scores[i] for i in indices]
        classes = [classes[i] for i in indices]"""

    # Cropped images + zip
    detections_list, cropped_images = [], []
    zip_filename = "cropped_detections.zip"

    # Annotated copy of original image
    annotated_img = img.copy()

    with zipfile.ZipFile(zip_filename, 'w') as zip_buffer:
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box

            # Clamp coordinates to image size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Skip invalid crops
            if x2 <= x1 or y2 <= y1:
                continue

            # Draw on annotated image
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_img,
                f"Class {class_id}: {score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Crop from original image
            cropped = img[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            # Encode and save to zip
            is_success, buffer = cv2.imencode(".jpg", cropped)
            if not is_success:
                continue
            img_bytes = buffer.tobytes()
            zip_name = f"crop_{i}_class{class_id}_{int(score*100)}.jpg"
            zip_buffer.writestr(zip_name, img_bytes)
            cropped_images.append(Image.open(BytesIO(img_bytes)))

            detections_list.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": class_id,
                "score": score,
                "zip_name": zip_name
            })

    # Simple ELA AI detection
    def ai_detector_from_bytes(image_bytes, threshold=100):
        image_array = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if original is None:
            return {"score": 0, "ai_like": False}
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
    cv2.imwrite(output_path, annotated_img)

    return detections_list, output_path, ai_results, cropped_images
