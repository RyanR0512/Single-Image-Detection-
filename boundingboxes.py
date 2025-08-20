import tensorflow as tf
import cv2
import numpy as np
import zipfile
import os
import gdown
from PIL import Image
from io import BytesIO

# Download model if missing
model_path = "yolov5m-fp16.tflite"
if not os.path.exists(model_path):
    file_id = "1WgchUqXf1mrLJ8pl3l0qwgRuwcCgi2S_"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)


# ----------------------------
# Preprocessing: Letterbox resize (YOLO style)
# ----------------------------
def letterbox_image(image, target_size=(640, 640)):
    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw / w, ih / h)
    nw, nh = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image, (nw, nh))

    new_image = np.full((ih, iw, 3), 128, dtype=np.uint8)
    top = (ih - nh) // 2
    left = (iw - nw) // 2
    new_image[top:top+nh, left:left+nw] = image_resized
    return new_image, scale, left, top


# ----------------------------
# Non-Maximum Suppression (NMS)
# ----------------------------
def non_max_suppression(detections, iou_threshold=0.5, score_threshold=0.25):
    boxes = []
    scores = []
    class_ids = []

    for det in detections:
        cx, cy, w, h = det[0:4]
        confidence = det[4]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        score = confidence * class_probs[class_id]

        if score > score_threshold:
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(int(class_id))

    indices = cv2.dnn.NMSBoxes(
        bboxes=np.array(boxes).tolist(),
        scores=np.array(scores).tolist(),
        score_threshold=score_threshold,
        nms_threshold=iou_threshold
    )

    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_detections.append({
                "bbox": boxes[i],
                "class_id": class_ids[i],
                "score": scores[i]
            })

    return final_detections


# ----------------------------
# Run Detection
# ----------------------------
def run_detection(img_path, model_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Failed to load image at path: " + img_path)

    # Preprocess (letterbox + normalize)
    img_resized, scale, left, top = letterbox_image(img)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Load TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Apply NMS
    detections_list = non_max_suppression(output)

    cropped_images = []
    zip_filename = "cropped_detections.zip"
    height, width, _ = img_resized.shape

    with zipfile.ZipFile(zip_filename, 'w') as zip_buffer:
        for i, det in enumerate(detections_list):
            x1, y1, x2, y2 = det["bbox"]
            class_id = det["class_id"]
            score = det["score"]

            # Convert back to original scale
            x1 = int((x1 * width - left) / scale)
            y1 = int((y1 * height - top) / scale)
            x2 = int((x2 * width - left) / scale)
            y2 = int((y2 * height - top) / scale)

            # Draw box
            label = f"Class {class_id}: {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop detection
            cropped = img[max(0, y1):min(y2, img.shape[0]),
                          max(0, x1):min(x2, img.shape[1])]

            # Save crop
            is_success, buffer = cv2.imencode(".jpg", cropped)
            img_bytes = buffer.tobytes()
            zip_name = f"crop_{i}_class{class_id}_{int(score * 100)}.jpg"
            zip_buffer.writestr(zip_name, img_bytes)

            pil_img = Image.open(BytesIO(img_bytes))
            cropped_images.append(pil_img)

            # Add details back to detection
            det["bbox"] = [x1, y1, x2, y2]
            det["zip_name"] = zip_name

    # ----------------------------
    # AI detection on crops
    # ----------------------------
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

    # Save annotated image
    output_path = "annotated_output.jpg"
    cv2.imwrite(output_path, img)

    return detections_list, output_path, ai_results, cropped_images
