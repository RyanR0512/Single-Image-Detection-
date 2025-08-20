import streamlit as st
from PIL import Image
import gdown
import os
import boundingboxes

st.set_page_config(page_title="Single Image AI Detection", layout="centered")

MODEL_FILE_ID = "1WgchUqXf1mrLJ8pl3l0qwgRuwcCgi2S_"  # your Drive file ID
MODEL_PATH = "yolov5m-fp16.tflite"


@st.cache_resource
def get_model_path():
    """Download YOLOv5 model if missing and validate header."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        st.info("📥 Downloading YOLOv5 model from Google Drive...")
        gdown.download(url, MODEL_PATH, quiet=False)

    # Validate header
    with open(MODEL_PATH, "rb") as f:
        magic = f.read(4)
    if magic != b"TFL3":
        raise ValueError("Downloaded file is not a valid TFLite model (TFL3 header missing).")

    return MODEL_PATH


st.title("Single Image AI Detection")
st.markdown("Upload an image to detect whether it's AI-generated or human-created.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Save uploaded image temporarily
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Ensure model is ready (cached)
        model_path = get_model_path()

        # Run detection
        detections, output_img_path, ai_results, cropped_images = boundingboxes.run_detection(
            temp_path, model_path
        )

        # Show annotated image
        output_img = Image.open(output_img_path)
        st.image(output_img, caption="Processed Image with Bounding Boxes", use_container_width=True)

        # Show results
        st.subheader("Detection Results and Crops")
        for i, det in enumerate(detections):
            label = "AI" if det.get("ai_like", False) else "Human"
            st.markdown(
                f"""
                **Crop {i}**  
                - Class: `{det['class_id']}`  
                - Confidence Score: `{det['score']:.2f}`  
                - Label: **{label}**  
                - AI Score: `{det['ai_score']:.2f}`
                """
            )
            st.image(cropped_images[i], caption=f"Crop {i}", use_container_width=False)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
