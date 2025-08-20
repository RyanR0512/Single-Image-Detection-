import streamlit as st
from PIL import Image
import boundingboxes  # your updated module with safer model validation

st.set_page_config(page_title="Single Image AI Detection", layout="centered")
st.title("Single Image AI Detection")
st.markdown("Upload an image to detect whether it's AI-generated or human-created.")

# ----------------------
# Public URL Model
# ----------------------
model_url = "https://singleimageitemdetector.s3.us-east-2.amazonaws.com/yolov5m-fp16.tflite"
model_path = "yolov5m-fp16.tflite"

# Optional SHA256 checksum (replace with known hash if available)
expected_sha256 = None
# Example: expected_sha256 = "7c3c59e9b6c7d71f82462f066f2dbe3e4f87a9e2f05c0a6539a43b1dcb8c7e3d"

# Download & validate model
boundingboxes.download_model_url(model_url, model_path, expected_sha256=expected_sha256)

# ----------------------
# File Upload
# ----------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run detection
        detections, output_img_path, ai_results, cropped_images = boundingboxes.run_detection(
            temp_path, model_path
        )

        # Show annotated image
        st.image(
            Image.open(output_img_path),
            caption="Processed Image with Bounding Boxes",
            use_container_width=True,
        )

        # Show crops and AI results
        st.subheader("Detection Results")
        for i, det in enumerate(detections):
            label = "AI" if det["ai_like"] else "Human"
            st.markdown(
                f"**Crop {i}**  \n"
                f"- Class: `{det['class_id']}`  \n"
                f"- Confidence: `{det['score']:.2f}`  \n"
                f"- Label: **{label}**  \n"
                f"- AI Score: `{det['ai_score']:.2f}`"
            )
            st.image(cropped_images[i], caption=f"Crop {i}", use_container_width=False)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
