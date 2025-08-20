import streamlit as st
from PIL import Image
import boundingboxes
import os
import zipfile

st.set_page_config(page_title="Single Image AI Detection", layout="centered")
st.title("Single Image AI Detection")
st.markdown("Upload an image to detect whether it's AI-generated or human-created.")

# ----------------------
# Public URL Model
# ----------------------
model_url = "https://singleimageitemdetector.s3.us-east-2.amazonaws.com/yolov5m-fp16.tflite"
model_path = "yolov5m-fp16.tflite"

# Download model if not present
boundingboxes.download_model_url(model_url, model_path)

# ----------------------
# File Upload
# ----------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
debug_mode = st.checkbox("Enable debug mode", value=False)

if uploaded_file is not None:
    try:
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run detection
        detections, output_img_path, ai_results, cropped_images = boundingboxes.run_detection(
            temp_path, model_path, debug=debug_mode
        )

        # Show annotated image
        st.image(Image.open(output_img_path), caption="Processed Image with Bounding Boxes", use_column_width=True)

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
            st.image(cropped_images[i], caption=f"Crop {i}", use_column_width=False)

        # Create ZIP download for cropped images
        zip_filename = "cropped_detections.zip"
        with zipfile.ZipFile(zip_filename, 'w') as zip_buffer:
            for i, det in enumerate(detections):
                crop_img = cropped_images[i]
                buf = io.BytesIO()
                crop_img.save(buf, format="JPEG")
                zip_buffer.writestr(det["zip_name"], buf.getvalue())

        with open(zip_filename, "rb") as f:
            st.download_button(
                label="Download All Cropped Detections",
                data=f.read(),
                file_name=zip_filename,
                mime="application/zip"
            )

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
