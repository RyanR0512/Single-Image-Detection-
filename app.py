import streamlit as st
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import boundingboxes  # your module that handles object detection + cropping

# Streamlit Page Setup
st.set_page_config(page_title="Single Image AI Detection", layout="centered")

st.title("🧠 Single Image AI Detection")
st.markdown("Upload an image to detect whether it's AI-generated or human-created. ")

# Model Loader (cached for performance)
@st.cache_resource
def load_ai_detector():
    """Load lizhil/AIGC-Detector only once."""
    processor = AutoImageProcessor.from_pretrained("lizhil/AIGC-Detector")
    model = AutoModelForImageClassification.from_pretrained("lizhil/AIGC-Detector")
    model.eval()
    return processor, model

processor, ai_model = load_ai_detector()

# Helper function: AI vs. Human classification
def detect_ai_in_crop(crop_bytes):
    """Run the AIGC detector model on a cropped object image."""
    try:
        image = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = ai_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        # lizhil/AIGC-Detector typically has two classes: Real (0), AI (1)
        ai_score = probs[1].item()
        is_ai = ai_score > 0.5
        return is_ai, ai_score

    except Exception as e:
        st.error(f"AI detection error: {str(e)}")
        return False, 0.0

# UI Controls
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

focus_class = st.selectbox(
    "🎯 Focus on a specific object (optional)",
    options=["All"] + boundingboxes.COCO_CLASSES
)

# Main App Logic
if uploaded_file is not None:
    try:
        # Save uploaded image to a temp file
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run your object detection + cropping pipeline
        detections, output_img_path, ai_results, crops = boundingboxes.run_detection(temp_path)

        # Run AI detection on each crop
        for det, crop_bytes in zip(detections, crops):
            det["ai_like"], det["ai_score"] = detect_ai_in_crop(crop_bytes)

        # Filter detections based on user selection
        if focus_class != "All":
            filtered = [(det, crop) for det, crop in zip(detections, crops) if det["class_name"] == focus_class]
        else:
            filtered = list(zip(detections, crops))

        # Display annotated image
        output_img = Image.open(output_img_path)
        st.image(output_img, caption="Processed Image with Bounding Boxes", use_container_width=True)

        # Display detection results
        st.subheader("📊 Detection Results")
        if filtered:
            for i, (det, crop_bytes) in enumerate(filtered):
                crop_img = Image.open(io.BytesIO(crop_bytes))
                label = "🧠 AI" if det["ai_like"] else "🧍 Human"
                color = "red" if det["ai_like"] else "green"

                st.image(crop_img, caption=f"Crop {i + 1}: {det['class_name']}", width=200)
                st.markdown(
                    f"""
                    **Class:** `{det['class_name']}` (ID: {det['class_id']})  
                    **Detection Confidence:** `{det['score']:.2f}`  
                    **AI Detection Label:** <span style='color:{color};font-weight:bold'>{label}</span>  
                    **AI Confidence:** `{det['ai_score']:.2f}`
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info(f"No objects of class '{focus_class}' detected.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")

else:
    st.info("👆 Upload an image to get started.")
