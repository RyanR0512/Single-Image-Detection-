import streamlit as st
from PIL import Image
import boundingboxes

st.set_page_config(page_title="Single Image AI Detection", layout="centered")

st.title("Single Image AI Detection")
st.markdown("Upload an image to detect whether it's AI-generated or human-created.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Optional: choose a class to focus on
focus_class = st.selectbox(
    "Focus on a specific object (optional)",
    options=["All"] + boundingboxes.COCO_CLASSES
)

if uploaded_file is not None:
    try:
        # Save uploaded image temporarily
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run detection
        detections, output_img_path, ai_results = boundingboxes.run_detection(temp_path)

        # Filter detections if a specific class is chosen
        if focus_class != "All":
            detections = [det for det in detections if det["class_name"] == focus_class]

        # Display output image
        output_img = Image.open(output_img_path)
        st.image(output_img, caption="Processed Image with Bounding Boxes", use_container_width=True)

        # Display detection results
        st.subheader("Detection Results")
        if detections:
            for i, det in enumerate(detections):
                label = "AI" if det["ai_like"] else "Human"
                st.markdown(
                    f"""
                    **Crop {i}**  
                    - Class: `{det['class_name']}` (ID: {det['class_id']})  
                    - Confidence Score: `{det['score']:.2f}`  
                    - Label: **{label}**  
                    - AI Score: `{det['ai_score']:.2f}`
                    """
                )
        else:
            st.info(f"No objects of class '{focus_class}' detected.")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
