import streamlit as st
from PIL import Image
import io
import boundingboxes

st.set_page_config(page_title="Single Image AI Detection", layout="centered")

st.title("Single Image AI Detection")
st.markdown("Upload an image to detect whether objects are AI-generated or human-created.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

focus_class = st.selectbox(
    "Focus on a specific object (optional)",
    options=["All"] + boundingboxes.alphabetized_list
)

if uploaded_file is not None:
    try:
        # Save uploaded file to a temporary path
        temp_path = "temp_uploaded_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run detection + per-crop AI classification
        detections, output_img_path, _, crops = boundingboxes.run_detection(temp_path)

        # Filter detections and crops if a specific class is selected
        if focus_class != "All":
            filtered = [(det, crop) for det, crop in zip(detections, crops) if det["class_name"] == focus_class]
        else:
            filtered = list(zip(detections, crops))

        # Display annotated image
        output_img = Image.open(output_img_path)
        st.image(output_img, caption="Processed Image with Bounding Boxes", use_container_width=True)

        # Display results for each crop
        st.subheader("Detection Results")
        if filtered:
            for i, (det, crop_bytes) in enumerate(filtered):
                label = "AI" if det["ai_like"] else "Human"
                crop_img = Image.open(io.BytesIO(crop_bytes))
                st.image(crop_img, caption=f"Crop {i}: {det['class_name']}", width=200)
                st.markdown(
                    f"""
                    **Class:** `{det['class_name']}` (ID: {det['class_id']})  
                    **Confidence:** `{det['score']:.2f}`  
                    **Label:** **{label}**  
                    **AI Score:** `{det['ai_score']:.2f}`
                    """
                )
        else:
            st.info(f"No objects of class '{focus_class}' detected.")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
