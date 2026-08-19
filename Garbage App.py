import io
import os

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="garbage_detection",
    page_icon="GD",
    layout="wide"
)

Garbage_Classes = ['Paper', 'Plastic']

DESCRIPTIONS = {
    "Plastic": "Recyclable plastic items like bottles and containers. Clean before recycling.",
    "Paper": "Dry paper waste. Can be recycled in paper bins."}

APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def get_detections(result):
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "Garbage": result.names[class_id],
            "Confidence": f"{confidence * 100:.1f}%",
            "Box": f"({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})"
        })

    return detections

def image_to_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


st.title("♻️ Smart Waste Routing & Monitoring System ")
st.write(
    "Upload a waste photo or use your camera. The fine-tuned YOLO model will "
    "locate and identify the items it recognizes."
)


with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05
    )
    source = st.radio("Choose image source", ["Upload Image", "Use Camera"])

    st.header("Garbage Classes")
    for food in Garbage_Classes:
        st.write(f"• {item}")

if source == "Upload Image":
    image_file = st.file_uploader(
        "Upload an unseen waste image",  
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a photo of the waste")  

if image_file is None:
    st.info("Upload an image or take a camera photo to begin detection.")
elif not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place your trained YOLO weights named "
        "best.pt in the same folder as app.py."
    )
else:
    image = Image.open(image_file).convert("RGB")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    try:
        with st.spinner("Detecting waste..."): 
            model = load_model()
            result = model.predict(image, conf=confidence)[0]

        plotted_image = result.plot()[:, :, ::-1]
        annotated_image = Image.fromarray(plotted_image)
        detections = get_detections(result)

        with right_column:
            st.subheader("Detection Result")
            st.image(annotated_image, use_container_width=True)

        st.subheader("Detected Garbage") 


        if detections:
            detection_data = pd.DataFrame(detections)
            garbage_counts = detection_data["Garbage"].value_counts() 

            count_columns = st.columns(min(len(garbage_counts), 4))
            for index, (item, count) in enumerate(garbage_counts.items()):
                count_columns[index % len(count_columns)].metric(item, int(count))

            st.dataframe(detection_data, use_container_width=True, hide_index=True)

            
            detected_names = list(dict.fromkeys(detection_data["Garbage"].tolist()))
            for item in detected_names:
                description = GARBAGE_DESCRIPTIONS.get(item.title()) 
                if description:
                    st.write(f"**{item}:** {description}")
        else:
            st.warning(
                "No garbage was detected. Try lowering the confidence " 
                "or using a clearer, closer image."
            )

        st.download_button(
            "Download Annotated Image",
            data=image_to_bytes(annotated_image),
            file_name="garbage_detection_result.jpg",  
            mime="image/jpeg"
        )

    except Exception as error:
        st.error(f"The image could not be processed: {error}")

with st.expander("About this project"):
    st.write(
        "This prototype was created for a garbage detection project"
        "It uses a fine-tuned YOLO model to detect waste items and place a bounding box around each detected object"
    )