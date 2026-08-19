import io
import os

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Smart Waste Routing",
    page_icon="♻️",
    layout="wide"
)
st.markdown("""
    <style>
    /* خلفية التطبيق بيضاء ونظيفة */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    
    /* تنسيق العناوين الرئيسية بلون أخضر هادئ وجميل */
    h1 {
        color: #0d9488;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #0f766e;
        font-family: 'Inter', sans-serif;
    }
    
    /* الشريط الجانبي بلون فاتح ومرتب */
    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    
    /* المربعات الإحصائية (Metrics) بخلفية بيضاء وظل خفيف */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0d9488 !important;
    }
    
    /* الأزرار بلون أخضر حيوي وجذاب */
    .stButton>button {
        background-color: #0d9488;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(13, 148, 136, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #0f766e;
        color: white;
    }
    
    /* زر التحميل بلون مميز */
    [data-testid="stDownloadButton"]>button {
        background-color: #0284c7;
    }
    [data-testid="stDownloadButton"]>button:hover {
        background-color: #0369a1;
    }

    /* صناديق المعلومات والإرشادات بخلفية فاتحة وناعمة */
    div.stInfo {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        border-radius: 8px;
        color: #166534;
    }
    
    div.stSuccess {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        border-radius: 8px;
        color: #166534;
    }
    </style>
""", unsafe_allow_html=True)


Garbage_Classes = ['Glass', 'Metal', 'Paper', 'Plastic', 'Waste']

GARBAGE_DESCRIPTIONS = {
    "Glass": "Glass: Please place it in the glass container and ensure it is clean and unbroken.",
    "Metal": "Metal: Aluminum cans and packaging are recyclable, please empty them.",
    "Paper": "Paper: Clean and dry paper, please place it in the designated paper bin.",
    "Plastic": "Plastic: Plastic bottles and containers, please rinse them before recycling.",
    "Waste": "General Waste: These are non-recyclable wastes, please dispose of them in the general waste bin."
}

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

        detections.append({"Garbage": result.names[class_id],"Confidence": f"{confidence * 100:.1f}%","Box": f"({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f})"})

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
    confidence = st.slider("Minimum confidence",min_value=0.10, max_value=0.90)
    
    source = st.radio("Choose image source", ["Upload Image", "Use Camera"])

    st.header("Garbage Classes")
    for Garbage in Garbage_Classes:
        st.write(f"• {Garbage}")

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
