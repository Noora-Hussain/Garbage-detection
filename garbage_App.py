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

# --- تنسيقات CSS احترافية وعصرية جداً ---
st.markdown("""
    <style>
    /* خلفية عامة ونظيفة */
    .stApp {
        background: linear-gradient(135deg, #f4f9f4 0%, #eef5f0 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* تنسيق العناوين الرئيسية */
    h1, h2, h3 {
        color: #1b4332 !important;
        font-weight: 700 !important;
    }
    
    h1 {
        border-bottom: 2px solid #d8f3dc;
        padding-bottom: 10px;
    }

    /* تصميم الـ Sidebar بلون متناسق */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d8f3dc;
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }

    /* تصميم بطاقات الـ Metrics (العدادات) بشكل فخم */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 6px 15px rgba(27, 67, 50, 0.05);
        border: 1px solid #d8f3dc;
        transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
    }

    /* الأزرار العصرية وأزرار التحميل */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #2d6a4f 0%, #1b4332 100%);
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(45, 106, 79, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #40916c 0%, #2d6a4f 100%);
        box-shadow: 0 6px 15px rgba(45, 106, 79, 0.3);
        color: white;
    }

    /* إطارات الصور والعرض */
    div.stImage > img {
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        border: 2px solid #ffffff;
    }

    /* تنسيق الـ Expander */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #d8f3dc;
        font-weight: 600;
        color: #1b4332;
    }
    
    /* صناديق التنبيهات والوصف */
    .stInfo {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        border-radius: 12px;
        color: #1b4332;
    }
    </style>
""", unsafe_allow_html=True)
# ---------------------------------------------
    
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


st.title("♻️ Smart Waste Routing & Monitoring System")
st.write(
    "Upload a waste photo or use your camera. The fine-tuned YOLO model will "
    "locate and identify the items it recognizes."
)


with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Minimum confidence", min_value=0.10, max_value=0.90)
    
    source = st.radio("Choose image source", ["Upload Image", "Use Camera"])

    st.markdown("---")
    st.header("🗑️ Garbage Classes")
    for Garbage in Garbage_Classes:
        st.write(f"• **{Garbage}**")

if source == "Upload Image":
    image_file = st.file_uploader(
        "Upload an unseen waste image",  
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a photo of the waste")  

if image_file is None:
    st.info("💡 Upload an image or take a camera photo to begin detection.")
elif not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place your trained YOLO weights named "
        "best.pt in the same folder as app.py."
    )
else:
    image = Image.open(image_file).convert("RGB")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    try:
        with st.spinner("✨ Detecting waste with AI..."): 
            model = load_model()
            result = model.predict(image, conf=confidence)[0]

        plotted_image = result.plot()[:, :, ::-1]
        annotated_image = Image.fromarray(plotted_image)
        detections = get_detections(result)

        with right_column:
            st.subheader("🎯 Detection Result")
            st.image(annotated_image, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Detected Garbage Statistics") 

        if detections:
            detection_data = pd.DataFrame(detections)
            garbage_counts = detection_data["Garbage"].value_counts() 

            count_columns = st.columns(min(len(garbage_counts), 4))
            for index, (item, count) in enumerate(garbage_counts.items()):
                count_columns[index % len(count_columns)].metric(item, int(count))

            st.write("") 
            st.dataframe(detection_data, use_container_width=True, hide_index=True)
            
            st.markdown("### 📝 Recycling Guidelines")
            detected_names = list(dict.fromkeys(detection_data["Garbage"].tolist()))
            for item in detected_names:
                description = GARBAGE_DESCRIPTIONS.get(item.title()) 
                if description:
                    st.info(description)
        else:
            st.warning(
                "⚠️ No garbage was detected. Try lowering the confidence " 
                "or using a clearer, closer image."
            )

        st.write("")
        st.download_button(
            "📥 Download Annotated Image",
            data=image_to_bytes(annotated_image),
            file_name="garbage_detection_result.jpg",  
            mime="image/jpeg"
        )

    except Exception as error:
        st.error(f"❌ The image could not be processed: {error}")

st.write("")
with st.expander("ℹ️ About this project"):
    st.write(
        "This prototype was created for a smart waste routing and monitoring project. "
        "It utilizes a fine-tuned YOLO object detection model to accurately classify waste items "
        "and provide eco-friendly disposal guidelines."
    )
