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

"""
Smart Waste — custom look & feel for the Streamlit app.

This file only changes DESIGN (colors, spacing, cards, fonts).
It does not touch any detection logic.

How to use it in garbage_App.py:

    import streamlit as st
    from app_style import inject_custom_css

    st.set_page_config(
        page_title="Smart Waste Routing",
        page_icon="♻️",
        layout="wide"
    )
    inject_custom_css()   # <-- add this one line right after set_page_config

That's it — everything below is CSS injected once at the top of the app.
"""

import streamlit as st

# Same palette used in the Lightning Talk 2 slides
PRIMARY = "#1B4332"    # deep forest green
SECONDARY = "#52796F"  # slate teal-green
ACCENT = "#F2B134"     # amber
LIGHT = "#F4F6F3"      # near-white card background
INK = "#1B1B1B"
MUTED = "#5B6B63"

# Per-class chip colors (matches the slide legend)
CLASS_COLORS = {
    "Glass": "#52796F",
    "Metal": "#8D99AE",
    "Paper": "#F2B134",
    "Plastic": "#E76F51",
    "Waste": "#6B4226",
}


def inject_custom_css():
    st.markdown(
        f"""
        <style>
        /* ---------- page background & base font ---------- */
        .stApp {{
            background-color: {LIGHT};
        }}
        html, body, [class*="css"] {{
            font-family: 'Calibri', 'Segoe UI', sans-serif;
            color: {INK};
        }}

        /* ---------- main title ---------- */
        h1 {{
            color: {PRIMARY} !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }}
        h2, h3 {{
            color: {PRIMARY} !important;
        }}

        /* ---------- sidebar ---------- */
        section[data-testid="stSidebar"] {{
            background-color: {PRIMARY};
        }}
        section[data-testid="stSidebar"] * {{
            color: #F4F6F3 !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {ACCENT} !important;
        }}

        /* ---------- buttons ---------- */
        .stButton > button, .stDownloadButton > button {{
            background-color: {PRIMARY};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            font-weight: 600;
            transition: 0.2s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {SECONDARY};
            color: white;
        }}

        /* ---------- slider ---------- */
        .stSlider [data-baseweb="slider"] > div > div {{
            background: {ACCENT} !important;
        }}

        /* ---------- radio buttons ---------- */
        div[role="radiogroup"] label {{
            font-weight: 500;
        }}

        /* ---------- file uploader / camera box ---------- */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: white;
            border: 2px dashed {SECONDARY};
            border-radius: 12px;
        }}

        /* ---------- metric cards (garbage counts) ---------- */
        [data-testid="stMetric"] {{
            background-color: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        [data-testid="stMetricValue"] {{
            color: {PRIMARY} !important;
            font-weight: 800 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {MUTED} !important;
        }}

        /* ---------- dataframe / table ---------- */
        [data-testid="stDataFrame"] {{
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ---------- info / warning / error boxes ---------- */
        div[data-testid="stAlert"] {{
            border-radius: 10px;
        }}

        /* ---------- expander ("About this project") ---------- */
        details {{
            background-color: white;
            border-radius: 10px;
            border: 1px solid #E3E8E5;
        }}

        /* ---------- subtle divider under the title ---------- */
        .block-container > div:first-child {{
            padding-top: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def class_badge_html(label: str) -> str:
    """
    Returns a small colored HTML pill for a waste class,
    e.g. st.markdown(class_badge_html("Plastic"), unsafe_allow_html=True)
    """
    color = CLASS_COLORS.get(label, SECONDARY)
    return (
        f'<span style="background-color:{color}; color:white; '
        f'padding:3px 10px; border-radius:999px; font-size:13px; '
        f'font-weight:600;">{label}</span>'
    )
    
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
