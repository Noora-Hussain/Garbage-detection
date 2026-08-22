import io
import os
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="EcoVision | Smart Waste Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GARBAGE CLASSES & DISPOSAL INFORMATION
# =========================================================
Garbage_Classes = ["Glass", "Metal", "Paper", "Plastic", "Waste"]

GARBAGE_DESCRIPTIONS = {
    "Glass": "Place clean and unbroken glass in the designated glass recycling container.",
    "Metal": "Empty metal cans and packaging, then place them in the metal recycling bin.",
    "Paper": "Keep paper clean and dry, then place it in the designated paper recycling bin.",
    "Plastic": "Rinse plastic bottles and containers when possible, then place them in the plastic recycling bin.",
    "Waste": "General waste should be placed in the general waste bin."
}


# =========================================================
# MODEL PATH
# =========================================================
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# =========================================================
# DETECTION FUNCTIONS
# =========================================================
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


def get_street_status(number_of_objects):
    if number_of_objects == 0:
        return "CLEAN", "The street appears clean. No garbage was detected."
    elif number_of_objects <= 3:
        return "SOME GARBAGE", "Some garbage was detected on this street."
    else:
        return "NEEDS CLEANING", "Several garbage objects were detected. This area may need cleaning."


def show_detection_results(result, mode):
    plotted_image = result.plot()[:, :, ::-1]
    annotated_image = Image.fromarray(plotted_image)
    detections = get_detections(result)

    left, right = st.columns(2)

    with left:
        st.markdown("### 📷 Original Image")
        st.image(st.session_state.current_image, use_container_width=True)

    with right:
        st.markdown("### 🔎 AI Detection")
        st.image(annotated_image, use_container_width=True)

    if mode == "Street Detection":
        st.markdown("## 🛣️ Street Status")

        if detections:
            status, message = get_street_status(len(detections))

            if status == "NEEDS CLEANING":
                st.error(f"🚨 {status}")
            else:
                st.warning(f"⚠️ {status}")

            st.write(message)
        else:
            st.success("✅ CLEAN")
            st.write("No garbage was detected in this image.")

    st.markdown("## ♻️ Detected Garbage")

    if detections:
        detection_data = pd.DataFrame(detections)
        garbage_counts = detection_data["Garbage"].value_counts()

        count_columns = st.columns(min(len(garbage_counts), 4))

        for index, (item, count) in enumerate(garbage_counts.items()):
            count_columns[index % len(count_columns)].metric(
                item,
                int(count)
            )

        st.dataframe(
            detection_data,
            use_container_width=True,
            hide_index=True
        )

        detected_names = list(
            dict.fromkeys(detection_data["Garbage"].tolist())
        )

        st.markdown("### 💡 What should you do?")

        for item in detected_names:
            description = GARBAGE_DESCRIPTIONS.get(item.title())

            if description:
                st.info(f"**{item}:** {description}")

        st.download_button(
            "⬇️ Download Detection Result",
            data=image_to_bytes(annotated_image),
            file_name="garbage_detection_result.jpg",
            mime="image/jpeg"
        )

    else:
        st.warning(
            "No garbage was detected. Try a clearer image or lower the confidence threshold."
        )


# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7F5;
    }

    [data-testid="stSidebar"] {
        background-color: #18352B;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .hero {
        background: linear-gradient(135deg, #18352B, #2F6B4F);
        padding: 35px;
        border-radius: 22px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.9;
    }

    .feature-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #DDE7E1;
        min-height: 150px;
        margin-bottom: 15px;
    }

    .feature-card h3 {
        color: #18352B;
    }

    .section-title {
        color: #18352B;
        font-size: 28px;
        font-weight: 700;
        margin-top: 15px;
    }

    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #2F6B4F;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #718078;
        margin-top: 45px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ♻️ EcoVision")
    st.caption("AI-Powered Smart Waste Detection")

    st.markdown("---")

    page = st.radio(
        "Choose a feature",
        [
            "🏠 Home",
            "🛣️ Street Detection",
            "♻️ Waste Assistant"
        ]
    )

    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.markdown("---")

    st.caption("Supported waste classes")
    for garbage in Garbage_Classes:
        st.write(f"• {garbage}")


# =========================================================
# HOME PAGE
# =========================================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>♻️ EcoVision</h1>
        <p>
            Smart AI-powered waste detection for cleaner streets
            and better waste disposal decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">What can EcoVision do?</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🛣️ Street Detection</h3>
            <p>
                Upload a street image and let the AI identify garbage
                objects and indicate whether the area needs cleaning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>♻️ Waste Assistant</h3>
            <p>
                Take or upload a photo of an individual waste item.
                The AI identifies its type and provides disposal guidance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🤖 Supported Waste Types")

    cols = st.columns(len(Garbage_Classes))

    for index, garbage in enumerate(Garbage_Classes):
        cols[index].metric(garbage, "AI Class")

    st.markdown("""
    <div class="footer">
        EcoVision • Smart Waste Detection Prototype
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# STREET DETECTION
# =========================================================
elif page == "🛣️ Street Detection":

    st.markdown("## 🛣️ Street Garbage Detection")
    st.write(
        "Upload a street image and the AI will detect garbage objects, "
        "show their locations, and estimate the cleanliness status."
    )

    image_file = st.file_uploader(
        "Upload a street image",
        type=["jpg", "jpeg", "png"],
        key="street_upload"
    )

    if image_file is not None:
        if not os.path.exists(MODEL_PATH):
            st.error(
                "Model file not found. Place your trained YOLO weights "
                "named best.pt in the same folder as this app."
            )
        else:
            try:
                image = Image.open(image_file).convert("RGB")
                st.session_state.current_image = image

                with st.spinner("🤖 Analyzing the street..."):
                    model = load_model()
                    result = model.predict(
                        image,
                        conf=confidence,
                        verbose=False
                    )[0]

                show_detection_results(
                    result,
                    "Street Detection"
                )

            except Exception as error:
                st.error(
                    f"The image could not be processed: {error}"
                )


# =========================================================
# WASTE ASSISTANT
# =========================================================
elif page == "♻️ Waste Assistant":

    st.markdown("## ♻️ Personal Waste Assistant")
    st.write(
        "Take a photo or upload an image of a waste item. "
        "EcoVision will identify the item and tell you how to dispose of it."
    )

    source = st.radio(
        "Choose image source",
        ["Upload Image", "Use Camera"],
        horizontal=True
    )

    if source == "Upload Image":
        image_file = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png"],
            key="waste_upload"
        )
    else:
        image_file = st.camera_input(
            "Take a photo of the waste"
        )

    if image_file is not None:
        if not os.path.exists(MODEL_PATH):
            st.error(
                "Model file not found. Place your trained YOLO weights "
                "named best.pt in the same folder as this app."
            )
        else:
            try:
                image = Image.open(image_file).convert("RGB")
                st.session_state.current_image = image

                with st.spinner("🔍 Identifying the waste..."):
                    model = load_model()
                    result = model.predict(
                        image,
                        conf=confidence,
                        verbose=False
                    )[0]

                show_detection_results(
                    result,
                    "Waste Assistant"
                )

            except Exception as error:
                st.error(
                    f"The image could not be processed: {error}"
                )
