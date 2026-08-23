import io
import os
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

# =========================================================
# 1. PAGE CONFIGURATION & STYLING (التصميم والواجهة)
# =========================================================
st.set_page_config(
    page_title="EcoVision | Smart Waste Detection & Analytics",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    .ai-badge {
        background-color: #E2F0EC;
        color: #18352B;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. CONSTANTS & AI CONFIGURATION (الثوابت وإعدادات الذكاء الاصطناعي)
# =========================================================
Garbage_Classes = ['Battery', 'Glass', 'Medical', 'Metal', 'Organic', 'Paper', 'Plastic', 'SmartPhone']

GARBAGE_DESCRIPTIONS = {
    "Battery": "Do not throw batteries in regular trash. Take them to a specialized battery recycling drop-off point.",
    "Glass": "Place clean and unbroken glass in the designated glass recycling container.",
    "Medical": "Dispose of medical waste and pharmaceuticals safely in designated biohazard or pharmacy-takeback bins.",
    "Metal": "Empty metal cans and packaging, then place them in the metal recycling bin.",
    "Organic": "Compost food scraps and organic waste in an organic bin or composting system.",
    "Paper": "Keep paper clean and dry, then place it in the designated paper recycling bin.",
    "Plastic": "Rinse plastic bottles and containers when possible, then place them in the plastic recycling bin.",
    "SmartPhone": "Old smartphones and electronics should be taken to an e-waste recycling facility or a certified trade-in center."
}

APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")


# =========================================================
# 3. HELPER FUNCTIONS & LOGIC (الدوال والمنطق البرمجي)
# =========================================================
@st.cache_resource
def load_ai_model():
    """تحميل نموذج الذكاء الاصطناعي YOLO"""
    return YOLO(MODEL_PATH)

def extract_detections(result):
    """استخراج نتائج الكشف من نموذج الذكاء الاصطناعي"""
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

def convert_image_to_bytes(image):
    """تحويل الصورة إلى بايتات للتحميل"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

def evaluate_street_cleanliness(number_of_objects):
    """تقييم حالة الشارع بناءً على النفايات المكتشفة بالذكاء الاصطناعي"""
    if number_of_objects == 0:
        return "CLEAN", "The street appears clean. No garbage was detected by AI."
    elif number_of_objects <= 3:
        return "SOME GARBAGE", "Some garbage was detected by AI on this street."
    else:
        return "NEEDS CLEANING", "Several garbage objects were detected by AI. This area requires immediate cleaning."

def render_area_selector(key_suffix):
    """دوال اختيار المنطقة الجغرافية للبلاغات"""
    main_areas = ["Manama", "Muharraq", "Riffa", "Isa Town", "Hamad Town", "Other"]
    selected_main_area = st.selectbox("Select Area:", main_areas, key=f"area_select_{key_suffix}")
    
    if selected_main_area == "Other":
        custom_area = st.text_input("Please enter the area name:", key=f"custom_area_{key_suffix}")
        return custom_area if custom_area else "Unspecified"
    return selected_main_area

def render_detection_results_view(result, mode):
    """عرض مخرجات وتحليلات نموذج الذكاء الاصطناعي للمستخدم"""
    plotted_image = result.plot()[:, :, ::-1]
    annotated_image = Image.fromarray(plotted_image)
    detections = extract_detections(result)

    left, right = st.columns(2)

    with left:
        st.markdown("### 📷 Original Image")
        st.image(st.session_state.current_image, use_container_width=True)

    with right:
        st.markdown("### 🤖 AI Vision Detection")
        st.image(annotated_image, use_container_width=True)

    if mode == "Street Detection":
        st.markdown("## 🛣️ AI Street Status Analysis")
        if detections:
            status, message = evaluate_street_cleanliness(len(detections))
            if status == "NEEDS CLEANING":
                st.error(f"🚨 {status}")
            else:
                st.warning(f"⚠️ {status}")
            st.write(message)
        else:
            st.success("✅ CLEAN")
            st.write("No garbage was detected by the AI model in this image.")

    st.markdown("## ♻️ AI Detected Garbage Categories")

    if detections:
        detection_data = pd.DataFrame(detections)
        garbage_counts = detection_data["Garbage"].value_counts()

        count_columns = st.columns(min(len(garbage_counts), 4))
        for index, (item, count) in enumerate(garbage_counts.items()):
            count_columns[index % len(count_columns)].metric(item, int(count))

        st.dataframe(detection_data, use_container_width=True, hide_index=True)

        detected_names = list(dict.fromkeys(detection_data["Garbage"].tolist()))
        st.markdown("### 💡 AI Smart Disposal Recommendations")

        for item in detected_names:
            description = GARBAGE_DESCRIPTIONS.get(item.title())
            if description:
                st.info(f"**{item}:** {description}")

        st.download_button(
            "⬇️ Download AI Vision Result",
            data=convert_image_to_bytes(annotated_image),
            file_name="ai_garbage_detection_result.jpg",
            mime="image/jpeg"
        )
    else:
        st.warning("No garbage was detected. Try adjusting the AI confidence slider or uploading a clearer image.")


# =========================================================
# 4. INITIALIZE SESSION STATE (تهيئة حالة البيانات للبلاغات)
# =========================================================
if "reports_list" not in st.session_state:
    st.session_state.reports_list = [
        {"ID": "Report #1021", "Area": "Manama", "Objects": 4, "Priority": "🟠 Medium Priority", "Date": "20 Aug 2026", "Status": "Resolved"},
        {"ID": "Report #1022", "Area": "Muharraq", "Objects": 6, "Priority": "🔴 High Priority", "Date": "21 Aug 2026", "Status": "Pending Review"},
        {"ID": "Report #1023", "Area": "Riffa", "Objects": 2, "Priority": "🟢 Low Priority", "Date": "22 Aug 2026", "Status": "In Progress"}
    ]


# =========================================================
# 5. SIDEBAR & NAVIGATION (القائمة الجانبية والتنقل)
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
            "♻️ Waste Assistant",
            "🚨 Report a Dirty Area",
            "📊 Analytics Dashboard"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚙️ AI Settings")

    confidence = st.slider(
        "AI Minimum Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.markdown("---")
    st.caption("Supported AI Waste Classes")
    for garbage in Garbage_Classes:
        st.write(f"• {garbage}")


# =========================================================
# 6. APPLICATION PAGES (صفحات التطبيق)
# =========================================================

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <span class="ai-badge">Powered by Deep Learning & Computer Vision</span>
        <h1>♻️ EcoVision</h1>
        <p>
            Smart AI-powered waste detection system for cleaner streets,
            automated municipal reporting, and optimal waste management.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What can EcoVision AI do?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🛣️ AI Street Detection</h3>
            <p>Upload a street image to let our computer vision model detect garbage locations, count objects, and evaluate cleanliness status instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>♻️ AI Waste Assistant</h3>
            <p>Scan individual waste items using your camera or uploaded images to get instant AI classification and expert recycling instructions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🤖 Supported AI Waste Classes")
    cols = st.columns(len(Garbage_Classes))
    for index, garbage in enumerate(Garbage_Classes):
        cols[index].markdown(f"**{garbage}**")

    st.markdown("""
    <div class="footer">
        EcoVision • Smart Waste Detection Prototype
    </div>
    """, unsafe_allow_html=True)


# --- STREET DETECTION ---
elif page == "🛣️ Street Detection":
    st.markdown('<span class="ai-badge">YOLO Computer Vision Model</span>', unsafe_allow_html=True)
    st.markdown("## 🛣️ Street Garbage Detection")
    st.write("Upload a street photo and let the AI analyze waste presence and street cleanliness status.")

    image_file = st.file_uploader("Upload a street image", type=["jpg", "jpeg", "png"], key="street_upload")

    if image_file is not None:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found. Place your trained YOLO weights named best.pt in the app directory.")
        else:
            try:
                image = Image.open(image_file).convert("RGB")
                st.session_state.current_image = image

                with st.spinner("🤖 AI model is analyzing the street image..."):
                    model = load_ai_model()
                    result = model.predict(image, conf=confidence, verbose=False)[0]

                render_detection_results_view(result, "Street Detection")

            except Exception as error:
                st.error(f"The image could not be processed by AI: {error}")


# --- WASTE ASSISTANT ---
elif page == "♻️ Waste Assistant":
    st.markdown('<span class="ai-badge">AI Image Classification</span>', unsafe_allow_html=True)
    st.markdown("## ♻️ Personal Waste Assistant")
    st.write("Take a photo or upload an image of a waste item. The AI will classify it and provide recycling steps.")

    source = st.radio("Choose image source", ["Upload Image", "Use Camera"], horizontal=True)

    if source == "Upload Image":
        image_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"], key="waste_upload")
    else:
        image_file = st.camera_input("Take a photo of the waste item")

    if image_file is not None:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found. Place your trained YOLO weights named best.pt in the app directory.")
        else:
            try:
                image = Image.open(image_file).convert("RGB")
                st.session_state.current_image = image

                with st.spinner("🔍 AI is identifying the waste item..."):
                    model = load_ai_model()
                    result = model.predict(image, conf=confidence, verbose=False)[0]

                render_detection_results_view(result, "Waste Assistant")

            except Exception as error:
                st.error(f"The image could not be processed by AI: {error}")


# --- REPORT A DIRTY AREA ---
elif page == "🚨 Report a Dirty Area":
    st.markdown('<span class="ai-badge">Automated Municipal Ticketing</span>', unsafe_allow_html=True)
    st.markdown("## 🚨 Report a Dirty Area")
    st.write("Help keep the community clean! Select the target area, capture or upload evidence, and let AI evaluate priority.")

    selected_area = render_area_selector("report")

    report_source = st.radio("Choose image source for report", ["Upload Image", "Use Camera"], horizontal=True, key="report_source")

    if report_source == "Upload Image":
        report_image_file = st.file_uploader("Upload a photo of the dirty area", type=["jpg", "jpeg", "png"], key="report_upload")
    else:
        report_image_file = st.camera_input("Take a photo of the dirty area", key="report_camera")

    if report_image_file is not None:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found. Place your trained YOLO weights named best.pt in the app directory.")
        else:
            try:
                image = Image.open(report_image_file).convert("RGB")
                
                with st.spinner("🤖 AI vision model is evaluating waste density..."):
                    model = load_ai_model()
                    result = model.predict(image, conf=confidence, verbose=False)[0]

                detections = extract_detections(result)
                num_objects = len(detections)

                if num_objects > 5:
                    priority = "🔴 High Priority"
                elif num_objects > 2:
                    priority = "🟠 Medium Priority"
                else:
                    priority = "🟢 Low Priority"

                current_date = datetime.now().strftime("%d %b %Y")
                next_id_num = 1021 + len(st.session_state.reports_list)
                report_id = f"Report #{next_id_num}"

                st.markdown("---")
                st.markdown("### 📋 AI Report Preview")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Submitted Photo Evidence", use_container_width=True)
                with col2:
                    st.markdown(f"### **{report_id}**")
                    st.markdown(f"📍 **Location:** {selected_area}")
                    st.markdown(f"🗑️ **Garbage Objects Detected:** {num_objects}")
                    st.markdown(f"⚡ **AI Priority Score:** {priority}")
                    st.markdown(f"📅 **Date:** {current_date}")
                    st.markdown(f"🔄 **Status:** 🟡 Pending Review")

                st.markdown("---")
                if st.button("🚀 Submit Official Report", type="primary"):
                    new_report = {
                        "ID": report_id,
                        "Area": selected_area,
                        "Objects": num_objects,
                        "Priority": priority,
                        "Date": current_date,
                        "Status": "Pending Review"
                    }
                    st.session_state.reports_list.append(new_report)
                    st.success(f"✅ Your report (**{report_id}**) for **{selected_area}** has been successfully submitted and integrated into the live analytics system!")

            except Exception as error:
                st.error(f"Could not process the report image: {error}")


# --- ANALYTICS DASHBOARD ---
elif page == "📊 Analytics Dashboard":
    st.markdown('<span class="ai-badge">Live Municipal Data</span>', unsafe_allow_html=True)
    st.markdown("## 📊 Waste Management Analytics Dashboard")
    st.write("Real-time telemetry and overview of submitted AI municipal reports and regional distribution.")

    df_reports = pd.DataFrame(st.session_state.reports_list)

    total_reports = len(df_reports)
    resolved_count = len(df_reports[df_reports["Status"] == "Resolved"])
    pending_count = len(df_reports[df_reports["Status"].isin(["Pending Review", "In Progress"])])
    resolved_percentage = (resolved_count / total_reports * 100) if total_reports > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reports", str(total_reports), "Live Data")
    col2.metric("Resolved Areas", str(resolved_count), f"{resolved_percentage:.1f}%")
    col3.metric("Pending Cleanup", str(pending_count), "Active")
    col4.metric("Active Regions", str(df_reports["Area"].nunique()), "Tracked")

    st.markdown("---")

    st.markdown("### 📈 Live Reports by Area")
    if not df_reports.empty:
        area_counts = df_reports["Area"].value_counts().reset_index()
        area_counts.columns = ["Area", "Reports Count"]
        st.bar_chart(area_counts.set_index("Area"))
    else:
        st.info("No reports submitted yet.")

    st.markdown("### 📋 Real-Time Municipal Reports Log")
    st.dataframe(df_reports, use_container_width=True, hide_index=True)
