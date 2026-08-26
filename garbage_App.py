import io
import os
import tempfile
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(page_title="EcoVision | Smart Waste Detection", page_icon="♻️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #F4F7F5; }
    [data-testid="stSidebar"] { background-color: #18352B; }
    [data-testid="stSidebar"] * { color: white; }
    .hero {
        background: linear-gradient(135deg, #18352B, #2F6B4F);
        padding: 35px; border-radius: 22px; color: white; margin-bottom: 25px;
    }
    .hero h1 { font-size: 42px; margin-bottom: 8px; }
    .hero p { font-size: 18px; opacity: 0.9; }
    .feature-card {
        background: white; padding: 24px; border-radius: 18px;
        border: 1px solid #DDE7E1; min-height: 150px; margin-bottom: 15px;
    }
    .feature-card h3 { color: #18352B; }
    .section-title { color: #18352B; font-size: 28px; font-weight: 700; margin-top: 15px; margin-bottom: 15px; }
    div.stButton > button { border-radius: 12px; border: 1px solid #2F6B4F; font-weight: 600; }
    .footer { text-align: center; color: #718078; margin-top: 45px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

GARBAGE_DESCRIPTIONS = {
    "Glass": "Place in the designated glass recycling container.",
    "Metal": "Place in the metal recycling bin.",
    "Paper": "Place in the designated paper recycling bin.",
    "Plastic": "Place in the plastic recycling bin.",
    "General Waste": "Place in the general waste bin for non-recyclable items.",
}

APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")

@st.cache_resource
def load_my_model():
    return YOLO(MODEL_PATH)

def count_detected_objects(result):
    if result.boxes is None:
        return []
    
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({"Garbage": result.names[class_id], "Confidence": f"{conf * 100:.1f}%"})
    return detections

def choose_area_menu(unique_key):
    areas = ["Manama", "Muharraq", "Riffa", "Isa Town", "Hamad Town", "Other"]
    selected = st.selectbox("Select Area:", areas, key=unique_key)
    
    if selected == "Other":
        custom = st.text_input("Enter area name:", key=f"custom_{unique_key}")
        return custom if custom else "Unspecified"
    return selected

if "reports_list" not in st.session_state:
    st.session_state.reports_list = [
        {"ID": "Report #1", "Area": "Manama", "Objects": 4, "Priority": "🟠 Medium", "Date": "20 Aug 2026", "Status": "Resolved"},
        {"ID": "Report #2", "Area": "Muharraq", "Objects": 6, "Priority": "🔴 High", "Date": "21 Aug 2026", "Status": "Pending Review"}
    ]

with st.sidebar:
    st.markdown("## ♻️ EcoVision")
    st.caption("Smart Waste Detection")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🛣️ Street Detection", "♻️ Waste Assistant", "🚨 Report a Dirty Area", "📊 Analytics Dashboard"]
    )

    st.markdown("---")
    confidence = st.slider("AI Confidence", 0.10, 0.90, 0.25, 0.05)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>♻️ EcoVision</h1>
        <p>Smart AI-powered waste detection system for cleaner streets and communities.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <h3>🛣️ Street Detection</h3>
            <p>Analyze street images and live video footage to detect waste presence and evaluate cleanliness levels.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <h3>♻️ Waste Assistant</h3>
            <p>Classify individual waste items via photo or camera and receive immediate recycling recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <h3>🚨 Report a Dirty Area</h3>
            <p>Submit photos of polluted areas with automatic AI object count and priority level assignment.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Analytics Dashboard</h3>
            <p>Track regional clean-up statistics, total reports, resolution progress, and active area distributions.</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== STREET DETECTION ====================
elif page == "🛣️ Street Detection":
    st.markdown("## 🛣️ Street Garbage Detection")
    st.write("Analyze street footage to detect waste presence.")

    source_type = st.radio(
        "Source",
        ["📷 Image", "🎞️ Video Upload"],
        horizontal=True
    )

    if source_type == "📷 Image":
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if img_file is not None:
            image = Image.open(img_file).convert("RGB")

            with st.spinner("AI is analyzing..."):
                model = load_my_model()
                res = model.predict(image, conf=confidence, verbose=False)[0]

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(res.plot()[:, :, ::-1], caption="AI Detection", use_container_width=True)

            items = count_detected_objects(res)
            if items:
                found_types = sorted(set(i["Garbage"] for i in items))
                st.warning(f"⚠️ Garbage found! Type(s): {', '.join(found_types)}")
                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
            else:
                st.success("✅ Clean street! No garbage found.")

    elif source_type == "🎞️ Video Upload":
        video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
        frame_skip = st.slider("Analyze every Nth frame (higher = faster)", 1, 15, 5)

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
            tfile.write(video_file.read())
            tfile.close()

            
            st.markdown("---")
            if all_detections:
                st.warning(f"⚠️ Found {len(all_detections)} garbage detections across analyzed frames!")
                summary = pd.DataFrame(all_detections)["Garbage"].value_counts().reset_index()
                summary.columns = ["Garbage", "Count"]
                st.dataframe(summary, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Clean footage! No garbage found.")

# ==================== WASTE ASSISTANT ====================
elif page == "♻️ Waste Assistant":
    st.markdown("## ♻️ Personal Waste Assistant")
    
    src = st.radio("Source", ["Upload", "Camera"], horizontal=True)
    img_file = st.file_uploader("Upload item", type=["jpg", "png", "jpeg"]) if src == "Upload" else st.camera_input("Take photo")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        with st.spinner("Identifying..."):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]

        st.image(res.plot()[:, :, ::-1], caption="Result", use_container_width=True)
        items = count_detected_objects(res)
        
        if items:
            for i in items:
                name = i["Garbage"]
                desc = GARBAGE_DESCRIPTIONS.get(name, "Recycle properly.")
                st.info(f"**{name}**: {desc}")
        else:
            st.warning("No item recognized.")

# ==================== REPORT DIRTY AREA ====================
elif page == "🚨 Report a Dirty Area":
    st.markdown("## 🚨 Report a Dirty Area")
    
    chosen_area = choose_area_menu("report_area")
    img_file = st.file_uploader("Upload Area Photo", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        
        with st.spinner("Analyzing area..."):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]

        items = count_detected_objects(res)
        num_obj = len(items)
        priority = "🔴 High" if num_obj > 5 else ("🟠 Medium" if num_obj > 2 else "🟢 Low")
        
        next_id = 1021 + len(st.session_state.reports_list)
        report_id = f"Report #{next_id}"

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Evidence", use_container_width=True)
        with col2:
            st.markdown(f"### **{report_id}**")
            st.markdown(f"📍 **Location:** {chosen_area}")
            st.markdown(f"🗑️ **Objects:** {num_obj}")
            st.markdown(f"⚡ **Priority:** {priority}")
            st.markdown(f"🔄 **Status:** 🟡 Pending")

        if st.button("🚀 Submit Report", type="primary"):
            new_rep = {
                "ID": report_id, "Area": chosen_area, "Objects": num_obj,
                "Priority": priority, "Date": datetime.now().strftime("%d %b %Y"), "Status": "Pending Review"
            }
            st.session_state.reports_list.append(new_rep)
            st.success(f"Report {report_id} submitted successfully!")

# ==================== ANALYTICS DASHBOARD ====================
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Analytics Dashboard")
    
    df = pd.DataFrame(st.session_state.reports_list)
    
    total = len(df)
    resolved = len(df[df["Status"] == "Resolved"])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reports", total)
    c2.metric("Resolved", resolved)
    c3.metric("Active Regions", df["Area"].nunique() if not df.empty else 0)

    st.markdown("---")
    st.markdown("### Reports by Area")
    if not df.empty:
        st.bar_chart(df["Area"].value_counts())
    
    st.markdown("### Reports Log")
    st.dataframe(df, use_container_width=True, hide_index=True)
