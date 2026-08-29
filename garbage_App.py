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
    st.markdown("📍 **Location & GPS Source**")
    use_gps = st.checkbox("Use Simulated GPS Coordinates", value=True, key=f"gps_{unique_key}")
    
    if use_gps:
        # محاكاة لجلب الإحداثيات الجغرافية الحقيقية (GPS Simulation for Bahrain regions)
        gps_lat = 26.2285
        gps_lon = 50.5860
        st.caption(f"🛰️ GPS Coordinates Acquired: Lat {gps_lat}, Lon {gps_lon} (Manama Sector)")
        return "Manama (GPS Verified)"
    else:
        areas = ["Manama", "Muharraq", "Riffa", "Isa Town", "Hamad Town", "Other"]
        selected = st.selectbox("Select Area manually:", areas, key=unique_key)
        
        if selected == "Other":
            custom = st.text_input("Enter area name:", key=f"custom_{unique_key}")
            return custom if custom else "Unspecified"
        return selected

CSV_FILE = os.path.join(APP_FOLDER, "reports.csv")

def load_reports():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        initial_data = pd.DataFrame([
            {"ID": "Report #1001", "Area": "Manama", "Objects": 4, "Priority": "🟠 Medium", "Date": "20 Aug 2026", "Status": "Resolved", "Details": "Plastic, Metal"},
            {"ID": "Report #1002", "Area": "Muharraq", "Objects": 6, "Priority": "🔴 High", "Date": "21 Aug 2026", "Status": "Pending Review", "Details": "General Waste, Plastic"}
        ])
        initial_data.to_csv(CSV_FILE, index=False)
        return initial_data

def add_report(new_report_dict):
    df = load_reports()
    updated_df = pd.concat([df, pd.DataFrame([new_report_dict])], ignore_index=True)
    updated_df.to_csv(CSV_FILE, index=False)

with st.sidebar:
    st.markdown("## ♻️ EcoVision")
    st.caption("Smart Waste Detection")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🛣️ Street Detection", "♻️ Waste Assistant", "🚨 Report a Dirty Area", "📊 Analytics Dashboard"])

    st.markdown("---")
    # رفع القيمة الافتراضية للـ Confidence إلى 0.45 لتفادي الإنذارات الخاطئة (False Positives)
    confidence = st.slider("AI Confidence Threshold", 0.10, 0.90, 0.45, 0.05, 
                           help="Adjust to filter out low-confidence false positives.")

# HOME PAGE 
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
            <p>Submit photos of polluted areas with GPS location tagging, automatic AI object count, and priority level assignment.</p>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Analytics Dashboard</h3>
            <p>Track high-density garbage locations, regional clean-up statistics, and active area distributions.</p>
        </div>
        """, unsafe_allow_html=True)

# STREET DETECTION 
elif page == "🛣️ Street Detection":
    st.markdown("## 🛣️ Street Garbage Detection")
    st.write("Analyze street footage with optimized AI thresholding to avoid false detections.")

    source_type = st.radio(
        "Source",
        ["📷 Image", "🎞️ Video Upload"],
        horizontal=True)
    
    if source_type == "📷 Image":
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if img_file is not None:
            image = Image.open(img_file).convert("RGB")

            with st.spinner("AI is analyzing and filtering noise..."):
               model = load_my_model()
               res = model.predict(image, conf=confidence, verbose=False)[0]

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(res.plot()[:, :, ::-1], caption="AI Filtered Detection", use_container_width=True)

            items = count_detected_objects(res)
            if items:
                found_types = sorted(set(i["Garbage"] for i in items))
                st.warning(f"⚠️ Garbage found! Type(s): {', '.join(found_types)}")
                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
            else:
                st.success("✅ Clean street! No significant garbage detected.")

    elif source_type == "🎞️ Video Upload":
        video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
        frame_skip = 5 

        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
            tfile.write(video_file.read())
            tfile.close()

            all_detections = []
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            st_frame = st.empty()
            progress_bar = st.progress(0)
            model = load_my_model()

            with st.spinner("Processing video frames..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = model.predict(frame_rgb, conf=confidence, verbose=False)[0]

                    items = count_detected_objects(res)
                    all_detections.extend(items)

                    annotated_frame = res.plot()
                    st_frame.image(annotated_frame[:, :, ::-1], caption=f"Analyzing Frame {frame_count}", use_container_width=True)

                    if total_frames > 0:
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            os.unlink(tfile.name)

            st.markdown("---")
            if all_detections:
                st.warning(f"⚠️ Found {len(all_detections)} validated garbage instances across analyzed frames!")
                summary = pd.DataFrame(all_detections)["Garbage"].value_counts().reset_index()
                summary.columns = ["Garbage", "Count"]
                st.dataframe(summary, use_container_width=True, hide_index=True)
            else:
                st.success("✅ Clean footage! No garbage found.")

# WASTE ASSISTANT 
elif page == "♻️ Waste Assistant":
    st.markdown("## ♻️ Personal Waste Assistant")
    
    src = st.radio("Source", ["Upload", "Camera"], horizontal=True)
    img_file = st.file_uploader("Upload item", type=["jpg", "png", "jpeg"]) if src == "Upload" else st.camera_input("Take photo")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        with st.spinner("Identifying item..."):
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
            st.warning("No item recognized above confidence threshold.")

# REPORT DIRTY AREA 
elif page == "🚨 Report a Dirty Area":
    st.markdown("## 🚨 Report a Dirty Area with GPS Tagging")
    
    chosen_area = choose_area_menu("report_area")
    img_file = st.file_uploader("Upload Area Photo", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        
        with st.spinner("Analyzing area and calculating priority..."):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]

        items = count_detected_objects(res)
        num_obj = len(items)
        found_names = ", ".join(set([i["Garbage"] for i in items])) if items else "None"
        priority = "🔴 High" if num_obj > 5 else ("🟠 Medium" if num_obj > 2 else "🟢 Low")
        
        df_current = load_reports()
        next_id = 1001 + len(df_current)
        report_id = f"Report #{next_id}"

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Evidence Image", use_container_width=True)
        with col2:
            st.markdown(f"### **{report_id}**")
            st.markdown(f"📍 **Location/GPS:** {chosen_area}")
            st.markdown(f"🗑️ **Detected Objects Count:** {num_obj}")
            st.markdown(f"🏷️ **Waste Types:** {found_names}")
            st.markdown(f"⚡ **Assigned Priority:** {priority}")
            st.markdown(f"🔄 **Status:** 🟡 Pending Review")

        if st.button("🚀 Submit Report & Save Data", type="primary"):
            new_rep = {
                "ID": report_id, 
                "Area": chosen_area, 
                "Objects": num_obj,
                "Priority": priority, 
                "Date": datetime.now().strftime("%d %b %Y"), 
                "Status": "Pending Review",
                "Details": found_names
            }
            add_report(new_rep)
            st.success(f"Report {report_id} successfully submitted, geo-tagged, and saved to database!")

# ANALYTICS DASHBOARD 
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Analytics Dashboard & High-Density Insights")
    
    df = load_reports()
    
    total = len(df)
    resolved = len(df[df["Status"] == "Resolved"]) if not df.empty else 0
    high_priority_count = len(df[df["Priority"] == "🔴 High"]) if not df.empty else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reports", total)
    c2.metric("Resolved Reports", resolved)
    c3.metric("High-Density Areas", high_priority_count, help="Areas with heavy waste concentration")
    c4.metric("Active Regions", df["Area"].nunique() if not df.empty else 0)

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📈 Reports Distribution by Area")
        if not df.empty:
            st.bar_chart(df["Area"].value_counts())
            
    with col_b:
        st.markdown("### 🚨 High-Density Garbage Hotspots")
        if not df.empty and "Priority" in df.columns:
            hotspots = df[df["Priority"] == "🔴 High"]
            if not hotspots.empty:
                st.dataframe(hotspots[["ID", "Area", "Objects", "Date", "Status"]], use_container_width=True, hide_index=True)
            else:
                st.info("No high-density critical hotspots reported yet.")

    st.markdown("### 📋 Complete Reports Log")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
