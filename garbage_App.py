# Imports
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation
import folium
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2

st.set_page_config(page_title="EcoVision | Smart Waste Detection", page_icon="♻️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #F4F7F5; }
    [data-testid="stSidebar"] { background-color: #18352B; }
    [data-testid="stSidebar"] * { color: white; }
    .section-title { color: #18352B; font-size: 28px; font-weight: 700; margin-top: 15px; margin-bottom: 15px; }
    div.stButton > button { border-radius: 12px; border: 1px solid #2F6B4F; font-weight: 600; }
    .footer { text-align: center; color: #718078; margin-top: 45px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# Dictionary 
GARBAGE_DESCRIPTIONS = {
    "Glass": "Place in the designated glass recycling container.",
    "Metal": "Place in the metal recycling bin.",
    "Paper": "Place in the designated paper recycling bin.",
    "Plastic": "Place in the plastic recycling bin.",
    "General Waste": "Place in the general waste bin for non-recyclable items.",
}

# إحداثيات المناطق للخريطة التفاعلية
coords = {
    "Manama": (26.2285, 50.5860),
    "Muharraq": (26.2572, 50.6119),
    "Riffa": (26.1300, 50.5550),
    "Other": (26.2000, 50.5800)
} 

# دالة لتحديد اسم أقرب منطقة من الإحداثيات الحقيقية
def get_area_name_from_coords(lat, lon):
    min_dist = float('inf')
    closest_area = "Other"
    for area, (a_lat, a_lon) in coords.items():
        if area == "Other":
            continue
        dist = np.sqrt((lat - a_lat)**2 + (lon - a_lon)**2)
        if dist < min_dist:
            min_dist = dist
            closest_area = area
    return closest_area

# يبحث عن الملف ويحدد موقع ملف الـ best
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")

# يحمل ويخزن ملف الـ best عشان ما يتم اعادة تحميله في كل مرة 
@st.cache_resource
def load_my_model():
    return YOLO(MODEL_PATH)

# كلاس معالجة الفيديو للكاميرا المباشرة
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_my_model()
        self.conf = 0.25

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # الكشف ورسم المربعات عبر YOLO
        results = self.model.predict(img, conf=self.conf, verbose=False)
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# يستخرج فئات القمامة المكتشفة و درجات الثقة من نتائج النموذج
def count_detected_objects(result):
    if result.boxes is None:
        return []
    
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({"Garbage": result.names[class_id], "Confidence": f"{conf * 100:.1f}%"})
    return detections

# تحديد الموقع تلقائياً عبر GPS
def get_user_location():
    st.markdown("📍 **GPS Location Auto-Detection**")    
    location = streamlit_geolocation()

    if location and location.get("latitude") is not None and location.get("longitude") is not None:
        lat = location["latitude"]
        lon = location["longitude"]
        area_name = get_area_name_from_coords(lat, lon)
        st.success(f"📍 Location Captured: {area_name}")
        return {"name": area_name, "lat": lat, "lon": lon}
    else:
        return {"name": "Other", "lat": coords["Other"][0], "lon": coords["Other"][1]}

# يخزن البلاغات في ملف CSV عشان ما تختفي
CSV_FILE = os.path.join(APP_FOLDER, "reports.csv")

def load_reports():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        initial_data = pd.DataFrame([
            {"ID": "Report #1001", "Area": "Manama", "Objects": 4, "Priority": "🟠 Medium", "Date": "20 Aug 2026", "Status": "Resolved", "Details": "Plastic, Metal", "lat": 26.2285, "lon": 50.5860},
            {"ID": "Report #1002", "Area": "Muharraq", "Objects": 6, "Priority": "🔴 High", "Date": "21 Aug 2026", "Status": "Pending Review", "Details": "General Waste, Plastic", "lat": 26.2572, "lon": 50.6119}
        ])
        initial_data.to_csv(CSV_FILE, index=False)
        return initial_data

# دالة تحدد موقع البلاغ و تحفظه في الملف بشكل دائم 
def add_report(new_report_dict):
    df = load_reports()
    updated_df = pd.concat([df, pd.DataFrame([new_report_dict])], ignore_index=True)
    updated_df.to_csv(CSV_FILE, index=False)

def update_report_status(report_id, new_status):
    df = load_reports()
    df.loc[df["ID"] == report_id, "Status"] = new_status
    df.to_csv(CSV_FILE, index=False)

with st.sidebar:
    st.markdown("## ♻️ EcoVision")
    st.caption("Smart Waste Detection")
    st.markdown("---")

    page_options = ["🛣️ Street Detection", "🚨 Report a Dirty Area", "📊 Analytics Dashboard", "📄 Report Generation"]
    page = st.radio("Navigation", page_options)

    st.markdown("---")
    confidence = st.slider("AI Confidence Threshold", 0.10, 0.90, 0.25, 0.05)

# STREET DETECTION
if page == "🛣️ Street Detection":
    st.markdown("## 🛣️ Street Garbage Detection")
    st.write("Analyze street footage in real-time with AI bounding boxes.")

    source_type = st.radio("Source", ["📷 Image Upload", "🎥 Live Camera"], horizontal=True)

    if source_type == "📷 Image Upload":
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if img_file is not None:
            image = Image.open(img_file).convert("RGB")
            with st.spinner("AI is analyzing..."):
                model = load_my_model()
                res = model.predict(image, conf=confidence, verbose=False)[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    st.image(res.plot(), caption="AI Detection Result", use_container_width=True, channels="BGR")

                items = count_detected_objects(res)
                if items:
                    found_types = sorted(set(i["Garbage"] for i in items))
                    st.warning(f"⚠️ Garbage found! Type(s): {', '.join(found_types)}")
                    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                    
                    default_desc = "Recycle properly."
                    for i in items:
                        name = i["Garbage"]
                        desc_text = GARBAGE_DESCRIPTIONS.get(name, default_desc)
                        st.info(f"**{name}**: {desc_text}")
                else:
                    st.success("✅ Clean street! No significant garbage detected.")

    urls = ["turns:global.relay.metered.ca:80","turns:global.relay.metered.ca:80?transport=tcp","turns:global.relay.metered.ca:443","turns:global.relay.metered.ca:443?transport=tcp"]
    username =  "bdec4905eb8eaaa33b8aa1e4"
    credential = "euaq/TCzIAMInxyy
    
    elif source_type == "🎥 Live Camera":
        st.info("Click **START** below to enable webcam detection.")
        
        # إعدادات WebRTC وتمرير نسبة الثقة (Confidence) للمعالج
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        
        ctx = webrtc_streamer(
            key="yolo-live-detection",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False}
        )

# REPORT DIRTY AREA 
elif page == "🚨 Report a Dirty Area":
    st.markdown("## 🚨 Report a Dirty Area with GPS Tagging")
    
    chosen_area = get_user_location()
    img_file = st.file_uploader("Upload Area Photo", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")

        with st.spinner("Analyzing area & priority..."):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]

        items = count_detected_objects(res)
        num_obj = len(items)
        found_names = ", ".join(set([i["Garbage"] for i in items])) if items else "None"
        
        priority = "🔴 High" if num_obj > 5 else ("🟠 Medium" if num_obj > 2 else "🟢 Low")
        report_id = f"Report #{1001 + len(load_reports())}"

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Evidence Image", use_container_width=True)
        with col2:
            st.markdown(f"### **{report_id}**")
            st.markdown(f"📍 **Location:** {chosen_area['name']}")
            st.markdown(f"🗑️ **Count:** {num_obj}")
            st.markdown(f"🏷️ **Types:** {found_names}")
            st.markdown(f"⚡ **Priority:** {priority}")

        if st.button("🚀 Submit Report", type="primary"):
            new_rep = {
                "ID": report_id, 
                "Area": chosen_area["name"], 
                "Objects": num_obj,
                "Priority": priority, 
                "Date": datetime.now().strftime("%d %b %Y"), 
                "Status": "Pending Review",
                "Details": found_names,
                "lat": chosen_area["lat"],
                "lon": chosen_area["lon"]
            }
            add_report(new_rep)
            st.success(f"Report {report_id} successfully submitted and saved!")

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
    
    st.markdown("### 🗺️ Live Reports Map")
    if not df.empty:
        df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
        df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
        df["lat"] = df["lat"].fillna(df["Area"].map(lambda x: coords.get(x, coords["Other"])[0]))
        df["lon"] = df["lon"].fillna(df["Area"].map(lambda x: coords.get(x, coords["Other"])[1]))

        m = folium.Map(location=[26.2285, 50.5860], zoom_start=11)

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=min(max(float(row.get("Objects", 1)) * 3, 6), 20),
                popup=f"<b>{row['ID']}</b><br>Area: {row['Area']}",
                color="#e74c3c" if "High" in str(row.get("Priority", "")) else "#2ecc71",
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

        st_folium(m, use_container_width=True, height=400)
    
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

    st.markdown("---")
    st.markdown("### 🛠️ Admin Status Manager")
    if not df.empty:
        pending_reports = df[df["Status"] != "Resolved"]["ID"].tolist()
        if pending_reports:
            selected_rep = st.selectbox("Select Pending Report to Resolve:", pending_reports)
            if st.button("Mark as Resolved ✅"):
                update_report_status(selected_rep, "Resolved")
                st.success(f"Status of {selected_rep} updated to Resolved!")
                st.rerun()
        else:
            st.info("All reported areas are currently resolved!")

# REPORT GENERATION
elif page == "📄 Report Generation":
    st.markdown("## 📄 Report Generation")
    df = load_reports()

    period = st.radio("Report Period", ["Today", "Last 7 Days", "All"], horizontal=True)

    if period != "All" and not df.empty:
        days = 1 if period == "Today" else 7
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df["ParsedDate"] = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce")
        df = df[df["ParsedDate"] >= cutoff].drop(columns=["ParsedDate"])

    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=csv_data,
        file_name="ecovision_report.csv",
        mime="text/csv",
    )
