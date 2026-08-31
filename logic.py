# Imports
import os
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_geolocation import streamlit_geolocation
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration
import av

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

# best يبحث عن الملف ويحدد موقع ملف الـ
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")

# CSV يخزن البلاغات عشان ما تختفي
CSV_FILE = os.path.join(APP_FOLDER, "reports.csv")

# إعداد STUN server للكاميرا اللايف
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# يحمل ويخزن ملف الـ best عشان ما يتم اعادة تحميله في كل مرة
@st.cache_resource
def load_my_model():
    return YOLO(MODEL_PATH)

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

# تحديد الموقع تلقائياً عبر GPS 
def get_user_location():
    st.markdown("📍 **GPS Location Auto-Detection**")
    st.write("📡 Click below to fetch your current GPS location:")

    location = streamlit_geolocation()

    if location and location.get("latitude") is not None and location.get("longitude") is not None:
        lat = location["latitude"]
        lon = location["longitude"]
        area_name = get_area_name_from_coords(lat, lon)
        st.success(f"📍 Location Captured: {area_name} ({lat:.4f}, {lon:.4f})")
        return {"name": area_name, "lat": lat, "lon": lon}
    else:
        st.info("⚠️ Please allow GPS access to automatically capture your location.")
        return {"name": "Other", "lat": coords["Other"][0], "lon": coords["Other"][1]}

# دوال تخزين البلاغات (CSV)

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

# WebRTC: معالج الفيديو الحي - يشتغل على كل فريم يجي من الكاميرا


class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        # يحمل الموديل مرة وحدة لكل جلسة اتصال (session)
        self.model = load_my_model()
        self.confidence = 0.25  # قيمة افتراضية، تنحدث من السلايدر بالخارج

    def recv(self, frame):
        # يحول الفريم لمصفوفة numpy بصيغة BGR (طبيعية للفيديو)
        img = frame.to_ndarray(format="bgr24")

        # اليولو يتوقع RGB فنقلب الترتيب قبل الإرسال للموديل
        res = self.model.predict(img[..., ::-1], conf=self.confidence, verbose=False)[0]

        # res.plot() ترجع صورة بصيغة BGR أصلاً، فتطابق format="bgr24" مباشرة
        annotated = res.plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
