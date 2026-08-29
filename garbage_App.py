# Imports
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

# Dictionary 
GARBAGE_DESCRIPTIONS = {
    "Glass": "Place in the designated glass recycling container.",
    "Metal": "Place in the metal recycling bin.",
    "Paper": "Place in the designated paper recycling bin.",
    "Plastic": "Place in the plastic recycling bin.",
    "General Waste": "Place in the general waste bin for non-recyclable items.",
}

# bestيبحث عن الملف و يحدد موقع ملف ال 
APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")

# يحمل و يخزن ملف ال best عشان ما يتم اعادة تحميله في كل مرة 
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

# لتحديد الموقع و تحديد مكان البلاغ بدقة Checkbox تعرض للمستخدم خيار ال
def choose_area_menu(unique_key):
    st.markdown("📍 **Location & GPS Source**")
    use_gps = st.checkbox("Use Simulated GPS Coordinates", value=True, key=f"gps_{unique_key}")
    
    # اذا المستخدم اختار تحديد الموقع نحطه و اذا م اختار نحط ليه قائمة المناطق يختار منها 
    if use_gps:
        return "Manama (GPS)"
    
    selected = st.selectbox("Area:", ["Manama", "Muharraq", "Riffa", "Other"], key=unique_key)
    return selected

# CSV يخزن البلاغات عشان ما تختفي
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

# دالة تحدد موقع البلاغ و تحفظه في الملف بشكل دائم 
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
    confidence = st.slider("AI Confidence Threshold", 0.10, 0.90, 0.45, 0.05, )

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

    # وجود زرين لرفع صورة او كاميرا لايف
    source_type = st.radio(
        "Source",
        ["📷 Image"],
        horizontal=True)

    # في حال رفع صورة لازم تكون jpg", "png", "jpeg واذا ما كانت برفضها 
    if source_type == "📷 Image":
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if img_file is not None:
            image = Image.open(img_file).convert("RGB")
            
            with st.spinner("AI is analyzing and filtering noise"):
               model = load_my_model()
               res = model.predict(image, conf=confidence, verbose=False)[0]
                
            # يقسم الشاشة لقسمين عمود للصورة الاصلية و عمود للصورة بعد تحديد المربعات
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original")
            with col2:
                st.image(res.plot()[:, :, ::-1], caption="AI Filtered Detection", use_container_width=True) 

            # يحسب عدد الاجسام المكتشفة و اذا وجد يعرض تحذير واذا م وجد يعرض رسالة ان الشارع نظيف 
            items = count_detected_objects(res)
            if items:
                found_types = sorted(set(i["Garbage"] for i in items))
                st.warning(f"⚠️ Garbage found! Type(s): {', '.join(found_types)}")
                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
            else:
                st.success("✅ Clean street! No significant garbage detected.")
   #  في حال اختيار الكاميرا لايف          
    st.markdown("### 📸 Live Camera Detection")

    cam_file = st.camera_input("Take a photo")

    # نتاكد هل المستخدم التقط صورة او لا 
    if cam_file is not None:
        image = Image.open(cam_file).convert("RGB")
        # يمرر الصورة على الموديل عشان يشوفها
        with st.spinner("AI is analyzing..."):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]

    # عرض الصورة مع المربعات لتحديد مكان القمامة
        st.image(res.plot()[:, :, ::-1], caption="AI Detection Result", use_container_width=True) 

    # اذا شاف قمامة يجمعها و يحدد نوعها واذا ما شاف يكتب ان الشارع نظيف 
        items = count_detected_objects(res)
        if items:
            found_types = sorted(set(i["Garbage"] for i in items))
            st.warning(f"⚠️ Garbage found! Type(s): {', '.join(found_types)}")
            st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Clean area! No garbage detected.")

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

        # التحقق من وجود قمامة واذا فيه يوصف طريقة التخلص منها 
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

        with st.spinner("Analyzing area & priority..."):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]

        # يحسب عدد قطع النفايات اللي لقاها الموديل في الصورة
        items = count_detected_objects(res)
        num_obj = len(items)
        found_names = ", ".join(set([i["Garbage"] for i in items])) if items else "None"
        
        #  حساب الاولوية بناء على عدد الاوساخ
        priority = "🔴 High" if num_obj > 5 else ("🟠 Medium" if num_obj > 2 else "🟢 Low")
        
        # تحديد رقم البلاغ
        report_id = f"Report #{1001 + len(load_reports())}"

        # عرض تفاصيل البلاغ كلها
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Evidence Image", use_container_width=True)
        with col2:
            st.markdown(f"### **{report_id}**")
            st.markdown(f"📍 **Location:** {chosen_area}")
            st.markdown(f"🗑️ **Count:** {num_obj}")
            st.markdown(f"🏷️ **Types:** {found_names}")
            st.markdown(f"⚡ **Priority:** {priority}")

        # في حال ضغط زر البلاغ يجمع الكود كل المعلومات و يحفضها في النظام
        if st.button("🚀 Submit Report", type="primary"):
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
            st.success(f"Report {report_id} successfully submitted and saved!")


# ANALYTICS DASHBOARD 
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Analytics Dashboard & High-Density Insights")

    # اخذ كل البلاغات الموجوده في النظام وحساب عددها كامل وعدد يلي انحلت
    df = load_reports()
    
    total = len(df)
    resolved = len(df[df["Status"] == "Resolved"]) if not df.empty else 0
    high_priority_count = len(df[df["Priority"] == "🔴 High"]) if not df.empty else 0

    # عرض الارقام بشكل واضح
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reports", total)
    c2.metric("Resolved Reports", resolved)
    c3.metric("High-Density Areas", high_priority_count, help="Areas with heavy waste concentration")
    c4.metric("Active Regions", df["Area"].nunique() if not df.empty else 0)

    st.markdown("---")

    # الرسم البياني 
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📈 Reports Distribution by Area")
        if not df.empty:
            st.bar_chart(df["Area"].value_counts())
            
     # جدول الاماكن يلي اولويتها عالية    
    with col_b:
        st.markdown("### 🚨 High-Density Garbage Hotspots")
        if not df.empty and "Priority" in df.columns:
            hotspots = df[df["Priority"] == "🔴 High"]
            if not hotspots.empty:
                st.dataframe(hotspots[["ID", "Area", "Objects", "Date", "Status"]], use_container_width=True, hide_index=True)
            else:
                st.info("No high-density critical hotspots reported yet.")
    # يسوي جدول كامل يعرض البلاغات 
    st.markdown("### 📋 Complete Reports Log")
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
