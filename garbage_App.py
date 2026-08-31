# Imports
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="EcoVision | Smart Waste Detection", page_icon="♻️", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    lang = st.radio("🌐 Language / اللغة", ["English", "العربية"], horizontal=True)

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

GARBAGE_DESCRIPTIONS_AR = {
    "Glass": "ضعها في حاوية الزجاج المخصصة.",
    "Metal": "ضعها في حاوية إعادة تدوير المعادن.",
    "Paper": "ضعها في حاوية الورق المخصصة.",
    "Plastic": "ضعها في حاوية البلاستيك.",
    "General Waste": "ضعها في حاوية النفايات العامة.",
}

# إحداثيات المناطق للخريطة التفاعلية
coords = {
    "Manama": (26.2285, 50.5860),
    "Manama (GPS)": (26.2285, 50.5860),
    "Muharraq": (26.2572, 50.6119),
    "Riffa": (26.1300, 50.5550),
    "Other": (26.2000, 50.5800)
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
    st.markdown("📍 **Location & GPS Source**" if lang == "English" else "📍 **الموقع ومصدر GPS**")
    use_gps = st.checkbox(
        "Use Simulated GPS Coordinates" if lang == "English" else "استخدام إحداثيات GPS تجريبية",
        value=True, key=f"gps_{unique_key}"
    )
    
    # اذا المستخدم اختار تحديد الموقع نحطه و اذا م اختار نحط ليه قائمة المناطق يختار منها 
    if use_gps:
        return "Manama (GPS)"
    
    selected = st.selectbox(
        "Area:" if lang == "English" else "المنطقة:",
        ["Manama", "Muharraq", "Riffa", "Other"], key=unique_key
    )
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
    st.caption("Smart Waste Detection" if lang == "English" else "كشف ذكي للنفايات")
    st.markdown("---")

    page_options_en = ["🛣️ Street Detection", "🚨 Report a Dirty Area", "📊 Analytics Dashboard", "📄 Report Generation"]
    page_options_ar = ["🛣️ كشف الشوارع", "🚨 الإبلاغ عن منطقة متسخة", "📊 لوحة التحليلات", "📄 إنشاء التقارير"]

    nav_label = "Navigation" if lang == "English" else "التنقل"
    page_options = page_options_en if lang == "English" else page_options_ar
    page_choice = st.radio(nav_label, page_options)

    # نحول اختيار المستخدم لمفتاح إنجليزي موحّد يستخدمه باقي الكود
    page_index = page_options.index(page_choice)
    page = page_options_en[page_index]

    st.markdown("---")
    confidence_label = "AI Confidence Threshold" if lang == "English" else "درجة ثقة الذكاء الاصطناعي"
    confidence = st.slider(confidence_label, 0.10, 0.90, 0.45, 0.05)

# STREET DETECTION 

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}) 

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = load_my_model().predict(img, conf=0.45, verbose=False)[0]
        return av.VideoFrame.from_ndarray(results.plot(), format="bgr24")

if page == "🛣️ Street Detection":
    title = "🛣️ Street Garbage Detection" if lang == "English" else "🛣️ كشف القمامة في الشوارع"
    desc = ("Analyze street footage with optimized AI thresholding to avoid false detections."
            if lang == "English" else
            "تحليل صور الشوارع باستخدام الذكاء الاصطناعي لتجنب الأخطاء في الكشف.")
    st.markdown(f"## {title}")
    st.write(desc)

    # وجود زرين لرفع صورة او كاميرا لايف
    source_label = "Source" if lang == "English" else "المصدر"
    source_options_en = ["📷 Image Upload", "📸 Live Camera"]
    source_options_ar = ["📷 رفع صورة", "📸 كاميرا مباشرة"]
    source_options = source_options_en if lang == "English" else source_options_ar
    source_choice = st.radio(source_label, source_options, horizontal=True)
    source_type = source_options_en[source_options.index(source_choice)]

    # في حال رفع صورة لازم تكون jpg", "png", "jpeg واذا ما كانت برفضها 
    img_file = None
    if source_type == "📷 Image Upload":
        upload_label = "Upload Image" if lang == "English" else "ارفعي صورة"
        img_file = st.file_uploader(upload_label, type=["jpg", "png", "jpeg"])
     
    else:
        info_msg = "Allow camera access to start live detection." if lang == "English" else "اسمحي بالوصول للكاميرا لبدء الكشف المباشر."
        st.info(info_msg)
        webrtc_streamer(key="live-detection", video_processor_factory=YOLOProcessor)

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        spinner_msg = "AI is analyzing and filtering noise" if lang == "English" else "الذكاء الاصطناعي يحلل الصورة الآن"
        with st.spinner(spinner_msg):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]
                
            # يقسم الشاشة لقسمين عمود للصورة الاصلية و عمود للصورة بعد تحديد المربعات
            col1, col2 = st.columns(2)
            with col1:
                cap1 = "Original" if lang == "English" else "الصورة الأصلية"
                st.image(image, caption=cap1)
            with col2:
                cap2 = "AI Filtered Detection" if lang == "English" else "نتيجة الكشف بالذكاء الاصطناعي"
                st.image(res.plot()[:, :, ::-1], caption=cap2, use_container_width=True) 

            # يحسب عدد الاجسام المكتشفة و اذا وجد يعرض تحذير واذا م وجد يعرض رسالة ان الشارع نظيف 
            items = count_detected_objects(res)
            if items:
                found_types = sorted(set(i["Garbage"] for i in items))
                warn_msg = (f"⚠️ Garbage found! Type(s): {', '.join(found_types)}"
                            if lang == "English" else
                            f"⚠️ تم العثور على قمامة! النوع: {', '.join(found_types)}")
                st.warning(warn_msg)
                st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
            else:
                clean_msg = "✅ Clean street! No significant garbage detected." if lang == "English" else "✅ الشارع نظيف! لم يتم رصد قمامة."
                st.success(clean_msg)
                
        # التحقق من وجود قمامة واذا فيه يوصف طريقة التخلص منها 
        if items:
            desc_dict = GARBAGE_DESCRIPTIONS if lang == "English" else GARBAGE_DESCRIPTIONS_AR
            default_desc = "Recycle properly." if lang == "English" else "أعيدي تدويرها بالشكل الصحيح."
            for i in items:
                name = i["Garbage"]
                desc_text = desc_dict.get(name, default_desc)
                st.info(f"**{name}**: {desc_text}")
        else:
            warn2 = "No item recognized above confidence threshold." if lang == "English" else "لم يتم التعرف على أي عنصر ضمن نسبة الثقة المحددة."
            st.warning(warn2)

# REPORT DIRTY AREA 
elif page == "🚨 Report a Dirty Area":
    title = "🚨 Report a Dirty Area with GPS Tagging" if lang == "English" else "🚨 الإبلاغ عن منطقة متسخة مع تحديد الموقع"
    st.markdown(f"## {title}")
    
    chosen_area = choose_area_menu("report_area")
    upload_label2 = "Upload Area Photo" if lang == "English" else "ارفعي صورة المنطقة"
    img_file = st.file_uploader(upload_label2, type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")

        spinner2 = "Analyzing area & priority..." if lang == "English" else "جاري تحليل المنطقة وتحديد الأولوية..."
        with st.spinner(spinner2):
            model = load_my_model()
            res = model.predict(image, conf=confidence, verbose=False)[0]

        # يحسب عدد قطع النفايات اللي لقاها الموديل في الصورة
        items = count_detected_objects(res)
        num_obj = len(items)
        none_txt = "None" if lang == "English" else "لا يوجد"
        found_names = ", ".join(set([i["Garbage"] for i in items])) if items else none_txt
        
        #  حساب الاولوية بناء على عدد الاوساخ (نخليها إنجليزي داخليًا عشان باقي الكود يقارن عليها بدون مشاكل)
        priority = "🔴 High" if num_obj > 5 else ("🟠 Medium" if num_obj > 2 else "🟢 Low")
        priority_display = priority
        if lang == "العربية":
            priority_map = {"🔴 High": "🔴 عالية", "🟠 Medium": "🟠 متوسطة", "🟢 Low": "🟢 منخفضة"}
            priority_display = priority_map[priority]
        
        # تحديد رقم البلاغ
        report_id = f"Report #{1001 + len(load_reports())}"

        # عرض تفاصيل البلاغ كلها
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            cap3 = "Evidence Image" if lang == "English" else "صورة الإثبات"
            st.image(image, caption=cap3, use_container_width=True)
        with col2:
            st.markdown(f"### **{report_id}**")
            if lang == "English":
                st.markdown(f"📍 **Location:** {chosen_area}")
                st.markdown(f"🗑️ **Count:** {num_obj}")
                st.markdown(f"🏷️ **Types:** {found_names}")
                st.markdown(f"⚡ **Priority:** {priority_display}")
            else:
                st.markdown(f"📍 **الموقع:** {chosen_area}")
                st.markdown(f"🗑️ **العدد:** {num_obj}")
                st.markdown(f"🏷️ **الأنواع:** {found_names}")
                st.markdown(f"⚡ **الأولوية:** {priority_display}")

        # في حال ضغط زر البلاغ يجمع الكود كل المعلومات و يحفضها في النظام
        submit_label = "🚀 Submit Report" if lang == "English" else "🚀 إرسال البلاغ"
        if st.button(submit_label, type="primary"):
            new_rep = {
                "ID": report_id, 
                "Area": chosen_area, 
                "Objects": num_obj,
                "Priority": priority,   # نخزن دايمًا بالإنجليزي بالملف عشان الفلاتر ما تنكسر
                "Date": datetime.now().strftime("%d %b %Y"), 
                "Status": "Pending Review",
                "Details": found_names
            }
            add_report(new_rep)
            success_msg = f"Report {report_id} successfully submitted and saved!" if lang == "English" else f"تم إرسال وحفظ البلاغ {report_id} بنجاح!"
            st.success(success_msg)


# ANALYTICS DASHBOARD 

elif page == "📊 Analytics Dashboard":
    title = "📊 Analytics Dashboard & High-Density Insights" if lang == "English" else "📊 لوحة التحليلات والبيانات"
    st.markdown(f"## {title}")

    # اخذ كل البلاغات الموجوده في النظام وحساب عددها كامل وعدد يلي انحلت
    df = load_reports()
    
    total = len(df)
    resolved = len(df[df["Status"] == "Resolved"]) if not df.empty else 0
    high_priority_count = len(df[df["Priority"] == "🔴 High"]) if not df.empty else 0

    # عرض الارقام بشكل واضح
    c1, c2, c3, c4 = st.columns(4)
    if lang == "English":
        c1.metric("Total Reports", total)
        c2.metric("Resolved Reports", resolved)
        c3.metric("High-Density Areas", high_priority_count, help="Areas with heavy waste concentration")
        c4.metric("Active Regions", df["Area"].nunique() if not df.empty else 0)
    else:
        c1.metric("إجمالي البلاغات", total)
        c2.metric("البلاغات المحلولة", resolved)
        c3.metric("مناطق عالية الكثافة", high_priority_count, help="مناطق فيها تركيز كبير من النفايات")
        c4.metric("المناطق النشطة", df["Area"].nunique() if not df.empty else 0)

    st.markdown("---")
    
    # إضافة إحداثيات المناطق تلقائياً
    map_title = "### 🗺️ Live Reports Map" if lang == "English" else "### 🗺️ خريطة البلاغات المباشرة"
    st.markdown(map_title)
    if not df.empty:
        df["lat"] = df["Area"].map(lambda x: coords.get(x, (26.2285, 50.5860))[0])
        df["lon"] = df["Area"].map(lambda x: coords.get(x, (26.2285, 50.5860))[1])
        df["size"] = df["Objects"] * 30
        st.map(df, latitude="lat", longitude="lon", size="size", zoom=10)

    
    # الرسم البياني 
    col_a, col_b = st.columns(2)
    with col_a:
        chart_title = "### 📈 Reports Distribution by Area" if lang == "English" else "### 📈 توزيع البلاغات حسب المنطقة"
        st.markdown(chart_title)
        if not df.empty:
            st.bar_chart(df["Area"].value_counts())
            
     # جدول الاماكن يلي اولويتها عالية    
    with col_b:
        hotspot_title = "### 🚨 High-Density Garbage Hotspots" if lang == "English" else "### 🚨 المناطق شديدة التلوث"
        st.markdown(hotspot_title)
        if not df.empty and "Priority" in df.columns:
            hotspots = df[df["Priority"] == "🔴 High"]
            if not hotspots.empty:
                st.dataframe(hotspots[["ID", "Area", "Objects", "Date", "Status"]], use_container_width=True, hide_index=True)
            else:
                no_hotspot = "No high-density critical hotspots reported yet." if lang == "English" else "لا توجد مناطق حرجة عالية الكثافة حتى الآن."
                st.info(no_hotspot)

    # Report Generation
elif page == "📄 Report Generation":
    
    # يسوي جدول كامل يعرض البلاغات 
    title = "📄 Report Generation" if lang == "English" else "📄 إنشاء التقارير"
    st.markdown(f"## {title}")
    df = load_reports()

    # إضافة: فلترة يومي/أسبوعي
    period_label = "Report Period" if lang == "English" else "الفترة الزمنية للتقرير"
    period_options_en = ["Today", "Last 7 Days", "All"]
    period_options_ar = ["اليوم", "آخر 7 أيام", "الكل"]
    period_options = period_options_en if lang == "English" else period_options_ar
    period_choice = st.radio(period_label, period_options, horizontal=True)
    period = period_options_en[period_options.index(period_choice)]

    if period != "All":
        days = 1 if period == "Today" else 7
        cutoff = datetime.now() - pd.Timedelta(days=days)
        df = df[pd.to_datetime(df["Date"], format="%d %b %Y") >= cutoff]

    st.dataframe(df, use_container_width=True, hide_index=True)
 
    csv_data = df.to_csv(index=False).encode("utf-8")
    download_label = "⬇️ Download Report (CSV)" if lang == "English" else "⬇️ تحميل التقرير (CSV)"
    st.download_button(
        label=download_label,
        data=csv_data,
        file_name="ecovision_report.csv",
        mime="text/csv",
    )
