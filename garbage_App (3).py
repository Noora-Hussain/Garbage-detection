# Imports
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
import folium
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer

from logic(1) import (
    GARBAGE_DESCRIPTIONS,
    coords,
    RTC_CONFIGURATION,
    load_my_model,
    count_detected_objects,
    get_user_location,
    load_reports,
    add_report,
    update_report_status,
    YOLOProcessor,
)

st.set_page_config(page_title="EcoVision | Smart Waste Detection", page_icon="♻️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #F4F7F5; }
    [data-testid="stSidebar"] { background-color: #18352B; }
    [data-testid="stSidebar"] * { color: white; }
    div.stButton > button { border-radius: 12px; border: 1px solid #2F6B4F; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

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

    # خيارين بس الحين: رفع صورة، أو كاميرا لايف مباشرة (بدل صورة واحدة من الكاميرا)
    source_type = st.radio("Source", ["📷 Image Upload", "🎥 Live Camera"], horizontal=True)

    if source_type == "🎥 Live Camera":
        # webrtc_streamer يفتح اتصال مباشر بين المتصفح والسيرفر ويمرر كل فريم لـ YOLOProcessor
        ctx = webrtc_streamer(
            key="live-detection",
            video_processor_factory=YOLOProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

        # تحديث قيمة الثقة (confidence) بالمعالج كل مرة يتغير فيها السلايدر
        if ctx.video_processor:
            ctx.video_processor.confidence = confidence

    else:  # 📷 Image Upload
        img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if img_file is not None:
            image = Image.open(img_file).convert("RGB")
            with st.spinner("AI is analyzing..."):
                model = load_my_model()
                res = model.predict(image, conf=confidence, verbose=False)[0]

                # يقسم الشاشة لقسمين: عمود للصورة الأصلية وعمود للنتيجة بعد التحديد
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image", use_container_width=True)
                with col2:
                    # channels="BGR" عشان الألوان تطلع صحيحة (res.plot ترجع BGR)
                    st.image(res.plot(), caption="AI Detection Result", use_container_width=True, channels="BGR")

                items = count_detected_objects(res)
                if items:
                    found_types = sorted(set(i["Garbage"] for i in items))
                    st.warning(f"⚠️ Garbage found! Type(s): {', '.join(found_types)}")
                    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
                else:
                    st.success("✅ Clean street! No significant garbage detected.")

            if items:
                default_desc = "Recycle properly."
                for i in items:
                    name = i["Garbage"]
                    desc_text = GARBAGE_DESCRIPTIONS.get(name, default_desc)
                    st.info(f"**{name}**: {desc_text}")

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

        # يحسب عدد قطع النفايات اللي لقاها الموديل في الصورة
        items = count_detected_objects(res)
        num_obj = len(items)
        found_names = ", ".join(set([i["Garbage"] for i in items])) if items else "None"
        
        # حساب الأولوية بناءً على عدد الأوساخ
        priority = "🔴 High" if num_obj > 5 else ("🟠 Medium" if num_obj > 2 else "🟢 Low")
        
        # تحديد رقم البلاغ
        report_id = f"Report #{1001 + len(load_reports())}"

        # عرض تفاصيل البلاغ كاملة
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

        # حفظ البلاغ في الملف عند ضغط الزر
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
