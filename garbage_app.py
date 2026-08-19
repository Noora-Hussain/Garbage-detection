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

# --- 🎨 تصميم جديد: أزرق عصري واحترافي (Modern Blue Theme) ---
st.markdown("""
    <style>
    /* خلفية التطبيق رمادية فاتحة جداً ونظيفة */
    .stApp {
        background-color: #f4f6f9;
        color: #334155;
    }
    
    /* العناوين بلون أزرق داكن فخم */
    h1 {
        color: #1e3a8a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #2563eb;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* الشريط الجانبي بلون أزرق فاتح وهادئ */
    [data-testid="stSidebar"] {
        background-color: #e0f2fe;
        border-right: 1px solid #bae6fd;
    }
    
    /* المربعات الإحصائية (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #2563eb !important;
    }
    
    /* الأزرار بلون أزرق حيوي وجذاب */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    
    /* زر التحميل بلون مختلف مميز (نيلي) */
    [data-testid="stDownloadButton"]>button {
        background-color: #4f46e5;
    }
    [data-testid="stDownloadButton"]>button:hover {
        background-color: #4338ca;
    }

    /* صناديق المعلومات والإرشادات */
    div.stInfo {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        color: #1e40af;
    }
    
    div.stSuccess {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #22c55e;
        border-radius: 8px;
        color: #166534;
    }
    </style>
""", unsafe_allow_html=True)
