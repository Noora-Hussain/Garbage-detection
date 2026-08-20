import streamlit as st
import pandas as pd
import plotly.express as px
import io
import os
from PIL import Image
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Waste Bin",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# YOUR ORIGINAL FUNCTIONS
# =========================================================

Garbage_Classes = ['Glass', 'Metal', 'Paper', 'Plastic', 'Waste']

GARBAGE_DESCRIPTIONS = {
    "Glass": "Glass: Please place it in the glass container and ensure it is clean and unbroken.",
    "Metal": "Metal: Aluminum cans and packaging are recyclable, please empty them.",
    "Paper": "Paper: Clean and dry paper, please place it in the designated paper bin.",
    "Plastic": "Plastic: Plastic bottles and containers, please rinse them before recycling.",
    "Waste": "General Waste: These are non-recyclable wastes, please dispose of them in the general waste bin."
}


APP_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_FOLDER, "best.pt")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


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


def detect_garbage(image, confidence=0.25):
    """
    Detect garbage using the trained YOLO model.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Make sure best.pt is in the same folder."
        )

    model = load_model()

    result = model.predict(
        image,
        conf=confidence
    )[0]

    # Create annotated image
    plotted_image = result.plot()[:, :, ::-1]
    annotated_image = Image.fromarray(plotted_image)

    # Get detection information
    detections = get_detections(result)

    return annotated_image, detections


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7F6;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background-color: #17221D;
    padding-top: 25px;
}

[data-testid="stSidebar"] * {
    color: white;
}


/* ================= HIDE DEFAULT ================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ================= TITLES ================= */

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #17221D;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #68736D;
    margin-bottom: 30px;
}


/* ================= SIDEBAR LOGO ================= */

.logo-box {
    background-color: #DDF4E7;
    width: 55px;
    height: 55px;
    border-radius: 16px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 28px;
    margin-bottom: 12px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 2px;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #AAB5AF !important;
    margin-bottom: 30px;
}


/* ================= BUTTONS ================= */

.stButton > button {
    background-color: #1B4332;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #52796F;
    color: white;
}


/* ================= CARDS ================= */

.card {
    background-color: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #E4E9E6;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    min-height: 130px;
}

.card-title {
    color: #68736D;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}

.card-value {
    color: #17221D;
    font-size: 30px;
    font-weight: 800;
}

.card-icon {
    font-size: 27px;
    margin-bottom: 8px;
}


/* ================= WASTE CARDS ================= */

.waste-card {
    background-color: white;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #E4E9E6;
    text-align: center;
    min-height: 145px;
}

.waste-icon {
    font-size: 32px;
    margin-bottom: 8px;
}

.waste-name {
    font-weight: 700;
    color: #17221D;
    font-size: 15px;
}

.waste-number {
    font-size: 24px;
    font-weight: 800;
    color: #2D7A4D;
    margin-top: 5px;
}


/* ================= ONLINE ================= */

.online {
    display: inline-block;
    background-color: #D9F7E5;
    color: #1D7A46 !important;
    padding: 6px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
}


/* ================= SECTION ================= */

.section-title {
    font-size: 22px;
    font-weight: 800;
    color: #17221D;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* ================= UPLOAD ================= */

.upload-title {
    font-size: 18px;
    font-weight: 700;
    color: #17221D;
    margin-bottom: 5px;
}

.upload-description {
    color: #68736D;
    font-size: 13px;
    margin-bottom: 15px;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: white;
    border: 2px dashed #52796F;
    border-radius: 12px;
}


/* ================= METRICS ================= */

[data-testid="stMetric"] {
    background-color: white;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

[data-testid="stMetricValue"] {
    color: #1B4332 !important;
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] {
    color: #5B6B63 !important;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #8A938E;
    font-size: 12px;
    padding-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="logo-box">🗑️</div>

    <div class="sidebar-title">
        Smart Waste Bin
    </div>

    <div class="sidebar-subtitle">
        AI Waste Monitoring System
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📷 Waste Detection",
            "♻️ Waste Classification",
            "📊 Statistics",
            "🗺️ Street Monitoring"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style="
        font-size:12px;
        color:#AAB5AF;
    ">
        SYSTEM STATUS
    </div>

    <div style="
        margin-top:8px;
        padding:10px;
        background:#22352C;
        border-radius:10px;
        font-size:13px;
    ">
        🟢 &nbsp; System Online
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    col1, col2 = st.columns([4, 1])

    with col1:

        st.markdown(
            '<div class="main-title">Smart Waste Dashboard</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="subtitle">'
            'AI-powered street waste detection and classification'
            '</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            '<div style="text-align:right;">'
            '<span class="online">● SYSTEM ONLINE</span>'
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # STATUS CARDS
    # =====================================================

    col1, col2, col3 = st.columns([1.5, 1, 1])


    # STREET STATUS
    with col1:

        st.html("""
        <div style="
            background: linear-gradient(135deg, #173D2A, #245B3F);
            color: white;
            padding: 28px;
            border-radius: 22px;
            min-height: 125px;
        ">

            <div style="
                font-size: 15px;
                opacity: 0.8;
                color: white;
            ">
                CURRENT STREET STATUS
            </div>

            <div style="
                font-size: 34px;
                font-weight: 800;
                margin-top: 10px;
                color: white;
            ">
                🟢 CLEAN
            </div>

            <div style="
                font-size: 14px;
                opacity: 0.8;
                margin-top: 8px;
                color: white;
            ">
                No waste detected on the monitored street.
            </div>

        </div>
        """)


    # WASTE DETECTED
    with col2:

        st.html("""
        <div style="
            background: white;
            padding: 22px;
            border-radius: 18px;
            border: 1px solid #E4E9E6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
            min-height: 125px;
        ">

            <div style="
                font-size: 27px;
                margin-bottom: 8px;
            ">
                🗑️
            </div>

            <div style="
                color: #68736D;
                font-size: 14px;
                font-weight: 600;
            ">
                Waste Detected
            </div>

            <div style="
                color: #17221D;
                font-size: 30px;
                font-weight: 800;
            ">
                0
            </div>

        </div>
        """)


    # BIN STATUS
    with col3:

        st.html("""
        <div style="
            background: white;
            padding: 22px;
            border-radius: 18px;
            border: 1px solid #E4E9E6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
            min-height: 125px;
        ">

            <div style="
                font-size: 27px;
                margin-bottom: 8px;
            ">
                📡
            </div>

            <div style="
                color: #68736D;
                font-size: 14px;
                font-weight: 600;
            ">
                Bin Status
            </div>

            <div style="
                color: #17221D;
                font-size: 30px;
                font-weight: 800;
            ">
                Online
            </div>

        </div>
        """)


    # =====================================================
    # WASTE OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">♻️ Waste Overview</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(5)

    waste_data = [
        ("🪟", "Glass", 0),
        ("🔩", "Metal", 0),
        ("📄", "Paper", 0),
        ("🧴", "Plastic", 0),
        ("🗑️", "Waste", 0)
    ]

    for col, (icon, name, number) in zip(cols, waste_data):

        with col:

            st.markdown(f"""
            <div class="waste-card">

                <div class="waste-icon">
                    {icon}
                </div>

                <div class="waste-name">
                    {name}
                </div>

                <div class="waste-number">
                    {number}
                </div>

            </div>
            """, unsafe_allow_html=True)


    # =====================================================
    # ACTIVITY
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Detection Activity</div>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Time": [
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00"
        ],
        "Detected Waste": [
            0,
            0,
            0,
            0,
            0,
            0
        ]
    })

    fig = px.area(
        chart_data,
        x="Time",
        y="Detected Waste"
    )

    fig.update_layout(
        height=280,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#17221D"),
        xaxis_title="",
        yaxis_title="Waste Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# WASTE DETECTION
# =========================================================

elif page == "📷 Waste Detection":

    st.markdown(
        '<div class="main-title">Waste Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload an image to detect whether waste is present.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2, 1])


    with col1:

        st.markdown("""
        <div class="upload-title">
            📷 Upload Street Image
        </div>

        <div class="upload-description">
            Upload an image captured by the smart bin camera.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file:

            st.image(
                uploaded_file,
                caption="Uploaded Street Image",
                use_container_width=True
            )


    with col2:

        st.markdown("""
        <div class="card" style="min-height:300px;">

            <div class="card-icon">
                🔍
            </div>

            <div class="card-title">
                DETECTION RESULT
            </div>

            <div class="card-value">
                —
            </div>

            <p style="color:#68736D;">
                Upload an image and click Detect Waste.
            </p>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("")


    if st.button(
        "🔍 Detect Waste",
        use_container_width=True
    ):

        if uploaded_file is None:

            st.warning(
                "Please upload an image first."
            )

        else:

            try:

                with st.spinner("Detecting waste..."):

                    image = Image.open(
                        uploaded_file
                    ).convert("RGB")

                    annotated_image, detections = detect_garbage(
                        image,
                        confidence=0.25
                    )


                st.success(
                    "Waste detection completed!"
                )

                st.subheader(
                    "Detection Result"
                )

                st.image(
                    annotated_image,
                    use_container_width=True
                )


                st.subheader(
                    "Detected Garbage"
                )


                if detections:

                    detection_data = pd.DataFrame(
                        detections
                    )

                    garbage_counts = (
                        detection_data["Garbage"]
                        .value_counts()
                    )

                    count_columns = st.columns(
                        min(
                            len(garbage_counts),
                            4
                        )
                    )

                    for index, (
                        item,
                        count
                    ) in enumerate(
                        garbage_counts.items()
                    ):

                        count_columns[
                            index % len(count_columns)
                        ].metric(
                            item,
                            int(count)
                        )


                    st.dataframe(
                        detection_data,
                        use_container_width=True,
                        hide_index=True
                    )


                    detected_names = list(
                        dict.fromkeys(
                            detection_data[
                                "Garbage"
                            ].tolist()
                        )
                    )


                    for item in detected_names:

                        description = (
                            GARBAGE_DESCRIPTIONS.get(
                                item.title()
                            )
                        )

                        if description:

                            st.write(
                                f"**{item}:** {description}"
                            )


                    st.download_button(
                        "⬇️ Download Annotated Image",
                        data=image_to_bytes(
                            annotated_image
                        ),
                        file_name="garbage_detection_result.jpg",
                        mime="image/jpeg"
                    )


                else:

                    st.warning(
                        "No garbage was detected. "
                        "Try lowering the confidence "
                        "or using a clearer image."
                    )


            except Exception as error:

                st.error(
                    f"The image could not be processed: {error}"
                )


# =========================================================
# WASTE CLASSIFICATION
# =========================================================

elif page == "♻️ Waste Classification":

    st.markdown(
        '<div class="main-title">Waste Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Classify detected waste into one of five categories.'
        '</div>',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload Waste Image",
        type=["jpg", "jpeg", "png"]
    )


    if uploaded_file:

        col1, col2 = st.columns(2)


        with col1:

            st.image(
                uploaded_file,
                caption="Waste Image",
                use_container_width=True
            )


        with col2:

            st.markdown("""
            <div class="card">

                <div class="card-icon">
                    ♻️
                </div>

                <div class="card-title">
                    PREDICTED CATEGORY
                </div>

                <div class="card-value">
                    —
                </div>

                <p style="color:#68736D;">
                    Classification result will appear here.
                </p>

            </div>
            """, unsafe_allow_html=True)

            st.progress(0)

            st.caption(
                "Confidence: —"
            )


        if st.button(
            "♻️ Classify Waste",
            use_container_width=True
        ):

            st.info(
                "Your YOLO model already detects and classifies "
                "the five waste categories."
            )


    st.markdown(
        '<div class="section-title">Supported Categories</div>',
        unsafe_allow_html=True
    )


    cols = st.columns(5)

    categories = [
        ("🪟", "Glass"),
        ("🔩", "Metal"),
        ("📄", "Paper"),
        ("🧴", "Plastic"),
        ("🗑️", "Waste")
    ]


    for col, (icon, name) in zip(
        cols,
        categories
    ):

        with col:

            st.markdown(f"""
            <div class="waste-card">

                <div class="waste-icon">
                    {icon}
                </div>

                <div class="waste-name">
                    {name}
                </div>

            </div>
            """, unsafe_allow_html=True)


# =========================================================
# STATISTICS
# =========================================================

elif page == "📊 Statistics":

    st.markdown(
        '<div class="main-title">Waste Statistics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Overview of detected waste categories.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)


    metrics = [
        ("🗑️", "Total Waste", "0"),
        ("📷", "Images Analyzed", "0"),
        ("♻️", "Categories", "5"),
        ("📡", "System Status", "Online")
    ]


    for col, (icon, title, value) in zip(
        [col1, col2, col3, col4],
        metrics
    ):

        with col:

            st.markdown(f"""
            <div class="card">

                <div class="card-icon">
                    {icon}
                </div>

                <div class="card-title">
                    {title}
                </div>

                <div class="card-value">
                    {value}
                </div>

            </div>
            """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">Waste Distribution</div>',
        unsafe_allow_html=True
    )


    stats = pd.DataFrame({
        "Category": [
            "Glass",
            "Metal",
            "Paper",
            "Plastic",
            "Waste"
        ],
        "Count": [
            0,
            0,
            0,
            0,
            0
        ]
    })


    fig = px.bar(
        stats,
        x="Category",
        y="Count",
        text="Count"
    )


    fig.update_layout(
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        xaxis_title="",
        yaxis_title="Number of Items"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# STREET MONITORING
# =========================================================

elif page == "🗺️ Street Monitoring":

    st.markdown(
        '<div class="main-title">Street Monitoring</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Monitor streets and identify areas that require cleaning.'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns([1.5, 1])


    with col1:

        st.html("""
        <div style="
            background: linear-gradient(135deg, #173D2A, #245B3F);
            color: white;
            padding: 28px;
            border-radius: 22px;
            min-height: 125px;
        ">

            <div style="
                font-size: 15px;
                opacity: 0.8;
                color: white;
            ">
                MONITORED AREA
            </div>

            <div style="
                font-size: 34px;
                font-weight: 800;
                margin-top: 10px;
                color: white;
            ">
                🟢 STREET CLEAN
            </div>

            <div style="
                font-size: 14px;
                opacity: 0.8;
                margin-top: 8px;
                color: white;
            ">
                No visible waste detected in the monitored area.
            </div>

        </div>
        """)


    with col2:

        st.html("""
        <div style="
            background: white;
            padding: 22px;
            border-radius: 18px;
            border: 1px solid #E4E9E6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
            min-height: 125px;
        ">

            <div style="
                font-size: 27px;
                margin-bottom: 8px;
            ">
                📍
            </div>

            <div style="
                color: #68736D;
                font-size: 14px;
                font-weight: 600;
            ">
                CURRENT LOCATION
            </div>

            <div style="
                color: #17221D;
                font-size: 22px;
                font-weight: 800;
            ">
                Street #01
            </div>

        </div>
        """)


    st.markdown(
        '<div class="section-title">📹 Live Camera</div>',
        unsafe_allow_html=True
    )


    st.info(
        "Your live camera / street detection function "
        "can be connected here."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Smart Waste Bin • AI-Powered Waste Management System
</div>
""", unsafe_allow_html=True)
