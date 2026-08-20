import streamlit as st
import pandas as pd
import plotly.express as px

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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- Main Background ---------- */
    .stApp {
        background-color: #F5F7F6;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #17221D;
        padding-top: 25px;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    /* ---------- Hide Streamlit Default ---------- */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* ---------- Main Container ---------- */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Header ---------- */
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

    /* ---------- Logo ---------- */
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

    /* ---------- Cards ---------- */
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

    /* ---------- Status Card ---------- */
    .status-card {
        background: linear-gradient(135deg, #173D2A, #245B3F);
        color: white;
        padding: 28px;
        border-radius: 22px;
        min-height: 180px;
    }

    .status-title {
        font-size: 15px;
        opacity: 0.8;
    }

    .status-value {
        font-size: 34px;
        font-weight: 800;
        margin-top: 10px;
    }

    .status-description {
        font-size: 14px;
        opacity: 0.8;
        margin-top: 8px;
    }

    /* ---------- Online Indicator ---------- */
    .online {
        display: inline-block;
        background-color: #D9F7E5;
        color: #1D7A46;
        padding: 6px 13px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 700;
    }

    /* ---------- Section Title ---------- */
    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #17221D;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* ---------- Classification Cards ---------- */
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

    /* ---------- Upload Box ---------- */
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

    /* ---------- Footer ---------- */
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
    <div style="font-size:12px;color:#AAB5AF;">
        SYSTEM STATUS
    </div>

    <div style="
        margin-top:8px;
        padding:10px;
        background:#22352C;
        border-radius:10px;
        font-size:13px;">
        🟢 &nbsp; System Online
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    # Header
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
    # STATUS + QUICK INFO
    # =====================================================

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:

        st.markdown("""
        <div class="status-card">

            <div class="status-title">
                CURRENT STREET STATUS
            </div>

            <div class="status-value">
                🟢 CLEAN
            </div>

            <div class="status-description">
                No waste detected on the monitored street.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

            <div class="card-icon">🗑️</div>

            <div class="card-title">
                Waste Detected
            </div>

            <div class="card-value">
                0
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card">

            <div class="card-icon">📡</div>

            <div class="card-title">
                Bin Status
            </div>

            <div class="card-value">
                Online
            </div>

        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # WASTE TYPES
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
        margin=dict(l=10, r=10, t=10, b=10),
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
                Your detection function will appear here.
            </p>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if st.button(
            "🔍 Detect Waste",
            use_container_width=True
        ):

            # =================================================
            # ADD YOUR DETECTION FUNCTION HERE
            # =================================================

            st.info(
                "Connect your waste detection model here."
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

            st.caption("Confidence: —")

        if st.button(
            "♻️ Classify Waste",
            use_container_width=True
        ):

            # =================================================
            # ADD YOUR CLASSIFICATION FUNCTION HERE
            # =================================================

            st.info(
                "Connect your classification model here."
            )

    # Categories

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

    for col, (icon, name) in zip(cols, categories):

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

    # =====================================================
    # METRICS
    # =====================================================

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

    # =====================================================
    # CHART
    # =====================================================

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
        margin=dict(l=20, r=20, t=20, b=20),
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

    # Street status

    col1, col2 = st.columns([1.5, 1])

    with col1:

        st.markdown("""
        <div class="status-card">

            <div class="status-title">
                MONITORED AREA
            </div>

            <div class="status-value">
                🟢 STREET CLEAN
            </div>

            <div class="status-description">
                No visible waste detected in the monitored area.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">

            <div class="card-icon">
                📍
            </div>

            <div class="card-title">
                CURRENT LOCATION
            </div>

            <div class="card-value"
                 style="font-size:22px;">
                Street #01
            </div>

        </div>
        """, unsafe_allow_html=True)

    # Camera area

    st.markdown(
        '<div class="section-title">📹 Live Camera</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Your live camera / street detection function can be connected here."
    )

    # =====================================================
    # ADD CAMERA FUNCTION HERE
    # =====================================================

    # Example:
    #
    # frame = your_camera_function()
    # st.image(frame)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Smart Waste Bin • AI-Powered Waste Management System
</div>
""", unsafe_allow_html=True)
