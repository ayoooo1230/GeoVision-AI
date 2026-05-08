import streamlit as st
import cv2
import torch
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
import time
import math
import requests
from io import BytesIO
import base64
# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="GeoVision AI",
    layout="wide"
)

# ==========================================
# CLASS NAMES
# ==========================================

class_names = {

    0: "Urban Land",
    1: "Agriculture Land",
    2: "Rangeland",
    3: "Forest Land",
    4: "Water",
    5: "Barren Land",
    6: "Unknown"
}

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */

[data-testid="stAppViewContainer"]{
    background: linear-gradient(
        135deg,
        #0B0F19,
        #111827,
        #1F2937
    );
    color:white;
}

/* Sidebar */

[data-testid="stSidebar"]{
    background:#111827;
}

/* Upload box */

[data-testid="stFileUploader"]{
    background:#1F2937;
    padding:20px;
    border-radius:15px;
}

/* Buttons */

.stButton>button{
    width:100%;
    border-radius:12px;
    background:#00E5FF;
    color:black;
    font-weight:bold;
    border:none;
    height:3em;
}

/* Metric Cards */

.metric-card{
    background:rgba(255,255,255,0.05);
    padding:20px;
    border-radius:20px;
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 0 20px rgba(0,229,255,0.15);
    margin-bottom:20px;
}

/* Titles */

h1,h2,h3,h4{
    color:white;
}

</style>
""", unsafe_allow_html=True)

video_file = open("earth.mp4", "rb")
video_bytes = video_file.read()
video_base64 = base64.b64encode(video_bytes).decode()


# ==========================================
# HERO SECTION
# ==========================================

# ==========================================
# HERO SECTION WITH VIDEO
# ==========================================

video_file = open("earth.mp4", "rb")
video_bytes = video_file.read()
video_base64 = base64.b64encode(video_bytes).decode()

hero_html = f"""
<style>

.video-container {{
    position: relative;
    width: 100%;
    height: 420px;
    overflow: hidden;
    border-radius: 25px;
    margin-bottom: 30px;
}}

.video-container video {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}}

.video-overlay {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.45);

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    text-align: center;
    color: white;
}}

</style>

<div class="video-container">

<video autoplay muted loop playsinline>
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
</video>

<div class="video-overlay">

<h1 style="
font-size:65px;
margin-bottom:10px;
font-weight:bold;
">
  Ge🌍Vision AI
</h1>

<h3 style="
color:#E5E7EB;
font-weight:400;
">
AI-Based Geological Feature Extraction from Satellite Imagery
</h3>

<p style="
color:#D1D5DB;
font-size:20px;
margin-top:20px;
">
Deep Learning • Satellite Intelligence • Geological Segmentation
</p>

</div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)
# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = smp.Unet(

        encoder_name="efficientnet-b0",

        encoder_weights=None,

        in_channels=3,

        classes=7
    )

    model.load_state_dict(

        torch.load(
            "geovision_unet.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

# ==========================================
# CLASS COLORS
# ==========================================

colors = np.array([

    [0,255,255],    # Urban → Cyan

    [255,255,0],    # Agriculture → Yellow

    [255,0,255],    # Rangeland → Magenta

    [0,255,0],      # Forest → Green

    [0,0,255],      # Water → Blue

    [255,255,255],  # Barren → White

    [0,0,0]         # Unknown → Black

], dtype=np.uint8)
# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_image(image):

    image = image.convert("RGB")

    image = np.array(image)

    original = image.copy()

    resized = cv2.resize(
        image,
        (256,256)
    )

    normalized = resized / 255.0

    tensor = torch.tensor(
        normalized
    ).permute(2,0,1).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = model(tensor)

        prediction = torch.argmax(
            prediction,
            dim=1
        )

        prediction = prediction.squeeze().cpu().numpy()

    segmented = colors[prediction]

    segmented = segmented.astype(np.uint8)


    overlay = cv2.addWeighted(

    resized,
    0.7,

    segmented,
    0.3,

    0
)

    return original, segmented, overlay, prediction
# ==========================================
# LAT/LON → TILE CONVERSION
# ==========================================

def deg2num(lat_deg, lon_deg, zoom):

    lat_rad = math.radians(lat_deg)

    n = 2.0 ** zoom

    xtile = int((lon_deg + 180.0) / 360.0 * n)

    ytile = int(
        (1.0 - math.log(
            math.tan(lat_rad) +
            (1 / math.cos(lat_rad))
        ) / math.pi) / 2.0 * n
    )

    return xtile, ytile

# ==========================================
# REPORT FUNCTION
# ==========================================

def generate_report(prediction):

    st.subheader("Geological Classification Report")

    total_pixels = prediction.size

    unique, counts = np.unique(
        prediction,
        return_counts=True
    )

    percentages = {}

    for cls, count in zip(unique, counts):

        percent = (count / total_pixels) * 100

        percentages[class_names[cls]] = percent

        st.markdown(f"""
<div class="metric-card">

<h3>{class_names[cls]}</h3>

<h1 style="
color:#00E5FF;
">
{percent:.2f}%
</h1>

<p>
Geological Coverage Detected
</p>

</div>
""", unsafe_allow_html=True)

        st.progress(int(percent))

        st.divider()

    # --------------------------------------
    # PIE CHART
    # --------------------------------------

    fig, ax = plt.subplots(
        figsize=(7,7),
        facecolor='#111827'
    )

    pie_colors = []

    for cls in unique:

        pie_colors.append(
            colors[cls] / 255.0
        )

    ax.pie(

        counts,

        labels=[
            class_names[c]
            for c in unique
        ],

        autopct='%1.1f%%',

        colors=pie_colors,

        startangle=90,

        wedgeprops={
            'linewidth':2,
            'edgecolor':'white'
        },

        textprops={
            'color':'white',
            'fontsize':14
        }
    )

    ax.set_title(

        "Geological Land Distribution",

        fontsize=24,

        color='white',

        pad=30
    )

    fig.patch.set_facecolor('#111827')

    ax.set_facecolor('#111827')

    st.pyplot(fig)

    return percentages


# ==========================================
# REGION SELECTION MAP
# ==========================================

st.subheader("Select Region for AI Analysis")

m = folium.Map(

    location=[20.5937, 78.9629],

    zoom_start=5,

    tiles='CartoDB dark_matter'
)

# Satellite Layer

folium.TileLayer(

    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',

    attr='Esri Satellite',

    name='Satellite',

    overlay=False,

    control=True

).add_to(m)

# Enable Layer Control

folium.LayerControl().add_to(m)

# Display Map

map_data = st_folium(

    m,

    width=1200,

    height=500
)

# ==========================================
# COORDINATE EXTRACTION
# ==========================================

if map_data and map_data.get("last_clicked"):

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(
        f"📍 Selected Coordinates: Latitude {lat:.4f}, Longitude {lon:.4f}"
    )

    

    # --------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------
    st.markdown("""
        <style>
        div.stButton > button:first-child {

     background: white;

    color: black;

    font-size: 18px;

    height: 3.2em;

        border: none;
        border-radius: 18px;

     box-shadow:
        0 0 20px rgba(0,229,255,0.5),
        0 0 40px rgba(0,229,255,0.25);

        transition: all 0.3s ease;
    }

    div.stButton > button:first-child:hover {

    transform: scale(1.03);

    background: white;

    color: black;

    box-shadow:
        0 0 25px rgba(255,255,255,0.8);
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Analyze Selected Region"):

        st.success(
            "GeoVision AI analysis started..."
        )

        # --------------------------------------
        # LOADING BAR
        # --------------------------------------

        progress_bar = st.progress(0)

        status_text = st.empty()

        process_steps = [

            "Fetching satellite region...",
            "Preprocessing imagery...",
            "Running GeoVision AI model...",
            "Detecting geological features...",
            "Generating segmentation report...",
            "Finalizing analysis..."
        ]

        for i, step in enumerate(process_steps):

            status_text.info(step)

            progress_bar.progress(
                int((i + 1) / len(process_steps) * 100)
            )

            time.sleep(1)

        # --------------------------------------
        # FINAL STATUS
        # --------------------------------------

        st.success(
            "✅ Region analysis completed successfully!"
        )

        st.toast("✅ GeoVision AI Analysis Completed Successfully!")
    
        # ==========================================
        # FETCH REAL SATELLITE TILE
        # ==========================================

        zoom = 15

        x, y = deg2num(lat, lon, zoom)

        tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"

        
        response = requests.get(tile_url, timeout=10)
        if response.status_code != 200:
            st.error("Failed to fetch satellite imagery.")
            st.stop()

        satellite_tile = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        # --------------------------------------
        # RUN GEOVISION AI
        # --------------------------------------

        original, segmented, overlay, prediction = predict_image(
            satellite_tile
        )

        
# --------------------------------------
# REAL AI INSIGHTS
# --------------------------------------
        
# --------------------------------------
# AI INSIGHTS
# --------------------------------------
       
# --------------------------------------
# DISPLAY RESULTS
# --------------------------------------

        st.subheader(" Live Satellite Region Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                original,
                caption="Satellite Tile",
                use_container_width=True
            )

        with col2:

            st.image(
                segmented,
                caption="AI Segmentation",
                use_container_width=True
            )

        with col3:

            st.image(
                overlay,
                caption="Overlay Analysis",
                 use_container_width=True
            )    
        # --------------------------------------
        # --------------------------------------
# AI INSIGHTS report
# --------------------------------------
  
        generate_report(prediction)
# ==========================================
# SIDEBAR DASHBOARD
# ==========================================

with st.sidebar:

    # --------------------------------------
    # PROJECT TITLE
    # --------------------------------------

    st.markdown("""
    <h1 style='
    text-align:center;
    color:#00E5FF;
    '>
     Ge🌍Vision AI
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------
    # SYSTEM STATUS
    # --------------------------------------

    st.markdown("##  System Status")

    st.success("AI Model Loaded")

    st.info("U-Net Segmentation Active")

    st.success("Satellite Analysis Ready")

    st.markdown("---")

    # --------------------------------------
    # DOWNLOAD RESEARCH PAPER
    # --------------------------------------

    st.markdown("## 📄 Research Paper")

    with open("journal.pdf", "rb") as pdf_file:

        PDFbyte = pdf_file.read()

    st.download_button(
        label="⬇ Download Journal PDF",
        data=PDFbyte,
        file_name="GeoVision_Research.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    st.markdown("---")

    # --------------------------------------
    # GEOLOGICAL CLASSES
    # --------------------------------------

    st.markdown("##  Geological Classes")

    for idx, name in class_names.items():

        st.markdown(f"""
        <div style="
        background:rgba(255,255,255,0.05);
        padding:10px;
        border-radius:10px;
        margin-bottom:8px;
        border:1px solid rgba(255,255,255,0.08);
        ">
        ◉ {name}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------------------
    # FOOTER
    # --------------------------------------

    st.markdown("""
    <div style='
    text-align:center;
    color:gray;
    font-size:14px;
    '>

    GeoVision v1.0<br>
    AI Geological Intelligence Platform

    </div>
    """, unsafe_allow_html=True)



# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "png", "jpeg"]
)

# ==========================================
# PROCESS IMAGE
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    original, segmented, overlay, prediction = predict_image(image)

    # --------------------------------------
    # IMAGE DISPLAY
    # --------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            original,
            caption="Original Image",
            use_container_width=True
        )

    with col2:

        st.image(
            segmented,
            caption="AI Segmentation",
            use_container_width=True
        )

    with col3:

        st.image(
            overlay,
            caption="Overlay Output",
            use_container_width=True
        )

         # --------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------

    generate_report(prediction)
    # --------------------------------------
    # MODERN PIE CHART
    # --------------------------------------
