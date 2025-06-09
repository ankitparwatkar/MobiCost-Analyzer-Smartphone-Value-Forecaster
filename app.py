import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import base64
from streamlit_extras.stylable_container import stylable_container
from PIL import Image
from io import BytesIO  # Added for image handling

# App configuration 
st.set_page_config(
    page_title="MobiCost Analyzer",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('phone_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Function to load images with error handling
def load_image(image_path):
    try:
        if image_path.startswith("http"):
            return image_path
        
        # Get absolute path to image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, image_path)
        
        if os.path.exists(full_path):
            # Read image and convert to base64
            with open(full_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()
            return f"data:image/png;base64,{encoded_string}"
        else:
            # Generate a placeholder image
            img = Image.new('RGB', (200, 200), color=(45, 60, 80))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            encoded_string = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        st.error(f"Error loading image: {e}")
        # Generate placeholder on error
        img = Image.new('RGB', (200, 200), color=(45, 60, 80))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{encoded_string}"

# Enhanced CSS with improved visibility
st.markdown("""
<style>
    /* Modern dark theme with improved contrast */
    .stApp {
        background: linear-gradient(135deg, #0d1526 0%, #1a2439 100%);
        color: #f0f4f8;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        line-height: 1.6;
    }
    
    /* Animated header with larger text */
    .dynamic-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(25, 35, 55, 0.8);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(70, 130, 180, 0.5);
        box-shadow: 0 10px 35px rgba(2, 8, 32, 0.5);
        transition: all 0.3s ease;
    }
    
    .dynamic-header:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(2, 8, 32, 0.7);
    }
    
    .dynamic-header h1 {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(45deg, #4da6ff, #1a8cff);
        -webkit-background-clip: text;
        margin-bottom: 0.7rem;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .dynamic-header p {
        font-size: 1.35rem;
        color: #d1e0f0;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.7;
    }
    
    /* Modern cards with enhanced visibility */
    .feature-card {
        background: rgba(30, 45, 65, 0.9);
        border-radius: 18px;
        padding: 1.8rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(70, 130, 180, 0.4);
        box-shadow: 0 6px 25px rgba(2, 8, 32, 0.25);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(2, 8, 32, 0.4);
    }
    
    .feature-card h3 {
        background: linear-gradient(45deg, #4da6ff, #1a8cff);
        -webkit-background-clip: text;
        font-size: 1.55rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 700;
    }
    
    /* Enhanced slider styling */
    .stSlider .thumb {
        background: linear-gradient(45deg, #4da6ff, #1a8cff) !important;
        border: 2px solid white !important;
        box-shadow: 0 0 12px rgba(77, 166, 255, 0.7) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    .stSlider .track {
        background: rgba(77, 166, 255, 0.3) !important;
        height: 8px !important;
    }
    
    .stSlider .st-ae {
        color: white !important;
        font-size: 1.1rem !important;
    }
    
    /* Modern button styling with larger text */
    .stButton>button {
        background: linear-gradient(45deg, #4da6ff, #1a8cff);
        color: white !important;
        border-radius: 14px;
        padding: 16px 32px;
        font-weight: 700;
        font-size: 1.25rem;
        border: none;
        width: 100%;
        box-shadow: 0 5px 18px rgba(77, 166, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(77, 166, 255, 0.6);
    }
    
    /* Animated prediction card with larger text */
    .result-card {
        background: rgba(25, 40, 60, 0.95);
        border-radius: 24px;
        padding: 3rem;
        margin: 2.5rem 0;
        border: 2px solid rgba(70, 130, 180, 0.5);
        text-align: center;
        box-shadow: 0 10px 40px rgba(2, 8, 32, 0.4);
        animation: fadeIn 0.8s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 1.8rem;
        color: #4da6ff;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .result-value {
        font-size: 4.5rem;
        font-weight: 900;
        margin: 2rem 0;
        background: linear-gradient(45deg, #4da6ff, #1a8cff);
        -webkit-background-clip: text;
        animation: pulse 2s infinite;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); }
        50% { transform: scale(1.08); text-shadow: 0 6px 12px rgba(0, 0, 0, 0.4); }
        100% { transform: scale(1); text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); }
    }
    
    /* Enhanced model cards with larger text */
    .models-section {
        background: rgba(30, 45, 65, 0.9);
        border-radius: 24px;
        padding: 2.2rem;
        margin: 2.5rem 0;
        border: 1px solid rgba(70, 130, 180, 0.4);
        box-shadow: 0 8px 30px rgba(2, 8, 32, 0.3);
    }
    
    .model-card {
        background: rgba(20, 30, 50, 0.9);
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        height: 100%;
        border: 1px solid rgba(70, 130, 180, 0.3);
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-7px);
        box-shadow: 0 10px 30px rgba(2, 8, 32, 0.4);
        border-color: rgba(70, 130, 180, 0.6);
    }
    
    .model-name {
        font-weight: 800;
        font-size: 1.4rem;
        margin: 1.5rem 0 0.8rem;
        color: #4da6ff;
    }
    
    .model-price {
        color: #e6f0ff;
        font-weight: 700;
        font-size: 1.35rem;
    }
    
    /* Modern tab styling with larger text */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4da6ff, #1a8cff) !important;
        color: white !important;
        font-weight: 800;
        font-size: 1.2rem;
        border-radius: 16px !important;
        box-shadow: 0 5px 15px rgba(77, 166, 255, 0.4);
        padding: 1rem !important;
    }
    
    .stTabs [aria-selected="false"] {
        background: rgba(30, 45, 65, 0.9) !important;
        color: #d1e0f0 !important;
        font-size: 1.1rem;
        border-radius: 16px !important;
        transition: all 0.3s ease;
        padding: 1rem !important;
    }
    
    .stTabs [aria-selected="false"]:hover {
        background: rgba(35, 50, 70, 1) !important;
    }
    
    /* Footer styling with larger text */
    .footer {
        text-align: center;
        padding: 2.2rem 0;
        color: #a8c6e0;
        font-size: 1.15rem;
        margin-top: 3.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        gap: 30px;
        margin-top: 25px;
    }
    
    .social-links a {
        color: #4da6ff;
        font-size: 1.75rem;
        transition: all 0.3s;
        background: rgba(25, 40, 60, 0.7);
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: 5px solid rgba(77, 166, 255, 0.4);
    }
    
    .social-links a:hover {
        color: #1a8cff;
        transform: translateY(-5px);
        background: rgba(35, 50, 70, 0.9);
        box-shadow: 0 5px 15px rgba(77, 166, 255, 0.3);
    }
    
    /* Feature labels with larger text */
    .feature-label {
        font-weight: 700;
        margin-bottom: 1rem;
        color: #e6f0ff;
        font-size: 1.25rem;
        letter-spacing: 0.5px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Section titles with larger text */
    .section-title {
        font-size: 2rem;
        font-weight: 900;
        margin: 2.5rem 0 1.5rem;
        background: linear-gradient(45deg, #4da6ff, #1a8cff);
        -webkit-background-clip: text;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid rgba(77, 166, 255, 0.4);
        text-align: center;
    }
    
    /* Tooltip styling */
    .stTooltip {
        background: rgba(25, 40, 60, 0.95) !important;
        border: 1px solid rgba(77, 166, 255, 0.5) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 35px rgba(2, 8, 32, 0.4) !important;
        font-size: 1.1rem !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4da6ff, #1a8cff) !important;
    }
    
    /* Improved spacing for all elements */
    .stSlider {
        margin-bottom: 1.8rem !important;
    }
    
    .stCheckbox, .stRadio {
        margin-bottom: 1.2rem !important;
    }
    
    /* Better contrast for text elements */
    .stMarkdown p, .stMarkdown li {
        color: #e6f0ff !important;
        font-size: 1.15rem;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #e6f0ff !important;
    }
    
    /* Enhanced gauge chart styling */
    .gauge-title {
        font-size: 1.4rem !important;
        fill: #e6f0ff !important;
    }
    
    /* Improved radial chart text */
    .polar .r-axistext, .polar .theta-axistext {
        font-size: 1.2rem !important;
        fill: #e6f0ff !important;
    }
    
    /* New: Image placeholder styling */
    .image-placeholder {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 180px;
        background: rgba(30, 45, 65, 0.6);
        border-radius: 12px;
        color: #a8c6e0;
        font-size: 1.1rem;
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Price range mapping
PRICE_RANGES = {
    0: "Budget",
    1: "Mid-Range",
    2: "Premium",
    3: "Luxury"
}

PRICE_COLORS = {
    0: "#10b981",  # Emerald
    1: "#f59e0b",  # Amber
    2: "#ef4444",  # Red
    3: "#8b5cf6"   # Violet
}

PRICE_ICONS = {
    0: "💰",
    1: "💸",
    2: "💳",
    3: "🏦"
}

# Popular phone models with real image URLs
POPULAR_MODELS = {
    0: [
        {"name": "Samsung Galaxy A14", "price": "$180", "image": "images/galaxya14.png"},
        {"name": "Xiaomi Redmi 12", "price": "$160", "image": "images/redmi12.png"},
        {"name": "Nokia G42", "price": "$200", "image": "images/g42.png"},
        {"name": "Motorola Moto G14", "price": "$150", "image": "images/motorolag14.png"},
        {"name": "Realme C55", "price": "$170", "image": "images/realmeC55.png"}
    ],
    1: [
        {"name": "Google Pixel 7a", "price": "$499", "image": "images/pixel7a.png"},
        {"name": "Samsung Galaxy A54", "price": "$449", "image": "images/galaxya54.png"},
        {"name": "iPhone SE (2022)", "price": "$429", "image": "images/iphonese.png"},
        {"name": "OnePlus Nord 3", "price": "$479", "image": "images/nord3.png"},
        {"name": "Xiaomi Poco F5", "price": "$399", "image": "images/pocof5.png"}
    ],
    2: [
        {"name": "iPhone 15", "price": "$799", "image": "images/iphone15.png"},
        {"name": "Samsung Galaxy S23", "price": "$799", "image": "images/galaxys23.png"},
        {"name": "Google Pixel 8", "price": "$699", "image": "images/pixel8.png"},
        {"name": "OnePlus 11", "price": "$699", "image": "images/oneplus11.png"},
        {"name": "Xiaomi 13", "price": "$749", "image": "images/xiaomi13.png"}
    ],
    3: [
        {"name": "iPhone 15 Pro Max", "price": "$1,199", "image": "images/iphone15promax.png"},
        {"name": "Samsung Galaxy S23 Ultra", "price": "$1,199", "image": "images/galaxys23ultra.png"},
        {"name": "Google Pixel Fold", "price": "$1,799", "image": "images/pixelfold.png"},
        {"name": "Samsung Galaxy Z Fold5", "price": "$1,799", "image": "images/zfold5.png"},
        {"name": "Huawei Mate X3", "price": "$1,999", "image": "images/matex3.png"}
    ]
}

# Feature list used in the trained model
FEATURE_LIST = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
    'int_memory', 'mobile_wt', 'n_cores', 'pc', 'ram', 'sc_h', 'sc_w',
    'talk_time', 'three_g', 'touch_screen', 'wifi', 'pixel_density',
    'screen_area', 'camera_total'
]

NUMERICAL_FEATURES = ['battery_power', 'ram', 'pixel_density', 'screen_area', 'int_memory', 'camera_total']

# Feature importance from trained model
FEATURE_IMPORTANCE = {
    'RAM': 0.348145,
    'Battery': 0.097577,
    'Screen Quality': 0.082037,
    'Internal Memory': 0.036340,
    'Camera System': 0.034207,
    'Processor': 0.028100
}

# Function to generate realistic values based on price range
def generate_realistic_values(price_range):
    if price_range == 0:  # Budget
        return {
            'battery_power': np.random.randint(3000, 4500),
            'ram': np.random.randint(2000, 4000),
            'int_memory': np.random.randint(32, 128),
            'pc': np.random.randint(8, 16),
            'fc': np.random.randint(5, 10),
            'clock_speed': round(np.random.uniform(1.8, 2.2), 1),
            'n_cores': np.random.randint(4, 6),
            'px_height': np.random.randint(720, 1080),
            'px_width': np.random.randint(1280, 1920),
            'sc_h': np.random.randint(12, 15),
            'sc_w': np.random.randint(6, 8),
            'mobile_wt': np.random.randint(180, 220),
            'talk_time': np.random.randint(10, 15)
        }
    elif price_range == 1:  # Mid-Range
        return {
            'battery_power': np.random.randint(4000, 5000),
            'ram': np.random.randint(4000, 6000),
            'int_memory': np.random.randint(128, 256),
            'pc': np.random.randint(12, 20),
            'fc': np.random.randint(8, 16),
            'clock_speed': round(np.random.uniform(2.2, 2.6), 1),
            'n_cores': np.random.randint(6, 8),
            'px_height': np.random.randint(1080, 1440),
            'px_width': np.random.randint(1920, 2560),
            'sc_h': np.random.randint(14, 16),
            'sc_w': np.random.randint(7, 8),
            'mobile_wt': np.random.randint(160, 190),
            'talk_time': np.random.randint(15, 20)
        }
    elif price_range == 2:  # Premium
        return {
            'battery_power': np.random.randint(4500, 5500),
            'ram': np.random.randint(6000, 8000),
            'int_memory': np.random.randint(256, 512),
            'pc': np.random.randint(20, 40),
            'fc': np.random.randint(12, 20),
            'clock_speed': round(np.random.uniform(2.6, 3.0), 1),
            'n_cores': np.random.randint(8, 10),
            'px_height': np.random.randint(1440, 1800),
            'px_width': np.random.randint(2560, 3200),
            'sc_h': np.random.randint(15, 17),
            'sc_w': np.random.randint(7, 9),
            'mobile_wt': np.random.randint(150, 180),
            'talk_time': np.random.randint(18, 24)
        }
    else:  # Luxury
        return {
            'battery_power': np.random.randint(5000, 7000),
            'ram': np.random.randint(8000, 12000),
            'int_memory': np.random.randint(512, 1024),
            'pc': np.random.randint(40, 100),
            'fc': np.random.randint(20, 40),
            'clock_speed': round(np.random.uniform(3.0, 3.5), 1),
            'n_cores': np.random.randint(10, 16),
            'px_height': np.random.randint(1800, 2400),
            'px_width': np.random.randint(3200, 3840),
            'sc_h': np.random.randint(16, 19),
            'sc_w': np.random.randint(8, 10),
            'mobile_wt': np.random.randint(180, 250),
            'talk_time': np.random.randint(20, 30)
        }

# Main app
def main():
    # Clean header with animation
    with st.container():
        st.markdown("""
        <div class="dynamic-header">
            <h1>📱 MobiCost Analyzer: Smartphone Value Forecaster </h1>
            <p>Enter specifications to predict phone price range and discover popular models in that category</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start buttons
    st.markdown('<div class="section-title">QUICK START</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("💰 Budget Phone", use_container_width=True, help="Generate realistic specs for a budget phone"):
            values = generate_realistic_values(0)
            st.session_state.update(values)
    with col2:
        if st.button("💸 Mid-Range Phone", use_container_width=True, help="Generate realistic specs for a mid-range phone"):
            values = generate_realistic_values(1)
            st.session_state.update(values)
    with col3:
        if st.button("💳 Premium Phone", use_container_width=True, help="Generate realistic specs for a premium phone"):
            values = generate_realistic_values(2)
            st.session_state.update(values)
    with col4:
        if st.button("🏦 Luxury Phone", use_container_width=True, help="Generate realistic specs for a luxury phone"):
            values = generate_realistic_values(3)
            st.session_state.update(values)
    
    # Create input form using tabs for better organization
    with st.container():
        tab1, tab2, tab3 = st.tabs(["🔋 Battery & Memory", "📸 Display & Camera", "⚙️ Connectivity & Features"])
        
        with tab1:
            st.markdown('<div class="feature-label">BATTERY & POWER</div>', unsafe_allow_html=True)
            battery_power = st.slider("Battery Power (mAh)", 500, 7000, 
                                      st.session_state.get('battery_power', 3500), 
                                      help="Total energy storage capacity")
            
            st.markdown('<div class="feature-label" style="margin-top: 1.8rem;">MEMORY</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                int_memory = st.slider("Internal Memory (GB)", 2, 1024, 
                                       st.session_state.get('int_memory', 128), 
                                       help="Storage capacity for apps and files")
            with col2:
                ram = st.slider("RAM (MB)", 500, 16000, 
                                st.session_state.get('ram', 4000), 
                                help="Memory for active applications")
        
        with tab2:
            st.markdown('<div class="feature-label">DISPLAY RESOLUTION</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                px_height = st.slider("Pixel Height", 0, 3000, 
                                      st.session_state.get('px_height', 1440), 
                                      help="Vertical screen resolution")
            with col2:
                px_width = st.slider("Pixel Width", 0, 4000, 
                                     st.session_state.get('px_width', 2560), 
                                     help="Horizontal screen resolution")
            
            st.markdown('<div class="feature-label" style="margin-top: 1.8rem;">SCREEN SIZE</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                sc_h = st.slider("Screen Height (cm)", 5, 25, 
                                 st.session_state.get('sc_h', 15), 
                                 help="Physical screen height")
            with col2:
                sc_w = st.slider("Screen Width (cm)", 5, 15, 
                                 st.session_state.get('sc_w', 8), 
                                 help="Physical screen width")
            
            st.markdown('<div class="feature-label" style="margin-top: 1.8rem;">CAMERA SYSTEM</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                pc = st.slider("Primary Camera (MP)", 0, 200, 
                               st.session_state.get('pc', 48), 
                               help="Main camera resolution")
            with col2:
                fc = st.slider("Front Camera (MP)", 0, 100, 
                               st.session_state.get('fc', 16), 
                               help="Selfie camera resolution")
        
        with tab3:
            st.markdown('<div class="feature-label">PERFORMANCE</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                clock_speed = st.slider("Clock Speed (GHz)", 0.5, 5.0, 
                                        st.session_state.get('clock_speed', 2.5), 
                                        step=0.1, format="%.1f",
                                        help="Processor speed")
            with col2:
                n_cores = st.slider("Processor Cores", 1, 16, 
                                    st.session_state.get('n_cores', 8), 
                                    help="Number of CPU cores")
            
            st.markdown('<div class="feature-label" style="margin-top: 1.8rem;">PHYSICAL ATTRIBUTES</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                mobile_wt = st.slider("Weight (grams)", 80, 300, 
                                      st.session_state.get('mobile_wt', 180), 
                                      help="Device weight")
            with col2:
                talk_time = st.slider("Talk Time (hours)", 2, 40, 
                                      st.session_state.get('talk_time', 18), 
                                      help="Battery life during calls")
            
            st.markdown('<div class="feature-label" style="margin-top: 1.8rem;">CONNECTIVITY</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                blue = st.checkbox("Bluetooth", value=st.session_state.get('blue', True))
                four_g = st.checkbox("4G", value=st.session_state.get('four_g', True))
            with col2:
                dual_sim = st.checkbox("Dual SIM", value=st.session_state.get('dual_sim', True))
                three_g = st.checkbox("3G", value=st.session_state.get('three_g', True))
            with col3:
                wifi = st.checkbox("WiFi", value=st.session_state.get('wifi', True))
                touch_screen = st.checkbox("Touch Screen", value=st.session_state.get('touch_screen', True))
    
    # Prediction button
    st.markdown("---")
    predict_btn = st.button("**PREDICT PRICE RANGE**", use_container_width=True, type="primary")
    
    # Placeholder for results
    result_placeholder = st.empty()
    
    if predict_btn:
        # Show loading animation with progress bar
        with st.spinner("🔍 Analyzing phone specifications..."):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Collect inputs
            features = {
                'battery_power': battery_power,
                'blue': int(blue),
                'clock_speed': clock_speed,
                'dual_sim': int(dual_sim),
                'fc': fc,
                'four_g': int(four_g),
                'int_memory': int_memory,
                'mobile_wt': mobile_wt,
                'n_cores': n_cores,
                'pc': pc,
                'ram': ram,
                'sc_h': sc_h,
                'sc_w': sc_w,
                'talk_time': talk_time,
                'three_g': int(three_g),
                'touch_screen': int(touch_screen),
                'wifi': int(wifi),
                'px_height': px_height,
                'px_width': px_width
            }
            
            # Feature engineering
            features['pixel_density'] = features['px_width'] * features['px_height']
            features['screen_area'] = features['sc_w'] * features['sc_h']
            features['camera_total'] = features['pc'] + features['fc']
            
            # Create input DataFrame with correct feature order
            input_df = pd.DataFrame([features])[FEATURE_LIST]
            
            # Scale numerical features
            input_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            price_range = PRICE_RANGES[prediction]
            color = PRICE_COLORS[prediction]
            icon = PRICE_ICONS[prediction]
            
            # Store values in session state
            st.session_state.update(features)
            
            # Display results with animation
            with result_placeholder.container():
                st.markdown("---")
                
                # Prediction card with animation
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">PREDICTED PRICE RANGE</div>
                    <div class="result-value">{icon} {price_range}</div>
                    <div style="font-size: 1.4rem; color: #d1e0f0; margin-top: 1.2rem;">
                        Confidence: {probabilities[prediction]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = probabilities[prediction] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediction Confidence", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickfont': {'size': 16}},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(30, 45, 65, 0.8)"},
                            {'range': [50, 100], 'color': "rgba(30, 45, 65, 0.6)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': probabilities[prediction] * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    height=350,
                    margin=dict(l=0, r=0, t=80, b=0),
                    font=dict(color="white", size=18),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Popular models section
                st.markdown('<div class="section-title">POPULAR MODELS IN THIS RANGE</div>', unsafe_allow_html=True)
                st.markdown("""
                <div class="models-section">
                    <div style="text-align: center; margin-bottom: 1.5rem; color: #d1e0f0; font-size: 1.25rem;">
                        Based on your predicted price range
                    </div>
                """, unsafe_allow_html=True)
                
            models = POPULAR_MODELS.get(prediction, [])
            cols = st.columns(5)
            for idx, model_info in enumerate(models):
                with cols[idx % 5]:
                    # Load image with error handling
                    img_path = load_image(model_info['image'])
                    st.markdown(f"""
                    <div class="model-card">
                        <div style="height: 180px; display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                            <img src="{img_path}" alt="{model_info['name']}" style="max-height: 160px; max-width: 100%; object-fit: contain;" />
                        </div>
                        <div class="model-name">{model_info['name']}</div>
                        <div class="model-price">{model_info['price']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Feature impact visualization
            st.markdown('<div class="section-title">FEATURE IMPACT ANALYSIS</div>', unsafe_allow_html=True)
            
            # Create a DataFrame for feature importances
            feature_impact = pd.DataFrame({
                'Feature': list(FEATURE_IMPORTANCE.keys()),
                'Impact': list(FEATURE_IMPORTANCE.values())
            })
            
            # Create an interactive radial chart
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=feature_impact['Impact'],
                theta=feature_impact['Feature'],
                fill='toself',
                name='Feature Impact',
                line=dict(color=color, width=3),
                hoverinfo='r+theta',
                marker=dict(size=10)
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 0.4],
                        tickfont=dict(color='white', size=16),
                        gridcolor='rgba(255, 255, 255, 0.15)',
                        tickvals=[0, 0.1, 0.2, 0.3, 0.4]
                    ),
                    angularaxis=dict(
                        tickfont=dict(color='white', size=16),
                        gridcolor='rgba(255, 255, 255, 0.15)'
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=False,
                height=450,
                margin=dict(l=60, r=60, t=60, b=60),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on price range
            st.markdown('<div class="section-title">MARKETING RECOMMENDATIONS</div>', unsafe_allow_html=True)
            
            if prediction == 0:  # Budget
                st.markdown(f"""
                <div style="
                    background: rgba(16, 185, 129, 0.2);
                    border-radius: 18px;
                    padding: 1.8rem;
                    border-left: 5px solid #10b981;
                    font-size: 1.15rem;
                ">
                    <h4 style="color: #10b981; margin-top: 0; font-size: 1.5rem;">Budget Phone Strategy</h4>
                    <ul>
                        <li>Target price-sensitive consumers with clear value messaging</li>
                        <li>Highlight battery life and essential features</li>
                        <li>Emphasize durability and reliability</li>
                        <li>Offer competitive pricing with trade-in options</li>
                        <li>Focus on emerging markets and first-time buyers</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif prediction == 1:  # Mid-Range
                st.markdown(f"""
                <div style="
                    background: rgba(245, 158, 11, 0.2);
                    border-radius: 18px;
                    padding: 1.8rem;
                    border-left: 5px solid #f59e0b;
                    font-size: 1.15rem;
                ">
                    <h4 style="color: #f59e0b; margin-top: 0; font-size: 1.5rem;">Mid-Range Phone Strategy</h4>
                    <ul>
                        <li>Position as the sweet spot between price and performance</li>
                        <li>Highlight camera capabilities and display quality</li>
                        <li>Emphasize premium features at accessible prices</li>
                        <li>Target upgraders from budget phones</li>
                        <li>Offer attractive financing options</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif prediction == 2:  # Premium
                st.markdown(f"""
                <div style="
                    background: rgba(239, 68, 68, 0.2);
                    border-radius: 18px;
                    padding: 1.8rem;
                    border-left: 5px solid #ef4444;
                    font-size: 1.15rem;
                ">
                    <h4 style="color: #ef4444; margin-top: 0; font-size: 1.5rem;">Premium Phone Strategy</h4>
                    <ul>
                        <li>Focus on performance and cutting-edge technology</li>
                        <li>Highlight camera systems and display technology</li>
                        <li>Emphasize premium materials and design</li>
                        <li>Target tech enthusiasts and professionals</li>
                        <li>Offer exclusive accessories and services</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:  # Luxury
                st.markdown(f"""
                <div style="
                    background: rgba(139, 92, 246, 0.2);
                    border-radius: 18px;
                    padding: 1.8rem;
                    border-left: 5px solid #8b5cf6;
                    font-size: 1.15rem;
                ">
                    <h4 style="color: #8b5cf6; margin-top: 0; font-size: 1.5rem;">Luxury Phone Strategy</h4>
                    <ul>
                        <li>Position as status symbols and exclusive devices</li>
                        <li>Highlight cutting-edge innovation and unique features</li>
                        <li>Emphasize premium materials and craftsmanship</li>
                        <li>Target affluent consumers and early adopters</li>
                        <li>Offer VIP services and personalized experiences</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <div style="font-size: 1.25rem; font-weight: 600;">MobiCost Analyzer: Smartphone Value Forecaster © 2025</div>
        <div style="margin-top: 0.5rem;">Developed by Ankit Parwatkar | Data Scientist </div>
        <div class="social-links">
            <a href="https://www.linkedin.com/in/ankitparwatkar" target="_blank" title="LinkedIn">
                <i class="fab fa-linkedin"></i>
            </a>
            <a href="https://github.com/ankitparwatkar" target="_blank" title="GitHub">
                <i class="fab fa-github"></i>
            </a>
            <a href="https://twitter.com/ankitparwatkar" target="_blank" title="Twitter">
                <i class="fab fa-twitter"></i>
            </a>
            <a href="https://share.streamlit.io/user/ankitparwatkar" target="_blank" title="Portfolio">
                <i class="fas fa-globe"></i>
            </a>
        </div>
        <div style="margin-top: 1.5rem; font-size: 1.05rem; color: #94a3b8;">
            This tool provides estimates based on market trends and specifications
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add Font Awesome for icons
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()