import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import io
import random
from collections import defaultdict
import google.generativeai as genai

# MUST BE FIRST - Page config
st.set_page_config(
    page_title="🦠 Microfossil Genus Classifier", 
    layout="wide",
    initial_sidebar_state="auto"
)

# JavaScript for theme and overlay control
st.components.v1.html("""
<script>
// Global theme state
let isDarkMode = false;

// Apply theme to all elements
function applyTheme(darkMode) {
    isDarkMode = darkMode;
    document.body.classList.toggle('dark-mode', darkMode);
    
    // Update all Streamlit components
    document.querySelectorAll('.stApp, .stTextInput, .stButton, .stSelectbox').forEach(el => {
        el.classList.toggle('dark-mode', darkMode);
    });
    
    return true;
}

// Scroll to top function
function scrollToTop() {
    window.scrollTo({top: 0, behavior: 'smooth'});
    document.querySelector('.main')?.scrollTo(0, 0);
    document.querySelector('.block-container')?.scrollTo(0, 0);
    return true;
}

// Initialize theme from localStorage
document.addEventListener('DOMContentLoaded', function() {
    const savedMode = localStorage.getItem('microfossilDarkMode');
    if (savedMode) {
        applyTheme(savedMode === 'true');
    }
    
    // Create top overlay if it doesn't exist
    if (!document.getElementById('android-top-overlay')) {
        const overlay = document.createElement('div');
        overlay.id = 'android-top-overlay';
        overlay.style.position = 'fixed';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.width = '100%';
        overlay.style.height = '50px';
        overlay.style.zIndex = '9999';
        overlay.style.backgroundColor = window.getComputedStyle(document.body).backgroundColor;
        document.body.appendChild(overlay);
    }
});

// Listen for theme change messages
window.addEventListener('message', function(event) {
    if (event.data.type === 'setTheme') {
        applyTheme(event.data.darkMode);
        localStorage.setItem('microfossilDarkMode', event.data.darkMode);
    }
});
</script>
""", height=0)

# CSS for dark mode and overlays
st.markdown("""
<style>
    .dark-mode {
        background-color: #121212 !important;
        color: #ffffff !important;
    }
    .dark-mode .stApp {
        background-color: #121212 !important;
    }
    .dark-mode .genus-box {
        background-color: #424242 !important;
        border-left-color: #bb86fc !important;
    }
    .dark-mode .progress-bar {
        background-color: #333333 !important;
    }
    .dark-mode .progress-fill {
        background: linear-gradient(90deg, #bb86fc, #3700b3) !important;
    }
    
    /* Android overlay styles */
    #android-top-overlay {
        display: none; /* Initially hidden */
    }
</style>
""", unsafe_allow_html=True)


# ============ 3. THEN your other configurations ============
MODEL_URL = st.secrets["MODEL_URL"]
MODEL_PATH = st.secrets["MODEL_PATH"]
API_KEYS = st.secrets["KEYS"].split(",")

GENUS_LIST = [
    'Ammobaculites', 'Dorothia', 'Eggerella', 'Gaudryna',
    'Lituola', 'Quinqueloculina', 'Spiroloculina',
    'Triloculina', 'Tritexia', 'Trochamminoides', 'Vernuilina'
]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)

        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        return tf.keras.models.load_model(MODEL_PATH)

    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# ------------------ Gemini Configuration ------------------
def setup_gemini():
    genai.configure(api_key=random.choice(API_KEYS))
    return genai.GenerativeModel(model_name="gemini-2.0-flash")

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

# ------------------ Fossil Check ------------------
def is_fossil_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        model = setup_gemini()
        image_bytes = image_to_base64(image)

        response = model.generate_content([
            "You are a paleontologist. Is this an image of a microfossil or not? Respond with 'yes' or 'no' only.",
            {
                "mime_type": "image/png",
                "data": image_bytes
            }
        ])

        answer = response.text.strip().lower()
        return "yes" in answer
    except Exception as e:
        st.error(f"❌ Fossil check failed: {str(e)}")
        return False

# ------------------ Feature Extraction ------------------
def get_gemini_features(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        model = setup_gemini()
        image_bytes = image_to_base64(image)

        response = model.generate_content([
            "You are a paleontologist. Given this microfossil image, extract the following features:\n"
            "- Chamber count\n"
            "- Chamber arrangement (linear, spiral, etc.)\n"
            "- Aperture type (central, terminal, etc.)\n"
            "- Wall type (agglutinated, calcareous, etc.)\n"
            "- Any unique shape traits\n"
            "Respond in bullet points.",
            {
                "mime_type": "image/png",
                "data": image_bytes
            }
        ])

        return response.text
    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

# ------------------ Feature to Genus ------------------
def extract_candidate_genera(gemini_text):
    text = gemini_text.lower()
    feature_genus_map = {
        "5": [],
        "6": [],
        "biserial": [],
        "terminal": [],
        "agglutinated": [],
        "elongate": [],
        "teardrop": []
    }

    genus_score = defaultdict(int)
    for feature, genera in feature_genus_map.items():
        if feature in text:
            for genus in genera:
                genus_score[genus] += 1

    sorted_genus = sorted(genus_score.items(), key=lambda x: x[1], reverse=True)
    return [genus for genus, score in sorted_genus]

# ------------------ Prediction ------------------
def predict_genus(image_file, prioritized_genera=None):
    model = load_model()
    if model is None:
        return -1, None, [], []

    try:
        img = Image.open(image_file).convert('RGB').resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Analyzing image..."):
            predictions = model.predict(img_array)[0]

        if prioritized_genera:
            adjusted_preds = []
            for i, genus in enumerate(GENUS_LIST):
                adjusted_preds.append(predictions[i] * (1.5 if genus in prioritized_genera else 0.5))
            predictions = np.array(adjusted_preds)

        predictions /= np.sum(predictions)  # Normalize

        sorted_indices = np.argsort(predictions)[::-1]
        top_index = sorted_indices[0]
        top_genus = GENUS_LIST[top_index]
        top_confidence = predictions[top_index]

        top_predictions = [(GENUS_LIST[i], predictions[i]) for i in sorted_indices[1:3]]  # Next 2

        return top_index, top_genus, top_confidence, top_predictions

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return -1, None, 0.0, []


# ------------------ Streamlit UI ------------------
st.markdown("""
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #3f51b5;
    }
    .genus-box {
        background-color: #e3f2fd;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #1e88e5;
        margin-top: 20px;
        font-size: 1.5rem;
        color: #0d47a1;
    }
    .progress-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 24px;
        margin-top: 8px;
        overflow: hidden;
    }
    .progress-fill {
        background: linear-gradient(90deg, #42a5f5, #1e88e5);
        height: 100%;
        text-align: center;
        color: white;
        line-height: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🔬 Microfossils Recognizer</div>", unsafe_allow_html=True)
st.markdown(
    "<strong>Note:</strong> The model has been trained on the following genera: " +
    ", ".join(f"<b>{genus}</b>" for genus in GENUS_LIST),
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="Uploaded Microfossil", width=300)

    if st.button("🔍 Analyze Image"):
        with st.spinner("🔎 Checking the image..."):
            if not is_fossil_image(uploaded_file):
                st.warning("⚠️ This doesn't appear to be a microfossil. Please upload another image.")
                st.stop()

        with st.spinner("✨ Extracting features..."):
            gemini_output = get_gemini_features(uploaded_file)

        prioritized_genera = extract_candidate_genera(gemini_output)

        result_index, predicted_genus, confidence, top_predictions = predict_genus(uploaded_file, prioritized_genera)

        with col2:
            st.markdown("### 📋 Morphological Features")
            for line in gemini_output.split('\n'):
                line = line.strip()
                if line:
                    st.markdown(f"✅ {line}")

            if result_index != -1:
                confidence_percent = int(confidence * 100)

                st.markdown("""
                    <style>
                    .genus-container {
                        background-color: #e3f2fd;
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin-bottom: 1rem;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                        font-size: 2.4rem;
                        font-weight: 700;
                        text-align: center;
                        color: #0d47a1;
                    }
                    .section-title {
                        font-size: 1.8rem;
                        font-weight: 700;
                        margin-top: 2rem;
                        margin-bottom: 1rem;
                        display: flex;
                        align-items: center;
                        gap: 0.6rem;
                    }
                    .progress-container {
                        background-color: #e0e0e0;
                        border-radius: 999px;
                        height: 24px;
                        width: 100%;
                        margin-bottom: 12px;
                        overflow: hidden;
                    }
                    .progress-fill {
                        background-color: #1e88e5;
                        height: 100%;
                        color: white;
                        font-weight: 600;
                        text-align: center;
                        line-height: 24px;
                    }
                    </style>
                """, unsafe_allow_html=True)

                st.markdown("<div class='section-title'>🧬 Predicted Genus</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='genus-container'>{predicted_genus}</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='section-title'>🟦 Confidence: {confidence_percent}%</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {confidence_percent}%;">
                            {confidence_percent}%
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if confidence_percent < 100 and top_predictions:
                    st.markdown("<div class='section-title'>🔄 Alternative Possibilities</div>", unsafe_allow_html=True)
                    for alt_genus, alt_conf in top_predictions:
                        alt_percent = int(alt_conf * 100)
                        st.markdown(f"<span style='font-size: 1.1rem; font-weight: 600;'>{alt_genus} - {alt_percent}%</span>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="progress-container">
                                <div class="progress-fill" style="width: {alt_percent}%;">
                                    {alt_percent}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
