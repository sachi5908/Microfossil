import streamlit as st
import tensorflow as tf
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import requests
import os
import io
import random
from collections import defaultdict
import google.generativeai as genai
import base64

# ------------------ Configuration ------------------
MODEL_URL = st.secrets["MODEL_URL"]
MODEL_PATH = st.secrets["MODEL_PATH"]
API_KEYS = st.secrets["KEYS"].split(",")

GENUS_LIST = [
    "Ammobaculites",
    "Bolivina",
    "Bulimina",
    "Dorothia",
    "Eggerella",
    "Frondicularia",
    "Gaudryna",
    "Globigerina",
    "Globotruncana",
    "Gublerina",
    "Heterohelix",
    "Lagena",
    "Lenticulina",
    "Lituola",
    "Marginulina",
    "Neoflabellina",
    "Nodosaria",
    "Pseudotextularia",
    "Quinqueloculina",
    "Spiroloculina",
    "Triloculina",
    "Tritexia",
    "Trochamminoides",
    "Vernuilina"
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
st.set_page_config(page_title="🦠 Microfossil Genus Classifier", layout="wide")
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

# Custom CSS for sticky header and hidden elements
st.markdown("""
    <style>
        /* Hide Streamlit default UI elements */
        #MainMenu, header, footer {
            visibility: hidden;
        }

        /* Sticky header styling */
        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 1000;
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1rem 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
        }

        /* Main content padding */
        .main-content {
            margin-top: 10px;
            padding: 1rem;
        }

        /* Title styling inside sticky header */
        .main-title {
            font-size: clamp(1.5rem, 5vw, 2rem);  /* Automatically scales */
            font-weight: 800;
            color: #0d47a1;
            text-align: center;
            word-wrap: break-word;
            max-width: 100%;
        }
        @media screen and (max-width: 600px) {
            .sticky-header {
                padding: 0.8rem 1rem;
            }

            .main-title {
                font-size: 1.5rem;
            }
        }
        /* Banner image styling */
        .full-width-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* Model note block */
        .model-note {
            background-color: #009287;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #3f51b5;
            font-size: 0.95rem;
            margin-bottom: 20px;
        }

        /* Predicted genus box */
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

        /* Progress bar container */
        .progress-container {
            background-color: #e0e0e0;
            border-radius: 999px;
            height: 24px;
            width: 100%;
            margin-bottom: 12px;
            overflow: hidden;
        }

        /* Progress bar fill */
        .progress-fill {
            background-color: #1e88e5;
            height: 100%;
            color: white;
            font-weight: 600;
            text-align: center;
            line-height: 24px;
        }

        /* Section titles */
        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 2rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        /* Streamlit app padding fix */
        body { 
            margin-bottom: 80px !important; 
        }
        .stApp {
            padding-bottom: 70px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Fixed header container
st.markdown("""
   <div class="sticky-header">
    <div class="main-title">🔬 Microfossils Recognizer</div>
</div>
    <div class="main-content">
""", unsafe_allow_html=True)

with open("pic.jpg", "rb") as file:
    img_data = file.read()
    img_base64 = base64.b64encode(img_data).decode()

with open("pic.jpg", "rb") as file:
    img_data = file.read()
    img_base64 = base64.b64encode(img_data).decode()

st.markdown(f"""
    <div style="width: 100%; margin-top: 0rem; margin-bottom: 1.5rem;">
        <img src="data:image/jpeg;base64,{img_base64}" 
             alt="Microfossil Banner"
             style="width: 100%; max-height: 250px; object-fit: cover;
                    border-radius: 10px;
                    box-shadow: 0 6px 18px rgba(0,0,0,0.15);">
    </div>
""", unsafe_allow_html=True)

# Note content
st.markdown(f"""
    <div class="model-note">
        <strong>Note:</strong> The model has been trained on the following genera: 
        {", ".join(f"<b>{genus}</b>" for genus in GENUS_LIST)}
    </div>
""", unsafe_allow_html=True)

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


components.html("""
<script>
(function() {
  const doc = window.parent === window ? document : parent.document;
  const win = window.parent === window ? window : parent;

  if (!doc.getElementById("custom-bottom-navbar")) {
    const style = doc.createElement("style");
    style.innerHTML = `
      #custom-bottom-navbar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60px;
        background: #ffffff;
        display: flex;
        justify-content: center;
        gap: 20px;
        align-items: center;
        border-top: 1px solid #ccc;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        font-family: sans-serif;
        z-index: 99999;
      }
      #custom-bottom-navbar button {
        background: none;
        border: none;
        cursor: pointer;
        font-size: 14px;
        color: #333;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 5px 10px;
        transition: all 0.2s ease;
        border-radius: 8px;
      }
      #custom-bottom-navbar button:hover {
        background-color: #f0f0f0;
        transform: scale(1.05);
      }
      #custom-bottom-navbar.dark {
        background-color: #1e1e1e;
        border-top: 1px solid #444;
      }
      #custom-bottom-navbar.dark button {
        color: white;
      }
      body.dark-mode, body.dark-mode * {
        background-color: #121212 !important;
        color: #ffffff !important;
      }
      body.dark-mode .stButton>button,
      body.dark-mode .stTextInput>div>div>input,
      body.dark-mode .stTextArea>div>textarea,
      body.dark-mode .stFileUploader,
      body.dark-mode .stSelectbox {
        background-color: #1f1f1f !important;
        color: white !important;
        border: 1px solid #555 !important;
      }
    `;
    doc.head.appendChild(style);

    const nav = doc.createElement("div");
    nav.id = "custom-bottom-navbar";

    const topBtn = doc.createElement("button");
    topBtn.innerHTML = "⬆️<div style='font-size:10px;'>Top</div>";
    topBtn.title = "Scroll to top";

    const darkBtn = doc.createElement("button");
    darkBtn.innerHTML = "🌙<div style='font-size:10px;'>Theme</div>";
    darkBtn.title = "Toggle dark mode";

    nav.appendChild(topBtn);
    nav.appendChild(darkBtn);
    doc.body.appendChild(nav);

    function doScrollToTop() {
      const anchor = doc.getElementById("top-anchor");
      if (anchor) {
        anchor.scrollIntoView({ behavior: 'smooth' });
      } else {
        win.scrollTo({ top: 0, behavior: 'smooth' });
      }
    }

    function toggleDarkMode() {
      const navBar = doc.getElementById("custom-bottom-navbar");
      const body = doc.body;
      const isDark = navBar.classList.toggle("dark");
      body.classList.toggle("dark-mode", isDark);
    }

    // Assign click handlers
    topBtn.addEventListener("click", doScrollToTop);
    darkBtn.addEventListener("click", toggleDarkMode);

    // Expose scroll function globally for WebView
    win.scrollToTopFromAndroid = doScrollToTop;
  }
})();
</script>
""", height=0)

# Close the content wrapper
st.markdown("</div>", unsafe_allow_html=True)
