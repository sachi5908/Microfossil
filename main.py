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
import base64
import streamlit.components.v1 as components

# ------------------ Config ------------------
MODEL_URL = st.secrets["MODEL_URL"]
MODEL_PATH = st.secrets["MODEL_PATH"]
API_KEYS = st.secrets["KEYS"].split(",")

GENUS_LIST = [
    "Ammobaculites", "Bolivina", "Bulimina", "Dorothia", "Eggerella", "Frondicularia",
    "Gaudryna", "Globigerina", "Globotruncana", "Gublerina", "Heterohelix", "Lagena",
    "Lenticulina", "Lituola", "Marginulina", "Neoflabellina", "Nodosaria",
    "Pseudotextularia", "Quinqueloculina", "Spiroloculina", "Triloculina", "Tritexia",
    "Trochamminoides", "Vernuilina"
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

# ------------------ Gemini ------------------
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
            "You are a paleontologist. Given this microfossil image, extract these features:\n"
            "- Chamber count\n- Arrangement (linear, spiral, etc.)\n"
            "- Aperture type\n- Wall type\n- Shape traits\nRespond in bullet points.",
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
    features = ["5", "6", "biserial", "terminal", "agglutinated", "elongate", "teardrop"]
    genus_score = defaultdict(int)
    for feature in features:
        if feature in text:
            for genus in GENUS_LIST:
                if random.random() > 0.9:  # simulate some matches
                    genus_score[genus] += 1
    return [genus for genus, _ in sorted(genus_score.items(), key=lambda x: x[1], reverse=True)]

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
            predictions = [p * (1.5 if GENUS_LIST[i] in prioritized_genera else 0.5) for i, p in enumerate(predictions)]

        predictions = np.array(predictions)
        predictions /= np.sum(predictions)

        sorted_indices = np.argsort(predictions)[::-1]
        top_index = sorted_indices[0]
        top_genus = GENUS_LIST[top_index]
        top_confidence = predictions[top_index]
        top_predictions = [(GENUS_LIST[i], predictions[i]) for i in sorted_indices[1:3]]

        return top_index, top_genus, top_confidence, top_predictions

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return -1, None, 0.0, []

# ------------------ UI ------------------
st.set_page_config(page_title="🦠 Microfossils Recognizer", layout="wide")
st.markdown("""<div id="top-anchor"></div>""", unsafe_allow_html=True)

# Add sticky header and anchor
st.markdown("""
    <style>
    /* Hide Streamlit's GitHub, Fork, and Theme icons in top right */
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0px !important;
    }

    /* Optional: Remove top padding left behind */
    .stApp {
        padding-top: 0rem !important;
    }
    /* Sticky header */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: white;
        padding: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Full width image with controlled height */
    .full-width-image {
        width: 100%;
        height: 200px;  /* Adjust this value to control height */
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    body { 
        margin-bottom: 80px !important; 
    }
    .stApp {
        padding-bottom: 70px !important;
    }
    #top-anchor {
        position: absolute;
        top: 0;
        visibility: hidden;
    }

    #custom-bottom-navbar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60px;
        background: #ffffff;
        display: flex;
        justify-content: space-around;
        align-items: center;
        border-top: 1px solid #ccc;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        font-family: sans-serif;
        z-index: 10000 !important;  /* Make it higher than Streamlit icons */
    }
    /* Fix z-index of Streamlit top-right buttons */
    [data-testid="stDecoration"] {
        z-index: 1 !important;
    }

    /* Ensure custom bottom navbar is on top */
    #custom-bottom-navbar {
        z-index: 10000 !important;
    }
    </style>
    
    <div id="top-anchor"></div>
    <div class="sticky-header">
        <h1 style='text-align:center; color:#4A90E2;'>🔬 Microfossils Recognizer</h1>
    </div>
""", unsafe_allow_html=True)

# Full width image with controlled height
with open("pic.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <img src='data:image/jpeg;base64,{img_base64}' class='full-width-image'>
    """, unsafe_allow_html=True)

st.info(f"""
**Note:** The model has been trained on the following genera:
{', '.join(GENUS_LIST)}
""")

uploaded_file = st.file_uploader("📤 Upload a microfossil image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("🔍 Analyze Image", type="primary"):
        if not is_fossil_image(uploaded_file):
            st.warning("⚠️ This does not appear to be a microfossil.")
            st.stop()

        gemini_output = get_gemini_features(uploaded_file)
        prioritized_genera = extract_candidate_genera(gemini_output)
        result_index, predicted_genus, confidence, top_predictions = predict_genus(uploaded_file, prioritized_genera)

        with col2:
            st.subheader("📋 Morphological Features")
            st.markdown(gemini_output)

            if predicted_genus:
                st.success(f"✅ Predicted Genus: **{predicted_genus}** with **{int(confidence * 100)}%** confidence")

                for alt_genus, alt_conf in top_predictions:
                    st.info(f"🔍 Alternative: {alt_genus} ({int(alt_conf * 100)}%)")

# ------------------ Bottom Navbar ------------------
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
        justify-content: space-around;
        align-items: center;
        border-top: 1px solid #ccc;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        font-family: sans-serif;
        z-index: 999999 !important;
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

      /* Force Streamlit decorations lower */
      [data-testid="stDecoration"] {
        z-index: 1 !important;
      }

      /* Optional dark mode support */
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

    const refreshBtn = doc.createElement("button");
    refreshBtn.innerHTML = "🔄<div style='font-size:10px;'>Refresh</div>";
    refreshBtn.title = "Refresh page";

    const darkBtn = doc.createElement("button");
    darkBtn.innerHTML = "🌙<div style='font-size:10px;'>Dark</div>";
    darkBtn.title = "Toggle dark mode";

    nav.appendChild(topBtn);
    nav.appendChild(refreshBtn);
    nav.appendChild(darkBtn);
    doc.body.appendChild(nav);

    topBtn.addEventListener("click", function() {
      const anchor = doc.getElementById("top-anchor");
      if (anchor) {
        anchor.scrollIntoView({ behavior: 'smooth' });
      } else {
        win.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });

    refreshBtn.addEventListener("click", function() {
      win.location.reload();
    });

    darkBtn.addEventListener("click", function() {
      const navBar = doc.getElementById("custom-bottom-navbar");
      const body = doc.body;
      const isDark = navBar.classList.toggle("dark");
      body.classList.toggle("dark-mode", isDark);
    });
  }
})();
</script>
""", height=0)

