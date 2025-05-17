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
import json
import re

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

def get_gemini_features(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        model = setup_gemini()
        image_bytes = image_to_base64(image)

        response = model.generate_content([
            """You are an expert paleontologist. Given this microfossil image, extract the following features and return them as a structured JSON object with keys exactly as shown:

{
  "chamber_arrangement": "e.g., uniserial, biserial, triserial, spiral (planispiral, trochospiral, involute, etc.)",
  "aperture": "e.g., terminal, central, interiomarginal, umbilical",
  "wall_type": "e.g., agglutinated, calcareous, porcelaneous, perforated, imperforated",
  "coiling": {
    "present": true or false,
    "type": "e.g., planispiral, trochospiral, involute, if present"
  },
  "traits": ["list of unique shape traits like unilocular, flattened, spined, etc."]
}

If any field is not observable, write null or an empty list.""",
            {
                "mime_type": "image/png",
                "data": image_bytes
            }
        ])

        return response.text
    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

# ------------------ Feature to Genus ------------------
import json
from collections import defaultdict

def extract_candidate_genera(gemini_text):
    try:
        data = json.loads(gemini_text)  # Now expecting proper JSON
    except json.JSONDecodeError:
        return []  # If Gemini fails to return valid JSON

    feature_genus_map = {
        "unilocular": ["Legena"],
        "flattened": ["Frondicularia", "Neoflabellina"],
        "uniserial": ["Lituola", "Ammobaculites", "Nodosaria", "Frondicularia", "Marginulina", "Neoflabellina"],
        "biserial": ["Dorothia", "Heterohelix", "Pseudotextularia", "Gublerina", "Bulimina", "Bolivina"],
        "triserial": ["Vernuilina", "Tritexia", "Bulimina"],
        "trochospiral": ["Eggerella", "Globigerina", "Globotruncana"],
        "steptospiral": ["Spiroloculina", "Quinqueloculina", "Triloculina"],
        "early triserial": ["Gaudryna"],
        "early trochospiral": ["Dorothia"],
        "terminal": ["Ammobaculites", "Lituola", "Quinqueloculina", "Triloculina", "Nodosaria", "Frondicularia", "Lenticulina", "Marginulina", "Neoflabellina", "Gublerina"],
        "interiomarginal": ["Trochamminoides", "Tritexia", "Gaudryna", "Dorothia", "Eggerella", "Globotruncana", "Heterohelix", "Pseudotextularia", "Bulimina"],
        "umbilical": ["Globigerina", "Globotruncana"],
        "agglutinated": ["Lituola", "Ammobaculites", "Trochamminoides", "Vernuilina", "Gaudryna", "Tritexia", "Dorothia", "Eggerella"],
        "calcareous": ["Spiroloculina", "Quinqueloculina", "Triloculina", "Nodosaria", "Legena", "Frondicularia", "Lenticulina", "Marginulina", "Neoflabellina", "Globigerina", "Globotruncana", "Heterohelix", "Pseudotextularia", "Gublerina", "Bulimina", "Bolivina"],
        "imperforated": ["Spiroloculina"],
        "perforated": ["Nodosaria", "Legena", "Lenticulina", "Marginulina", "Neoflabellina", "Globigerina", "Globotruncana", "Pseudotextularia", "Gublerina", "Bulimina", "Bolivina"],
        "porcellaneous": ["Spiroloculina", "Quinqueloculina", "Triloculina"],
        "coiling": ["Trochamminoides"],
        "planispiral": ["Trochamminoides", "Lenticulina"],
        "early planispirally": ["Lituola"]
    }

    genus_score = defaultdict(int)

    # Match single string features
    for key in ["chamber_arrangement", "aperture", "wall_type"]:
        value = data.get(key)
        if value:
            value = value.lower()
            if value in feature_genus_map:
                for genus in feature_genus_map[value]:
                    genus_score[genus] += 1

    # Handle coiling
    coiling = data.get("coiling", {})
    if isinstance(coiling, dict):
        if coiling.get("present") and coiling.get("type"):
            coiling_type = coiling["type"].lower()
            if coiling_type in feature_genus_map:
                for genus in feature_genus_map[coiling_type]:
                    genus_score[genus] += 1

    # Handle traits (list)
    traits = data.get("traits", [])
    for trait in traits:
        trait = trait.lower()
        if trait in feature_genus_map:
            for genus in feature_genus_map[trait]:
                genus_score[genus] += 1

    # Return sorted genus list
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

# Custom CSS for sticky header and hidden elements
st.markdown("""
<style>
    /* Hide Streamlit default UI elements */
    #MainMenu, header, footer {
        visibility: hidden;
    }

    /* Reset body margin and padding */
    body {
        margin: 0 !important;
        padding: 0 !important;
    
    }

    .stApp {
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Sticky header */
        .sticky-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: #ffffff;
        padding: 1rem 2rem;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12);
        margin: 0 auto;
        max-width: 90%;
        margin-top: 2px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-wrap: wrap;
    }


    .main-title {
        font-size: clamp(1.5rem, 5vw, 2rem);
        font-weight: 800;
        color: #0d47a1;
        text-align: center;
        max-width: 100%;
    }
    @media screen and (max-width: 600px) {
        .sticky-header {
            padding: 0.8rem 1rem;
            max-width: 95%;
            margin-top: 5px;
        }

        .main-title {
            font-size: 1.5rem;
        }
    }

    .main-content {

        padding: 1rem;
    }

    /* Other styles (unchanged) */
    .full-width-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .model-note {
        background-color: #a5ffe0;
        color: black;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3f51b5;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .model-note2 {
        background-color: #ffe2d9;
        color: black;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3f51b5;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

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

    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    div.streamlit-expanderHeader {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    div.streamlit-expanderHeader p {
        margin: 0;
        font-size: inherit;
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
            st.markdown(f"""
<div class="model-note2">
⚠️ <strong>Disclaimer:</strong> Extracted features and predicted genus generated by an AI model and may occasionally be incomplete or incorrect due to image quality, fossil damage, or complex morphology. Please verify all results with expert judgment where needed.
</div>
""", unsafe_allow_html=True)

            with st.expander("📋 Morphological Features"):
                try:
                    # Clean Gemini's response if wrapped in ```json ... ```
                    cleaned_text = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", gemini_output).strip()
                    parsed = json.loads(cleaned_text)

                    if parsed.get("chamber_arrangement"):
                        st.markdown(f"✅ **Chamber Arrangement:** {parsed['chamber_arrangement']}")

                    if parsed.get("aperture"):
                        st.markdown(f"✅ **Aperture Type:** {parsed['aperture']}")

                    if parsed.get("wall_type"):
                        st.markdown(f"✅ **Wall Type:** {parsed['wall_type']}")

                    coiling = parsed.get("coiling", {})
                    if isinstance(coiling, dict):
                        if coiling.get("present"):
                            coil_type = coiling.get("type", "unspecified")
                            st.markdown(f"✅ **Coiling:** Yes ({coil_type})")
                        else:
                            st.markdown("✅ **Coiling:** No")

                    traits = parsed.get("traits", [])
                    if traits:
                        trait_list = ', '.join(traits)
                        st.markdown(f"✅ **Shape Traits:** {trait_list}")

                except json.JSONDecodeError:
                    st.error("❌ Failed to parse morphological features. Invalid JSON format.")
                except Exception as e:
                    st.error(f"❌ Error displaying features: {str(e)}")


            # ✅ Prediction Output
            if result_index != -1:
                confidence_percent = int(confidence * 100)

                st.markdown("<div class='section-title'>🧬 Predicted Genus</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='genus-container'>{predicted_genus}</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='section-title'>🟦 Confidence: {confidence_percent}%</div>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-fill" style="width: {confidence_percent}%;">{confidence_percent}%</div>
                    </div>
                """, unsafe_allow_html=True)

                if confidence_percent < 100 and top_predictions:
                    st.markdown("<div class='section-title'>🔄 Alternative Possibilities</div>", unsafe_allow_html=True)
                    for alt_genus, alt_conf in top_predictions:
                        alt_percent = int(alt_conf * 100)
                        st.markdown(f"<span style='font-size: 1.1rem; font-weight: 600;'>{alt_genus} - {alt_percent}%</span>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div class="progress-container">
                                <div class="progress-fill" style="width: {alt_percent}%;">{alt_percent}%</div>
                            </div>
                        """, unsafe_allow_html=True)

# Close the content wrapper
st.markdown("</div>", unsafe_allow_html=True)
