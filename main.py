import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import time

# Configuration
MODEL_URL = "https://www.dropbox.com/scl/fi/73zt7x93er1ksywt2f2ah/trained_model.keras?rlkey=dwh6dt0rw6ly1bn6l6r3gqtze&dl=1"
MODEL_PATH = "trained_model.keras"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@st.cache_resource
def load_model():
    """Download and load the model"""
    try:
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH)
        
        # Show download progress only if needed
        download_message = st.empty()
        download_message.warning("Downloading model (70MB)... Please wait...")
        progress_bar = st.progress(0)
        
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                if time.time() - start_time > 0.5:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress)
                    start_time = time.time()
        
        progress_bar.progress(1.0)
        time.sleep(0.5)
        progress_bar.empty()
        download_message.empty()
        
        return tf.keras.models.load_model(MODEL_PATH)
    
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return None

def predict_disease(test_image):
    """Make prediction on uploaded image"""
    model = load_model()
    if model is None:
        return -1
    
    try:
        img = Image.open(test_image).convert('RGB').resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Analyzing image..."):
            predictions = model.predict(img_array)
        
        return np.argmax(predictions)
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return -1

# Streamlit UI Configuration
st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Main App - Only Disease Detection
st.title("🔬 Microfossils Recognizer")

uploaded_file = st.file_uploader("Choose a leaf image", 
                               type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", 
               use_column_width=True)
    
    if st.button("Analyze Image"):
        result = predict_disease(uploaded_file)
        
        if result != -1:
            class_names = [
                'Ammobaculites', 'Dorothia', 'Eggerella', 'Gaudryna',
                'Lituola', 'Quinqueloculina', 'Spiroloculina',
                'Triloculina', 'Tritexia', 'Trochamminoides', 'Vernuilina'
            ]
            
            with col2:
                st.success("Analysis Complete")
                st.markdown(f"""
                ### Prediction Result
                **Detected:** {class_names[result]}
                """)
