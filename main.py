import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Set TensorFlow logging to avoid unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Cache the model loading
@st.cache_resource
def load_model():
    try:
        # Check if model file exists
        if not os.path.exists("trained_model.keras"):
            with st.spinner("Downloading model from Google Drive... This may take a few minutes"):
                # Google Drive direct download link
                url = "https://drive.google.com/uc?id=1dkqcu5OpiKKoY6i7-i___3-za_18Abh6"
                output = "trained_model.keras"
                gdown.download(url, output, quiet=False)
        
        return tf.keras.models.load_model("trained_model.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def model_prediction(test_image):
    try:
        model = load_model()
        if model is None:
            return -1
            
        # Convert the uploaded file to an image
        image = Image.open(test_image)
        image = image.resize((128, 128))
        
        # Convert image to array and preprocess
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # convert single image to batch
        
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # return index of max element
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return -1

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    Upload a plant leaf image to detect diseases.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset.
    It consists of about 87K rgb images of healthy and diseased crop leaves categorized into 38 classes.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            with st.spinner("Analyzing the image..."):
                result_index = model_prediction(test_image)
                
                if result_index != -1:
                    class_name = ['Ammobaculites', 'Dorothia', 'Eggerella', 'Gaudryna', 
                                'Lituola', 'Quinqueloculina', 'Spiroloculina', 
                                'Triloculina', 'Tritexia', 'Trochamminoides', 'Vernuilina']
                    st.success(f"Model prediction: {class_name[result_index]}")
