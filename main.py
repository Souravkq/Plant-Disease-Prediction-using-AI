import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌱",
    layout="wide"
)

# ------------------ STYLE ------------------
st.markdown("""
<style>
body {
    background-color: #222222;
}
.main {
    background-color: #222222;
    color: #F7F7FF;
}
.stButton>button {
    background-color: #89E900;
    color: #222222;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}
.card {
    background-color: #2a2a2a;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #89E900;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL (CACHED) ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')

model = load_model()

#  PREDICTION FUNCTION 
def model_prediction(image):
    img = image.resize((128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return result_index, confidence

# SIDEBAR 
st.sidebar.title(" Dashboard")
app_mode = st.sidebar.radio("Navigate", [" Home", " About", " Predict"])

# CLASS LABELS
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy']

# HOME
if app_mode == " Home":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title(" Plant Disease Recognition System")
    st.write("AI-powered system to detect plant diseases instantly.")
    st.image("home.jpg", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ABOUT 
elif app_mode == " About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title(" About Project")
    st.write("""
    This system uses Deep Learning to classify plant diseases from leaf images.

     Dataset:
    - 87K images  
    - 38 classes  
    - Train/Validation split: 80/20  

     Tech Stack:
    - TensorFlow / Keras  
    - Streamlit  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# PREDICTION 
elif app_mode == " Predict":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title(" Disease Prediction")

    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=192)

        if st.button(" Predict"):
            with st.spinner("Analyzing Image..."):
                result_index, confidence = model_prediction(image)

            st.success("Prediction Complete!")

            st.markdown(f"""
            ###  Result:
            **{class_name[result_index]}**

            ###  Confidence:
            **{round(confidence*100,2)}%**
            """)

    st.markdown("</div>", unsafe_allow_html=True)