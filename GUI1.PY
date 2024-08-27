import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Set the page configuration as the first Streamlit command
st.set_page_config(layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    h1 {
        color: black;
        text-align: center;
        font-size: 48px;
        margin-top: 20px;
    }
    h2 {
        color: Maroon;
        text-align: center;
        font-size: 24px;
        margin-bottom: 40px;
        text-shadow: 2px 2px 2px rgba(255, 255, 255, 0.7);
    }
    .bold-text {
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-text {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .metric-text {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1>CLASSIFICATION OF MEDICAL IMAGE BASED ON TEXTURE ANALYSIS</h1>', unsafe_allow_html=True)

# Define a function to read and encode the image file
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set the background image
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: 224*224;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# File uploader for image classification
upload = st.file_uploader('*Insert image for classification*', type=['png', 'jpg'])

# Load models with custom objects (if any)
def load_custom_model(model_path):
    try:
        from tensorflow.keras.layers import DepthwiseConv2D
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                if 'groups' in kwargs:
                    kwargs.pop('groups')
                super().__init__(**kwargs)

        model = load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        st.success(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

# Paths for the models
model1_path = os.path.join(r'C:\Users\dell\Desktop\ram\DensNet121.h5')
model2_path = os.path.join(r'C:\Users\dell\Desktop\ram\MobileNet.h5')
model3_path = os.path.join(r'C:\Users\dell\Desktop\ram\VGG16.h5')

# Load the models if the paths are valid
model1 = load_custom_model(model1_path) if os.path.exists(model1_path) else None
model2 = load_custom_model(model2_path) if os.path.exists(model2_path) else None
model3 = load_custom_model(model3_path) if os.path.exists(model3_path) else None

# Define class labels
class_labels = ['LEFT_FOREARM', 'LEFT_HAND', 'LEFT_HAND_ELBOW', 'LEFT_HAND_FINGURE', 'LEFT_SHOULDER',
                'RIGHT_FOREARM', 'RIGHT_HAND', 'RIGHT_HAND_ELBOW', 'RIGHT_HAND_FINGURE', 'RIGHT_SHOULDER']

def predict_image(image, model):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224, 224))  # Resize the image to the required size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)

def plot_confidence_bar(confidences, predicted_class, model_names):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))

    confidence_values = [confidence[predicted_class] for confidence in confidences]
    plt.bar(x, confidence_values, color=['blue', 'orange', 'green'])
    
    plt.xlabel('Models')
    plt.ylabel('Confidence')
    plt.title('Prediction Confidence Comparison')
    plt.xticks(x, model_names)
    plt.ylim(0, 1)  # Since confidence scores are probabilities, they range from 0 to 1
    for i, v in enumerate(confidence_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

    st.pyplot(plt)

# Create tabs
tab1, tab2 = st.tabs(["Classification", "Model Comparison"])

with tab1:
    if upload is not None:
        im = Image.open(upload)
        st.image(im, caption='Uploaded Image', use_column_width=True)

        st.markdown('<div class="bold-text">Classifying...</div>', unsafe_allow_html=True)

        # Initialize lists to store predictions and confidences
        predictions = []
        confidences = []

        # Get predictions from each model
        if model1 is not None:
            prediction1 = predict_image(im, model1)
            predicted_class1 = np.argmax(prediction1, axis=1)[0]
            st.markdown(f"<div class='prediction-text'>VGG16 Prediction: {class_labels[predicted_class1]}</div>", unsafe_allow_html=True)
            predictions.append(predicted_class1)
            confidences.append(prediction1[0])

        if model2 is not None:
            prediction2 = predict_image(im, model2)
            predicted_class2 = np.argmax(prediction2, axis=1)[0]
            st.markdown(f"<div class='prediction-text'>MobileNetV2 Prediction: {class_labels[predicted_class2]}</div>", unsafe_allow_html=True)
            predictions.append(predicted_class2)
            confidences.append(prediction2[0])

        if model3 is not None:
            prediction3 = predict_image(im, model3)
            predicted_class3 = np.argmax(prediction3, axis=1)[0]
            st.markdown(f"<div class='prediction-text'>DenseNet121 Prediction: {class_labels[predicted_class3]}</div>", unsafe_allow_html=True)
            predictions.append(predicted_class3)
            confidences.append(prediction3[0])

        # Check if predictions were made
        if predictions:
            # Assuming the first prediction as the "true" label for this example
            y_true = [predictions[0]] * len(predictions)
            
            # Simple evaluation metrics
            accuracy = accuracy_score(y_true, predictions)
            st.markdown(f"<div class='metric-text'>Accuracy: {accuracy:.2f}</div>", unsafe_allow_html=True)

            # Plot the confidence bar graph for the predicted class
            predicted_class = predictions[0]
            model_names = ["VGG16", "MobileNetV2", "DenseNet121"]
            plot_confidence_bar(confidences, predicted_class, model_names)

with tab2:
    st.markdown("<h1>Model Accuracy Comparison</h1>", unsafe_allow_html=True)
    image_path1 = r"C:\Users\shivd\OneDrive\Desktop\Shiv\model_accuracy_comparison.png"
    
    if os.path.exists(image_path1):
        st.image(image_path1, caption="Model Accuracy Comparison", use_column_width=True)
    else:
        st.error(f"Error: {image_path1} does not exist.")
