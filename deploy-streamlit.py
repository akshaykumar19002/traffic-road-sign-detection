from PIL import Image as im
import pickle
import numpy as np
import streamlit as st
from keras.models import load_model
import os

num_classes = 43
label_classes = {
    0: 'Speed Limit (20km/h)',
    1: 'Speed Limit (30km/h)',
    2: 'Speed Limit (50km/h)',
    3: 'Speed Limit (60km/h)',
    4: 'Speed Limit (70km/h)',
    5: 'Speed Limit (80km/h)',
    6: 'Speed Limit (80km/h)',
    7: 'Speed Limit (100km/h)',
    8: 'Speed Limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General Caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic Signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def resize_image(input_image_path, size=(32, 32)):
    image = im.open(input_image_path)
    resized_image = image.resize(size)
    return np.array(resized_image)[np.newaxis, ...]

def predict_image(model_name, model, image):
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_probability = prediction[0][predicted_class_index]
    st.write(model_name)
    st.write("Predicted class index:", predicted_class_index, " :: Label: ", label_classes[predicted_class_index])
    st.write("Predicted probability:", predicted_probability)
    st.write()
    
def predict(image):
    predict_image('vgg', vgg_model, image)
    predict_image('inception', inception_model, image)
    predict_image('resnet', resnet_model, image)

# base path to the saved models
path = ''

def load_models():
    vgg_model = load_model(path + 'vgg-net.h5')
    inception_model = load_model(path + 'inception-v3.h5')
    resnet_model = load_model(path + 'resnet.h5')
    return vgg_model, inception_model, resnet_model

vgg_model, inception_model, resnet_model = load_models()

def run_app():
    st.title("Traffic Road Sign Classification")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with open(uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        image = resize_image(uploaded_file.name)
        predict(image)
        uploaded_file.seek(0)
        os.remove(uploaded_file.name)

if __name__ == "__main__":
    run_app()