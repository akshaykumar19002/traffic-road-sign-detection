# Traffic Road Sign Detection using CNN architectures

This project is focused on detecting traffic road signs using various CNN architectures such as LeNet, ResNet, VGG, and Inception V3. The project utilizes a primary dataset for training and testing taken from the Udemy course on Autonomous Cars - Deep Learning and Computer Vision in Python. Additionally, several other datasets were used only for testing purposes.

## Dataset used
1. Primary dataset (training and testing): Udemy course - Autonomous Cars: Deep Learning and Computer Vision in Python
* https://www.udemy.com/course/autonomous-cars-deep-learning-and-computer-vision-in-python/
2. Datasets used only for testing:
* https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb
* https://www.kaggle.com/datasets/bhavinmoriya/german-traffic-sign-recognition-benchmark
* https://www.kaggle.com/datasets/ibrahimkaratas/gtsrb-german-traffic-sign-recognition-benchmark

## Files
* detecting-traffic-signs-updated.ipynb: Jupyter Notebook containing implementation and saving of all models
* deploy-streamlit.py: Python script using Streamlit to deploy the saved models

## CNN Architectures used
* LeNet
* ResNet
* VGG
* Inception V3

## Usage
1. Clone the repository.
2. Install the required libraries listed in requirements.txt.
3. Run detecting-traffic-signs-updated.ipynb to train the models and save them.
4. Run deploy-streamlit.py to deploy the models using Streamlit.

## Conclusion
The project successfully demonstrates the effectiveness of various CNN architectures for detecting traffic road signs. Additionally, using Streamlit to deploy the models allows for easy usage and accessibility.
