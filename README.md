Overview
This project focuses on classifying human emotions from facial images using deep learning techniques. It employs convolutional neural network (CNN) architectures such as ResNet, MobileNet, and EfficientNet. Grad-CAM is used for model explainability, and the trained model is deployed using FastAPI, ONNX, and TensorFlow Lite for efficient inference.

Dataset
Classes: Angry, Happy, Sad
Structure:

EmotionsDataset_Splitted/
│── data/
│   ├── train/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── sad/
│   ├── val/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── sad/
│   ├── test/
│   │   ├── angry/
│   │   ├── happy/
│   │   ├── sad/




Setup Instructions

Clone the Repository
git clone https://github.com/your-username/Human-Emotion-Detection.git
cd Human-Emotion-Detection

Install Dependencies
pip install -r requirements.txt

Train the Model
python train.py

Run the API
uvicorn app:app --host 0.0.0.0 --port 8000

Model Architecture

CNN Models: ResNet, MobileNet, EfficientNet
Framework: TensorFlow/Keras
Explainability: Grad-CAM
Deployment: FastAPI, ONNX, TensorFlow Lite

Results & Performance

Achieved XX% accuracy on the test set
Grad-CAM applied for model interpretability

Future Improvements

Expand emotion classes
Improve dataset quality
Optimize model for edge devices
