# 🧬 Ultrasound Diagnostic Classification System

A TensorFlow model that classifies ultrasound scans into **benign**, **malignant**, or **normal**.  
Includes preprocessing, a FastAPI inference API, a React interface, and a public demo for quick testing.

## 🌐 Live Demo  
https://huggingface.co/spaces/samhitak10/ultrasound-demo

## 🖼 Model Result Preview
<img width="1084" height="511" alt="image" src="https://github.com/user-attachments/assets/11f21cba-72c4-4892-a87d-549ef73a5793" />

## ⚡ Features  
- Upload ultrasound images through the UI  
- Resize and normalize scans before inference  
- Softmax prediction across three classes  
- Confidence score displayed for the predicted class  
- FastAPI backend handles preprocessing and inference  
- React interface shows results in real time  
- Public demo deployed on Hugging Face for quick testing  

## 🧠 Model Details  
- TensorFlow Convolutional Neural Network (CNN)  
- Trained on **925 ultrasound samples**  
- Three diagnostic categories  
- Softmax output with argmax for label selection  
- Training accuracy ~74.8  
- Validation accuracy ~59.8  

**Model stored on Hugging Face:** 
https://huggingface.co/spaces/samhitak10/ultrasound-demo

## ▶️ Local Setup
### Backend
pip install -r requirements.txt  
uvicorn backend.main:app --reload

### Frontend
cd frontend  
npm install  
npm start

## 📝 What I Learned
- Building a multi class Convolutional Neural Network (CNN)  
- Running preprocessing on medical images  
- Using softmax outputs with argmax  
- Designing a FastAPI inference endpoint  
- Connecting a React frontend to a backend  
- Deploying a working demo on Hugging Face Spaces
  
