import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

MODEL_PATH = "Model Training/fetal_ultrasound_classifier.h5"

# Create ONE FastAPI app
app = FastAPI()

# CORS (fixes “Failed to fetch”)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["benign", "malignant", "normal"]

@app.get("/health")
def health():
    return {"status": "ok"}

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg"]:
        return JSONResponse(
            content={"error": "Only PNG or JPEG images allowed"},
            status_code=400
        )

    contents = await file.read()
    arr = preprocess_image(contents)

    preds = model.predict(arr)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return {
        "prediction": CLASS_NAMES[class_idx],
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
