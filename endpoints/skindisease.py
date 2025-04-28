# Skin disease classification endpoint logic
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import logging
from core.models import load_model
from core.responses import PredictionResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf

logger = logging.getLogger("uvicorn")

router = APIRouter()

# Updated class labels for Skinalyze
SKINALYZE_CLASS_LABELS = [
    'Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox'
]

skinalyze_model = None

@router.post("/skindisease_classification", response_model=PredictionResponse)
async def skindisease_classification(file: UploadFile = File(...)):
    global skinalyze_model
    try:
        logger.info(f"Received skin disease classification request: {file.filename}")
        if skinalyze_model is None:
            skinalyze_model = load_model("SkinNet-23M.h5")
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((180, 180))
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)
        prediction = skinalyze_model.predict(image_array)
        probabilities = tf.nn.softmax(prediction[0]).numpy()
        sorted_indices = np.argsort(probabilities)[::-1]
        top_idx = sorted_indices[0]
        top_conf = float(probabilities[top_idx])
        top_class = SKINALYZE_CLASS_LABELS[top_idx]
        # Get top 3 predictions
        top_3_indices = sorted_indices[:3]
        top_3_classes = [SKINALYZE_CLASS_LABELS[i] for i in top_3_indices]
        top_3_confidences = [float(probabilities[i]) for i in top_3_indices]
        logger.info(f"Top 3 predictions: {list(zip(top_3_classes, top_3_confidences))}")
        logger.info(f"Skin disease classification result: {top_class}, confidence: {top_conf}")
        return JSONResponse(content={
            "predicted_class": str(top_class),
            "confidence": top_conf,
            "alternatives": [
                {"class": top_3_classes[1], "confidence": top_3_confidences[1]},
                {"class": top_3_classes[2], "confidence": top_3_confidences[2]}
            ]
        })
    except Exception as e:
        logger.error(f"Error in skin disease classification: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})
