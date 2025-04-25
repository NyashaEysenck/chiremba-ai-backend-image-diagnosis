# Pneumonia endpoint logic
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import logging
from core.models import pn, load_model
from core.preprocessing import preprocess_image
from core.responses import PredictionResponse
import numpy as np

logger = logging.getLogger("uvicorn")

router = APIRouter()

@router.post("/pneumonia_detection", response_model=PredictionResponse)
async def pneumonia_detection(file: UploadFile = File(...)):
    global pn
    try:
        logger.info(f"Received pneumonia detection request: {file.filename}")
        if pn is None:
            pn = load_model("pneumonia_detection.h5")
        # Process the image (match hostlocal.py)
        image_array = preprocess_image(file, target_size=(300, 300))
        prediction = pn.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])
        # Map the class index to label (match hostlocal.py)
        class_labels = ["Normal", "Pneumonia"]
        predicted_class_label = class_labels[predicted_class]
        logger.info(f"Pneumonia detection result: {predicted_class_label}, confidence: {confidence}")
        return JSONResponse(content={
            "predicted_class": str(predicted_class_label),
            "confidence": confidence,
            "model_used": "pneumonia_detection.h5"
        })
    except Exception as e:
        logger.error(f"Error in pneumonia detection: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})
