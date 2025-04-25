# Brain tumor endpoint logic
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import logging
from core.models import bt, load_model
from core.preprocessing import preprocess_image
from core.responses import PredictionResponse
import numpy as np

logger = logging.getLogger("uvicorn")

router = APIRouter()

@router.post("/braintumor_detection", response_model=PredictionResponse)
async def braintumor_detection(file: UploadFile = File(...)):
    global bt
    try:
        logger.info(f"Received brain tumor detection request: {file.filename}")
        if bt is None:
            bt = load_model("braintumor_detection.h5")
        image_array = preprocess_image(file, target_size=(224, 224))
        prediction = bt.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])
        # Map the class index to label
        class_labels = ["glioma", "meningioma", "no tumor", "pituitary"]
        predicted_class_label = class_labels[predicted_class]
        logger.info(f"Brain tumor detection result: {predicted_class_label}, confidence: {confidence}")
        return JSONResponse(content={
            "predicted_class": str(predicted_class_label),
            "confidence": confidence,
            "model_used": "braintumor_detection.h5"
        })
    except Exception as e:
        logger.error(f"Error in brain tumor detection: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})
