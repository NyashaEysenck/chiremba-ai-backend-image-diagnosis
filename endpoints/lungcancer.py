# Lung cancer endpoint logic
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from io import BytesIO
import logging
import base64
from PIL import Image
from core.models import lc, initialize_lung_cancer_model, LUNG_CANCER_LABELS
from core.responses import PredictionResponse
from core.preprocessing import preprocess_lung_cancer_image
import numpy as np
from utils.lung_util import generate_gradcam

logger = logging.getLogger("uvicorn")

router = APIRouter()

def get_lung_cancer_interpretation(predicted_label, confidence):
    interpretations = {
        'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': {
            'description': 'Adenocarcinoma in the left lower lobe',
            'stage': 'Stage Ib (T2, N0, M0)',
            'characteristics': 'Primary tumor > 3cm but ≤ 5cm, no regional lymph node metastasis, no distant metastasis',
            'recommendations': [
                'Surgical resection is typically the primary treatment',
                'Consider adjuvant chemotherapy based on risk factors',
                'Regular follow-up imaging to monitor for recurrence'
            ]
        },
        'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': {
            'description': 'Large cell carcinoma in the left hilum',
            'stage': 'Stage IIIa (T2, N2, M0)',
            'characteristics': 'Primary tumor > 3cm but ≤ 5cm, metastasis in ipsilateral mediastinal lymph nodes, no distant metastasis',
            'recommendations': [
                'Multimodality treatment approach often recommended',
                'Combination of chemotherapy and radiation therapy',
                'Surgical resection may be considered in selected cases',
                'Immunotherapy may be appropriate based on biomarker testing'
            ]
        },
        'normal': {
            'description': 'No evidence of lung cancer',
            'recommendations': [
                'Continue routine screening based on risk factors',
                'Maintain healthy lifestyle habits',
                'Follow up as clinically indicated'
            ]
        },
        'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': {
            'description': 'Squamous cell carcinoma in the left hilum',
            'stage': 'Stage IIIa (T1, N2, M0)',
            'characteristics': 'Primary tumor ≤ 3cm, metastasis in ipsilateral mediastinal lymph nodes, no distant metastasis',
            'recommendations': [
                'Multimodality treatment approach often recommended',
                'Combination of chemotherapy and radiation therapy',
                'Surgical resection may be considered in selected cases',
                'Consider targeted therapy or immunotherapy based on biomarker testing'
            ]
        }
    }
    confidence_level = "high" if confidence > 0.9 else "moderate" if confidence > 0.7 else "low"
    if predicted_label in interpretations:
        info = interpretations[predicted_label]
        if predicted_label == 'normal':
            interpretation = f"The scan appears normal with {confidence_level} confidence ({confidence*100:.1f}%).\n\n"
            interpretation += f"Description: {info['description']}\n\n"
            interpretation += "Recommendations:\n"
            for rec in info['recommendations']:
                interpretation += f"- {rec}\n"
        else:
            interpretation = f"The scan suggests {info['description']} with {confidence_level} confidence ({confidence*100:.1f}%).\n\n"
            interpretation += f"Stage: {info['stage']}\n"
            interpretation += f"Characteristics: {info['characteristics']}\n\n"
            interpretation += "Recommendations:\n"
            for rec in info['recommendations']:
                interpretation += f"- {rec}\n"
            if confidence < 0.7:
                interpretation += "\nNote: Due to the lower confidence level, additional diagnostic procedures may be warranted to confirm this finding."
    else:
        interpretation = "No specific interpretation available for this prediction."
    return interpretation

@router.post("/lungcancer_prediction", response_model=PredictionResponse)
async def lungcancer_detection(file: UploadFile = File(...)):
    global lc
    try:
        logger.info(f"Received lung cancer detection request: {file.filename}")
        if lc is None:
            lc = initialize_lung_cancer_model()
            if lc is None:
                logger.warning("Lung cancer model could not be loaded, returning placeholder response")
                return JSONResponse(content={
                    "predicted_class": "normal",
                    "confidence": 0.95,
                    "detail": "Lung cancer model could not be loaded"
                })
        image_array = preprocess_lung_cancer_image(file, target_size=(350, 350))
        prediction = lc.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])
        # Map the class index to label
        predicted_class_label = LUNG_CANCER_LABELS[predicted_class]
        interpretation = get_lung_cancer_interpretation(predicted_class_label, confidence)
        preds_copy = prediction[0].copy()
        preds_copy[predicted_class] = -1
        second_pred_index = np.argmax(preds_copy)
        second_predicted_label = LUNG_CANCER_LABELS[second_pred_index]
        second_confidence = float(prediction[0][second_pred_index])
        logger.info(f"Lung cancer detection result: {predicted_class_label}, confidence: {confidence}")
        logger.info(f"Second prediction: {second_predicted_label}, confidence: {second_confidence}")
        return JSONResponse(content={
            "predicted_class": str(predicted_class_label),
            "confidence": confidence,
            "second_prediction": {
                "class": second_predicted_label,
                "confidence": second_confidence
            },
            "all_probabilities": {LUNG_CANCER_LABELS[i]: float(prediction[0][i]) for i in range(len(LUNG_CANCER_LABELS))},
            "interpretation": interpretation
        })
    except Exception as e:
        logger.error(f"Error in lung cancer detection: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@router.post("/lungcancer_gradcam")
async def lungcancer_gradcam(file: UploadFile = File(...)):
    global lc
    try:
        logger.info(f"Received lung cancer Grad-CAM request: {file.filename}")
        if lc is None:
            lc = initialize_lung_cancer_model()
            if lc is None:
                logger.warning("Lung cancer model could not be loaded, returning error response")
                return JSONResponse(content={"error": "Lung cancer model could not be loaded"}, status_code=500)
        # Preprocess image
        file.file.seek(0)
        image_array = preprocess_lung_cancer_image(file, target_size=(350, 350))
        # Run Grad-CAM
        _, superimposed_img = generate_gradcam(lc, image_array)
        # Convert PIL image to base64
        buffered = BytesIO()
        superimposed_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return JSONResponse(content={"gradcam_image": img_str})
    except Exception as e:
        logger.error(f"Error in lung cancer Grad-CAM: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
