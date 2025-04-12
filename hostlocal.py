from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import logging
import tensorflow as tf
import os
from scipy import ndimage
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
# Find the root directory (where .env is located)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join('.env')
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI()

# Enable CORS middleware with specific configuration for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174", "http://192.168.1.43:5173", "http://192.168.1.43:5174", "http://localhost:4173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"]  # Expose all headers in response
)

# Models will be loaded on-demand using lazy loading
pn = None
sc_d = None
sd_c = None
bt = None
isic_skin_model = None  # New ISIC 2024 model
lc = None  # Lung cancer model

# Flag to control which skin cancer model to use
USE_ISIC_MODEL = True  # Set to True to use the new ISIC 2024 model

# Define Pydantic model for the prediction response
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_used: str = None

# Helper function to preprocess images for model input
def preprocess_image(file: UploadFile, target_size=(224, 224)):
    """
    Preprocess an image file for model prediction.
    
    Args:
        file: The uploaded file
        target_size: The target size for resizing (width, height)
        
    Returns:
        A preprocessed numpy array ready for model input
    """
    try:
        # Read the image data
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Log image details
        logger.info(f"Original image mode: {image.mode}, size: {image.size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB mode")
        
        # Resize to the target size
        image = image.resize(target_size)
        
        # Convert to numpy array
        image_array = np.array(image, dtype=np.float32)
        
        # Handle grayscale images by duplicating to 3 channels
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
            
        # Normalize pixel values to [0, 1]
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

# Helper function to load models only when needed
def load_model(model_name):
    logger.info(f"Loading model: {model_name}")
    try:
        # First try to find the model in the models directory
        model_path = os.path.join("models", model_name)
        
        # If not found, try in src/models directory
        if not os.path.exists(model_path):
            model_path = os.path.join("src", "models", model_name)
            logger.info(f"Looking for model in: {model_path}")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path} or src/models/{model_name}")
        
        # Load the model with custom_objects to handle batch_shape parameter
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer
        }
        
        # Try to load the model directly
        try:
            logger.info(f"Attempting to load model directly: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model directly: {model_name}")
            return model
        except Exception as first_error:
            logger.warning(f"First attempt to load model failed: {str(first_error)}")
            
            # If direct loading fails, try with h5py to modify the model file
            import h5py
            
            logger.info(f"Attempting to load model with h5py: {model_path}")
            with h5py.File(model_path, 'r+') as h5file:
                # Check if there are layers with batch_shape
                if 'model_weights' in h5file:
                    for layer_name in h5file['model_weights']:
                        if 'batch_shape' in h5file['model_weights'][layer_name].attrs:
                            # Get the batch_shape attribute
                            batch_shape = h5file['model_weights'][layer_name].attrs['batch_shape']
                            # Create a new batch_input_shape attribute
                            h5file['model_weights'][layer_name].attrs['batch_input_shape'] = batch_shape
                            logger.info(f"Converted batch_shape to batch_input_shape for layer: {layer_name}")
            
            # Try loading again after modification
            logger.info(f"Attempting to load model after h5py modification: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model after h5py modification: {model_name}")
            return model
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Helper function to load the ISIC 2024 skin cancer model
def load_isic_model():
    logger.info(f"Loading ISIC 2024 skin cancer model")
    try:
        # First try to find the model in the models directory
        model_path = os.path.join("models", "skincancer_prediction.keras")
        
        # If not found, try in src/models directory
        if not os.path.exists(model_path):
            model_path = os.path.join("src", "models", "skincancer_prediction.keras")
            logger.info(f"Looking for model in: {model_path}")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path} or src/models/skincancer_prediction.keras")
        
        # Load the model with custom_objects to handle batch_shape parameter
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer
        }
        
        # Try to load the model directly
        try:
            logger.info(f"Attempting to load model directly: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model directly: ISIC 2024 skin cancer model")
            return model
        except Exception as first_error:
            logger.warning(f"First attempt to load model failed: {str(first_error)}")
            
            # If direct loading fails, try with h5py to modify the model file
            import h5py
            
            logger.info(f"Attempting to load model with h5py: {model_path}")
            with h5py.File(model_path, 'r+') as h5file:
                # Check if there are layers with batch_shape
                if 'model_weights' in h5file:
                    for layer_name in h5file['model_weights']:
                        if 'batch_shape' in h5file['model_weights'][layer_name].attrs:
                            # Get the batch_shape attribute
                            batch_shape = h5file['model_weights'][layer_name].attrs['batch_shape']
                            # Create a new batch_input_shape attribute
                            h5file['model_weights'][layer_name].attrs['batch_input_shape'] = batch_shape
                            logger.info(f"Converted batch_shape to batch_input_shape for layer: {layer_name}")
            
            # Try loading again after modification
            logger.info(f"Attempting to load model after h5py modification: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model after h5py modification: ISIC 2024 skin cancer model")
            return model
            
    except Exception as e:
        logger.error(f"Error loading ISIC 2024 skin cancer model: {str(e)}")
        raise

# Define lung cancer class labels
LUNG_CANCER_LABELS = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

# Helper function to initialize the lung cancer model
def initialize_lung_cancer_model():
    """Initialize and load the lung cancer prediction model."""
    logger.info("Initializing lung cancer model")
    
    # Check for model in different locations
    model_paths = [
        os.path.join("models", "lung_cancer_prediction_model_complete.h5"),
        os.path.join("src", "models", "lung_cancer_prediction_model_complete.h5"),
        os.path.join("models", "lungcancer_prediction.keras"),
        os.path.join("src", "models", "lungcancer_prediction.keras")
    ]
    
    # Try to load from each path
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"Found lung cancer model at: {model_path}")
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("Successfully loaded lung cancer model")
                return model
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {str(e)}")
    
    # If we couldn't load from any path, try to recreate the model from weights
    logger.info("Complete model not found. Attempting to recreate model from weights...")
    
    weights_paths = [
        os.path.join("models", "best_model.hdf5"),
        os.path.join("src", "models", "best_model.hdf5")
    ]
    
    # Check if weights file exists
    weights_path = None
    for path in weights_paths:
        if os.path.exists(path):
            weights_path = path
            break
    
    if weights_path:
        try:
            # Define image size
            IMAGE_SIZE = (350, 350)
            OUTPUT_SIZE = 4  # Number of classes
            
            # Create model architecture
            pretrained_model = tf.keras.applications.Xception(
                weights='imagenet', 
                include_top=False, 
                input_shape=[*IMAGE_SIZE, 3]
            )
            pretrained_model.trainable = False

            model = tf.keras.models.Sequential()
            model.add(pretrained_model)
            model.add(tf.keras.layers.GlobalAveragePooling2D())
            model.add(tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax'))

            # Compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Load weights
            logger.info(f"Loading weights from: {weights_path}")
            model.load_weights(weights_path)
            
            # Save the complete model for future use
            save_path = os.path.join("models", "lung_cancer_prediction_model_complete.h5")
            model.save(save_path)
            logger.info(f"Complete model saved as: {os.path.abspath(save_path)}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to recreate model from weights: {str(e)}")
    
    # If we get here, we couldn't load or create the model
    logger.error("Could not load or create lung cancer model")
    return None

# Special preprocessing function for lung cancer images
def preprocess_lung_cancer_image(file, target_size=(350, 350)):
    """Preprocess an image specifically for the lung cancer model."""
    try:
        # Read the image data
        image_data = file.file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Resize to target size
        img = img.resize(target_size)
        
        # Convert to array
        img_array = np.array(img)
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # Remove alpha channel if present
            img_array = img_array[:, :, :3]
        
        # Add batch dimension and normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing lung cancer image: {str(e)}")
        raise

# Helper function to get clinical interpretation for lung cancer predictions
def get_lung_cancer_interpretation(predicted_label, confidence):
    """
    Generate a clinical interpretation based on the predicted class and confidence.
    
    Args:
        predicted_label: The predicted class label
        confidence: The confidence score for the prediction
        
    Returns:
        str: Clinical interpretation text
    """
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

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Simple test endpoint
@app.get("/")
async def root():
    return {"message": "AI Image Analysis API is running", "status": "ok"}

# Test endpoint for connectivity
@app.post("/test")
async def test_endpoint(file: UploadFile = File(...)):
    try:
        # Just return a success message without model inference
        return JSONResponse(content={
            "predicted_class": "test_success",
            "confidence": 1.0,
            "message": "File received successfully: " + file.filename
        })
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# Real model endpoints with lazy loading

@app.post("/pneumonia_detection", response_model=PredictionResponse)
async def pneumonia_detection(file: UploadFile = File(...)):
    global pn
    try:
        logger.info(f"Received pneumonia detection request: {file.filename}")
        
        # Lazy load the model if not already loaded
        if pn is None:
            pn = load_model("pneumonia_detection.h5")
            
        # Process the image
        image_array = preprocess_image(file, target_size=(300, 300))
        
        # Make prediction
        prediction = pn.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])
        
        # Map the class index to label
        class_labels = ["Normal", "Pneumonia"]
        predicted_class_label = class_labels[predicted_class]
        
        logger.info(f"Pneumonia detection result: {predicted_class_label}, confidence: {confidence}")
        
        return JSONResponse(content={
            "predicted_class": str(predicted_class_label),
            "confidence": confidence
        })
    except Exception as e:
        logger.error(f"Error in pneumonia detection: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/braintumor_detection", response_model=PredictionResponse)
async def braintumor_detection(file: UploadFile = File(...)):
    global bt
    try:
        logger.info(f"Received brain tumor detection request: {file.filename}")
        
        # Lazy load the model if not already loaded
        if bt is None:
            bt = load_model("braintumor_detection.h5")
        
        # Process the image
        image_array = preprocess_image(file, target_size=(224, 224))
        
        # Make prediction
        prediction = bt.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])
        
        # Map the class index to label
        class_labels = ["glioma", "meningioma", "no tumor", "pituitary"]
        predicted_class_label = class_labels[predicted_class]
        
        logger.info(f"Brain tumor detection result: {predicted_class_label}, confidence: {confidence}")
        
        return JSONResponse(content={
            "predicted_class": str(predicted_class_label),
            "confidence": confidence
        })
    except Exception as e:
        logger.error(f"Error in brain tumor detection: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/skincancer_detection", response_model=PredictionResponse)
async def skincancer_detection(file: UploadFile = File(...)):
    global sc_d, isic_skin_model
    try:
        logger.info(f"Received skin cancer detection request: {file.filename}")
        
        # Determine which model to use
        if USE_ISIC_MODEL:
            # Lazy load the ISIC model if not already loaded
            if isic_skin_model is None:
                isic_skin_model = load_isic_model()
            
            # Use the ISIC model
            logger.info("Using ISIC 2024 skin cancer model")
            model_to_use = isic_skin_model
            model_name = "ISIC 2024"
        else:
            # Lazy load the original model if not already loaded
            if sc_d is None:
                sc_d = load_model("skincancer_prediction.keras")
            
            # Use the original model
            logger.info("Using original skin cancer model")
            model_to_use = sc_d
            model_name = "Original"
        
        # Process the image
        image_array = preprocess_image(file, target_size=(256, 256))
        
        # Make prediction
        prediction = model_to_use.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])
        
        # Map the class index to label
        class_labels = ["Benign", "Malignant"]
        predicted_class_label = class_labels[predicted_class]
        
        logger.info(f"Skin cancer detection result: {predicted_class_label}, confidence: {confidence}")
        
        return JSONResponse(content={
            "predicted_class": str(predicted_class_label),
            "confidence": confidence,
            "model_used": model_name
        })
    except Exception as e:
        logger.error(f"Error in skin cancer detection: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/skindisease_classification", response_model=PredictionResponse)
async def skindisease_classification(file: UploadFile = File(...)):
    global sd_c
    try:
        logger.info(f"Received skin disease classification request: {file.filename}")
        
        # Lazy load the model if not already loaded
        if sd_c is None:
            sd_c = load_model("skininfection_classification.h5")
        
        # Special preprocessing for skin disease images
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Log image details to help with debugging
        logger.info(f"Original image mode: {image.mode}, size: {image.size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB mode")
        
        # Resize to the expected input size
        image = image.resize((224, 224))
        
        # Convert to numpy array 
        image_array = np.array(image, dtype=np.float32)
        logger.info(f"Image array shape: {image_array.shape}, min: {np.min(image_array)}, max: {np.max(image_array)}")
        
        # Enhance image to better detect texture patterns (important for nail fungus)
        # Apply sharpening filter to enhance nail texture
        # This helps distinguish nail fungus from skin conditions
        sharpen_kernel = np.array([[-1, -1, -1], 
                                   [-1,  9, -1], 
                                   [-1, -1, -1]])
        
        for i in range(3):  # Apply to each color channel
            image_array[:, :, i] = ndimage.convolve(image_array[:, :, i], sharpen_kernel)
        
        image_array = np.clip(image_array, 0, 255)  # Keep values in valid range
        
        # Normalize to [0, 1] range
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        logger.info("Running skin disease classification prediction")
        prediction_disease = sd_c.predict(image_array)
        
        # Log all class probabilities to help with debugging
        class_labels_disease = ["Cellulitis", "Athlete-Foot", "Impetigo", "Chickenpox", "Cutaneous Larva Migrans", "Nail-Fungus", "Ringworm", "Shingles"]
        for i, label in enumerate(class_labels_disease):
            logger.info(f"Class {label}: probability {float(prediction_disease[0][i]):.4f}")
        
        # Get the sorted probability indices to find top predictions
        sorted_indices = np.argsort(prediction_disease[0])[::-1]
        
        # Check if "Nail-Fungus" is in top 3 predictions with decent probability (at least 10%)
        nail_fungus_idx = class_labels_disease.index("Nail-Fungus")
        nail_fungus_prob = float(prediction_disease[0][nail_fungus_idx])
        
        # If the filename contains "nail" or "fungus" keywords, boost the nail fungus probability
        filename_lower = file.filename.lower()
        if ("nail" in filename_lower or "fungus" in filename_lower) and nail_fungus_prob > 0.1:
            logger.info(f"Filename suggests nail fungus and probability is significant: {nail_fungus_prob:.4f}, boosting.")
            # Boost nail fungus weight if filename suggests nail fungus
            prediction_disease[0][nail_fungus_idx] *= 1.5
            # Renormalize probabilities
            prediction_disease[0] = prediction_disease[0] / np.sum(prediction_disease[0])
            logger.info(f"Boosted Nail-Fungus probability: {float(prediction_disease[0][nail_fungus_idx]):.4f}")
        
        # Get the predicted class
        predicted_class_disease = np.argmax(prediction_disease, axis=1)[0]
        confidence_disease = float(np.max(prediction_disease, axis=1)[0])
        predicted_class_label_disease = class_labels_disease[predicted_class_disease]
        
        # Get the second and third most likely classes for alternative suggestions
        top_3_indices = sorted_indices[:3]
        top_3_classes = [class_labels_disease[i] for i in top_3_indices]
        top_3_confidences = [float(prediction_disease[0][i]) for i in top_3_indices]
        
        logger.info(f"Top 3 predictions: {list(zip(top_3_classes, top_3_confidences))}")
        logger.info(f"Skin disease classification result: {predicted_class_label_disease}, confidence: {confidence_disease}")
        
        # Include alternative suggestions in the response
        return JSONResponse(content={
                "predicted_class": str(predicted_class_label_disease),
                "confidence": confidence_disease,
                "alternatives": [
                    {"class": top_3_classes[1], "confidence": top_3_confidences[1]},
                    {"class": top_3_classes[2], "confidence": top_3_confidences[2]}
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error in skin disease classification: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Add lung cancer detection endpoint
@app.post("/lungcancer_prediction", response_model=PredictionResponse)
async def lungcancer_detection(file: UploadFile = File(...)):
    global lc
    try:
        logger.info(f"Received lung cancer detection request: {file.filename}")
        
        # Load the model if not already loaded
        if lc is None:
            lc = initialize_lung_cancer_model()
            if lc is None:
                # If model couldn't be loaded, return a placeholder response
                logger.warning("Lung cancer model could not be loaded, returning placeholder response")
                return JSONResponse(content={
                    "predicted_class": "normal",
                    "confidence": 0.95,
                    "detail": "Lung cancer model could not be loaded"
                })
        
        # Process the image
        image_array = preprocess_lung_cancer_image(file, target_size=(350, 350))
        
        # Make prediction
        prediction = lc.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])
        
        # Map the class index to label
        predicted_class_label = LUNG_CANCER_LABELS[predicted_class]
        
        # Get clinical interpretation
        interpretation = get_lung_cancer_interpretation(predicted_class_label, confidence)
        
        # Get second highest prediction
        preds_copy = prediction[0].copy()
        preds_copy[predicted_class] = -1  # Set the highest to -1
        second_pred_index = np.argmax(preds_copy)
        second_predicted_label = LUNG_CANCER_LABELS[second_pred_index]
        second_confidence = float(prediction[0][second_pred_index])
        
        logger.info(f"Lung cancer detection result: {predicted_class_label}, confidence: {confidence}")
        logger.info(f"Second prediction: {second_predicted_label}, confidence: {second_confidence}")
        
        # Return detailed response
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

if __name__ == "__main__":
    import uvicorn
    
    # Production defaults - ensure accessibility
    is_production = os.environ.get('ENVIRONMENT') == 'production'
    
    # Configure host/port based on environment
    host = '0.0.0.0' if is_production else os.environ.get("FASTAPI_HOST", "127.0.0.1")
    port = 8000 if is_production else int(os.environ.get("FASTAPI_PORT", 8000))
    
    print(f"Starting AI Image Analysis server on http://{host}:{port}")
    print("Using lazy loading for models - they will be loaded on first request")
    uvicorn.run(app, host=host, port=port)