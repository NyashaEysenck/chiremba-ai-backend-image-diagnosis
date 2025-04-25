# Model loading and initialization logic
import os
import tensorflow as tf
import logging
import numpy as np
from PIL import Image
import io

logger = logging.getLogger("uvicorn")

# Get absolute project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Global model variables (initialized as None)
pn = None
sc_d = None
sd_c = None
bt = None
lc = None  # Lung cancer model

LUNG_CANCER_LABELS = [
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa',
    'normal',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
]

def preprocess_image(file, target_size=(224, 224)):
    try:
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        logger.info(f"Original image mode: {image.mode}, size: {image.size}")
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB mode")
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32)
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def preprocess_lung_cancer_image(file, target_size=(350, 350)):
    try:
        image_data = file.file.read()
        img = Image.open(io.BytesIO(image_data))
        img = img.resize(target_size)
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing lung cancer image: {str(e)}")
        raise

def load_model(model_name):
    logger.info(f"Loading model: {model_name}")
    try:
        model_path = os.path.join(PROJECT_ROOT, "models", model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_ROOT, "src", "models", model_name)
            logger.info(f"Looking for model in: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path} or src/models/{model_name}")
        custom_objects = {
            'InputLayer': tf.keras.layers.InputLayer
        }
        try:
            logger.info(f"Attempting to load model directly: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model: {model_name}")
            return model
        except Exception as first_error:
            logger.warning(f"First attempt to load model failed: {str(first_error)}")
            import h5py
            logger.info(f"Attempting to load model with h5py: {model_path}")
            with h5py.File(model_path, 'r+') as h5file:
                if 'model_weights' in h5file:
                    for layer_name in h5file['model_weights']:
                        if 'batch_shape' in h5file['model_weights'][layer_name].attrs:
                            batch_shape = h5file['model_weights'][layer_name].attrs['batch_shape']
                            h5file['model_weights'][layer_name].attrs['batch_input_shape'] = batch_shape
                            logger.info(f"Converted batch_shape to batch_input_shape for layer: {layer_name}")
            logger.info(f"Attempting to load model after h5py modification: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"Successfully loaded model after h5py modification: {model_name}")
            return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def initialize_lung_cancer_model():
    """Initialize and load the lung cancer prediction model."""
    logger.info("Initializing lung cancer model")
    model_paths = [
        os.path.join(PROJECT_ROOT, "models", "lung_cancer_prediction_model_complete.h5"),
        os.path.join(PROJECT_ROOT, "src", "models", "lung_cancer_prediction_model_complete.h5"),
        os.path.join(PROJECT_ROOT, "models", "lungcancer_prediction.keras"),
        os.path.join(PROJECT_ROOT, "src", "models", "lungcancer_prediction.keras")
    ]
    for model_path in model_paths:
        if os.path.exists(model_path):
            logger.info(f"Found lung cancer model at: {model_path}")
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("Successfully loaded lung cancer model")
                return model
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {str(e)}")
    logger.info("Complete model not found. Attempting to recreate model from weights...")
    weights_paths = [
        os.path.join(PROJECT_ROOT, "models", "best_model.hdf5"),
        os.path.join(PROJECT_ROOT, "src", "models", "best_model.hdf5")
    ]
    weights_path = None
    for path in weights_paths:
        if os.path.exists(path):
            weights_path = path
            break
    if weights_path:
        try:
            IMAGE_SIZE = (350, 350)
            OUTPUT_SIZE = 4
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
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            logger.info(f"Loading weights from: {weights_path}")
            model.load_weights(weights_path)
            save_path = os.path.join(PROJECT_ROOT, "models", "lung_cancer_prediction_model_complete.h5")
            model.save(save_path)
            logger.info(f"Complete model saved as: {os.path.abspath(save_path)}")
            return model
        except Exception as e:
            logger.error(f"Failed to recreate model from weights: {str(e)}")
    logger.error("Could not load or create lung cancer model")
    return None

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
                interpretation += "\nNote: The confidence is low. Further diagnostic workup may be warranted.\n"
        return interpretation
    else:
        return "No clinical interpretation available for this class."
