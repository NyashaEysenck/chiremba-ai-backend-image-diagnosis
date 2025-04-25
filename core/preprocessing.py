# Image preprocessing functions
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger("uvicorn")

def preprocess_image(file, target_size=(224, 224)):
    """
    Preprocess an image file for model prediction.
    """
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
    """Preprocess an image specifically for the lung cancer model."""
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
