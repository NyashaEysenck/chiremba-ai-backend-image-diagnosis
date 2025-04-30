# Chiremba AI Backend Image Diagnosis Service

A comprehensive medical image analysis service that provides AI-powered diagnosis for multiple medical conditions.

## Features

- Multi-model medical image analysis system
- Support for multiple conditions:
  - Brain tumor detection and classification
  - Lung cancer detection
  - Pneumonia analysis
  - Skin disease classification
- Advanced image preprocessing pipeline
- High-performance API endpoints

## Technology Stack

- FastAPI for high-performance API endpoints
- TensorFlow 2.17 for model inference
- OpenCV for image processing
- Uvicorn for ASGI server

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model files are present in the `models/` directory:
   - `brain_tumor_model.h5`
   - `lung_cancer_model.h5`
   - `pneumonia_model.h5`

3. Start the service:
```bash
python main.py
```

## API Endpoints

- `GET /`: Service information and status
- `GET /health`: Health check endpoint
- `POST /analyze/brain`: Brain tumor analysis endpoint
- `POST /analyze/lung`: Lung cancer detection endpoint
- `POST /analyze/pneumonia`: Pneumonia analysis endpoint

All analysis endpoints accept:
- Image file upload (DICOM, PNG, JPEG formats)
- Optional parameters for analysis configuration

## Environment Variables

- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Custom path to model directory
- `LOG_LEVEL`: Logging level (default: INFO)

## Image Processing Pipeline

1. Image validation and format checking
2. Preprocessing and normalization
3. Model-specific transformations
4. Inference and result aggregation
5. Confidence score calculation

## Dependencies

See `requirements.txt` for full list of dependencies.
