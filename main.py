from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from endpoints.pneumonia import router as pneumonia_router
from endpoints.braintumor import router as braintumor_router
from endpoints.skindisease import router as skindisease_router
from endpoints.lungcancer import router as lungcancer_router
import logging
from fastapi.responses import JSONResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5000",
        "https://chiremba-ai-frontend-production.up.railway.app",
        "https://chiremba-full-stack-160376271578.us-central1.run.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Health check endpoint (required for Cloud Run)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Register routers
app.include_router(pneumonia_router)
app.include_router(braintumor_router)
app.include_router(skindisease_router)
app.include_router(lungcancer_router)

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

if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv

    load_dotenv()
    
    port = int(os.environ.get("PORT", 8080))  # Cloud Run sets PORT=8080
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)