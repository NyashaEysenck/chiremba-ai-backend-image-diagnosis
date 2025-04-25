from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints.pneumonia import router as pneumonia_router
from endpoints.braintumor import router as braintumor_router
from endpoints.skindisease import router as skindisease_router
from endpoints.lungcancer import router as lungcancer_router

app = FastAPI()

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

# Register routers here
app.include_router(pneumonia_router)
app.include_router(braintumor_router)
app.include_router(skindisease_router)
app.include_router(lungcancer_router)

@app.get("/")
async def root():
    return {"message": "AI Image Analysis API is running", "status": "ok"}

@app.post("/test")
async def test_endpoint(file: bytes = None):
    return {"predicted_class": "test_success", "confidence": 1.0, "message": "File received successfully"}

if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv

    load_dotenv() 

    port = int(os.environ.get("PORT", "8080"))
    host = "0.0.0.0"
    print(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
