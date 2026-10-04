"""
app.py
------
The web server. This is the ONLY file your Next.js frontend talks to.

It exposes one endpoint: POST /predict
You send it an image file, it runs detection + classification, and
sends back JSON describing the banana's ripeness.

Run locally with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Then visit http://localhost:8000/docs to test it in the browser
without needing the frontend at all (FastAPI auto-generates this).
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import detect
import classification

app = FastAPI(title="Banana Ripeness API")

# ---------------------------------------------------------------------------
# CORS: without this, a browser running your Next.js app on a different
# port/domain will be BLOCKED from calling this API. Restrict allow_origins
# to your real frontend URL(s) once you deploy — "*" is fine for local dev.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g. ["http://localhost:3000", "https://yourapp.com"]
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Simple endpoint to confirm the server is alive."""
    return {"status": "ok", "message": "Banana ripeness API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns ripeness classification.
 
    Expects a multipart/form-data request with a field named "file".
    """
    # Basic validation: make sure they actually uploaded an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
 
    image_bytes = await file.read()
 
    try:
        detections, annotated_image_base64 = detect.run_detection(image_bytes)
        result = classification.classify(detections)
        result["annotated_image"] = annotated_image_base64
    except Exception as e:
        # Don't leak internal errors to the client in production;
        # this is fine for development so you can see what broke.
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
 
    return result

