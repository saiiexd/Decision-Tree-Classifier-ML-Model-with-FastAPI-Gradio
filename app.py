from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(
    title="Student Prediction API",
    description="Backend for Student Pass/Fail prediction using Decision Tree",
    version="2.0"
)

# --- Schemas ---
class StudentInput(BaseModel):
    study_hours: float = Field(..., ge=0, le=12, description="Hours studied per day (0-12)")
    attendance: float = Field(..., ge=0, le=100, description="Attendance percentage (0-100)")
    previous_score: float = Field(..., ge=0, le=100, description="Previous exam score (0-100)")

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    feature_importance: dict

# --- Load Model & Metadata ---
def load_assets():
    if not os.path.exists("model.pkl") or not os.path.exists("metadata.pkl"):
        print("Required assets missing. Running training...")
        import train_model
        train_model.train()
    
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, metadata

model, metadata = load_assets()

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Health check endpoint to verify API status."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentInput):
    """
    Takes student data and returns prediction with confidence score.
    """
    try:
        # Prepare input for model
        features = np.array([[data.study_hours, data.attendance, data.previous_score]])
        
        # Binary prediction
        pred_bin = model.predict(features)[0]
        label = "Pass" if pred_bin == 1 else "Fail"
        
        # Confidence score (probability)
        probs = model.predict_proba(features)[0]
        confidence = float(np.max(probs))
        
        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "feature_importance": metadata["feature_importance"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
