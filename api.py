from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Student Performance Prediction API",
    version="1.0.0",
    description="REST API to predict student pass probability and academic risk."
)

# 1. Load the trained pipeline
MODEL_PATH = "student_performance_logistic_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model at {MODEL_PATH}: {e}")

# 2. Define input schema using Pydantic
class StudentData(BaseModel):
    Age: float = Field(..., example=16.0)
    Math: float = Field(..., ge=0, le=100, example=65.0)
    Science: float = Field(..., ge=0, le=100, example=70.0)
    English: float = Field(..., ge=0, le=100, example=58.0)
    Social: float = Field(..., ge=0, le=100, example=62.0)
    Attendance: float = Field(..., ge=0, le=100, example=82.5)
    Enrollment_Year: int = Field(..., example=2023)
    Enrollment_Month: int = Field(..., ge=1, le=12, example=3)
    Gender: str = Field(..., example="Female")
    City: str = Field(..., example="Kathmandu")

# 3. Health check endpoint
@app.get("/")
def home():
    return {"status": "active", "message": "Student Performance API is running."}

# 4. Prediction endpoint
@app.post("/predict")
def predict_outcome(student: StudentData):
    try:
        # Convert request body into DataFrame matching model input format
        input_df = pd.DataFrame([student.model_dump()])

        # Calculate probability and predicted class
        prob_pass = float(model.predict_proba(input_df)[0][1])
        prediction = int(prob_pass >= 0.50)
        verdict = "Pass" if prediction == 1 else "At Risk / Fail"

        return {
            "prediction": prediction,
            "verdict": verdict,
            "pass_probability": round(prob_pass, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
