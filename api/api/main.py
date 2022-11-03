from fastapi import FastAPI
import joblib
from pydantic import BaseModel, validator
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).parent

class Digits(BaseModel):
    """8x8 image of integer pixels in the range 0..16"""
    data: list[list[int]]

    @validator("data")
    def check_input_data(cls, v):
        constraints = (
            len(l) == 64 
            and max(l) <= 16
            and min(l) >= 0
            for l in v
        )
        assert all(constraints)
        return v

    class Config:
        schema_extra = {
            "example": {
                "data": [[0] * 64],
            }
        }


app = FastAPI()

model: RandomForestClassifier  = joblib.load(ROOT.parent  / "model/rf_model.joblib")

@app.post("/predict/")
async def model_predict(input: Digits):
    pred: list[int] = model.predict(input.data)
    prob: list[list[float]] = model.predict_proba(input.data)
    return {"y_pred": pred.tolist(), "pos_prob": prob.tolist()}
