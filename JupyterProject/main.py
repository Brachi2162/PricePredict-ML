from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# 1. טעינת המודל והסקיילר בראש הקובץ לביצועים מקסימליים
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

# 2. הגדרת CORS - פתיחת השרת לבקשות מ-HTML חיצוני
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# הגדרת שמות המשתנים לפי סדר האימון
features = ['median_income', 'housing_median_age', 'total_rooms']


# 3. יצירת מחלקה (BaseModel) עבור נתוני הקלט
class HouseData(BaseModel):
    median_income: float
    housing_median_age: float
    total_rooms: float


@app.get("/")
def home():
    return {"status": "success", "message": "API is online"}


# --- אתגר: נקודת קצה נוספת (/info) להצגת המקדמים ---
@app.get("/info")
def get_info():
    # שליפת המקדמים מהמודל
    coeffs = model.coef_
    importance = {features[i]: float(coeffs[i]) for i in range(len(features))}

    return {
        "model_coefficients": importance,
        "most_important": features[np.argmax(np.abs(coeffs))],
        "note": "A higher coefficient means this feature has more weight in the prediction."
    }


# 4. לוגיקת ה-Predict המלאה
@app.post("/predict")
def predict(data: HouseData):
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)

    price_usd = float(prediction[0])
    # הוספת חישוב לשקלים
    price_ils = price_usd * 3.7

    return {
        "predicted_price_usd": round(price_usd, 2),
        "predicted_price_ils": round(price_ils, 2),
        "status": "success"
    }