import joblib
from app.features import extract_features

MODEL_PATH = "models/voice_model.pkl"

model = joblib.load(MODEL_PATH)

def predict_voice(file_path: str):
    features = extract_features(file_path)
    proba = model.predict_proba([features])[0]

    return {
        "human_probability": float(proba[1]),
        "ai_probability": float(proba[0])
    }
