from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import Optional
import base64
import tempfile
import os
import joblib
import numpy as np
import librosa

from app.features import extract_features_fast


# ================= CONFIG =================
API_KEY = "VOICEAUTH_AI_2026"
MODEL_PATH = "models/voice_model.pkl"
MODEL_VERSION = "1.0-fast"
MAX_DURATION = 8      # seconds
AI_THRESHOLD = 0.30
# =========================================


# ---------------- APP INIT ----------------
app = FastAPI(
    title="VoiceAuth AI",
    description="Fast AI-generated voice detection API",
    version=MODEL_VERSION
)

model = joblib.load(MODEL_PATH)


# ---------------- WARMUP ----------------
@app.on_event("startup")
def warmup():
    dummy = np.zeros((1, model.n_features_in_))
    model.predict_proba(dummy)


# ---------------- AUTH ----------------
def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing x-api-key")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------- SCHEMAS ----------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


class VoiceResult(BaseModel):
    verdict: str
    human_probability: float
    ai_probability: float
    model_version: str


# ---------------- AUDIO DECODER ----------------
def decode_audio_base64(audio_b64: str):
    try:
        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(
            tmp_path,
            sr=None,
            mono=True,
            duration=MAX_DURATION
        )

        os.remove(tmp_path)
        return y, sr

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio_base64 input")


# ---------------- API ENDPOINT ----------------
@app.post("/analyze", response_model=VoiceResult)
def analyze_voice(
    request: VoiceRequest,
    _: None = Depends(verify_api_key)
):
    y, sr = decode_audio_base64(request.audio_base64)

    features = extract_features_fast(y, sr)
    probs = model.predict_proba([features])[0]

    human_prob = float(probs[1])
    ai_prob = float(probs[0])

    verdict = "AI" if human_prob <= AI_THRESHOLD else "HUMAN"

    return {
        "verdict": verdict,
        "human_probability": round(human_prob, 3),
        "ai_probability": round(ai_prob, 3),
        "model_version": MODEL_VERSION
    }


# ---------------- HEALTH ----------------
@app.get("/")
def health():
    return {
        "status": "VoiceAuth AI running",
        "model_version": MODEL_VERSION
    }
