from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, HttpUrl
from typing import Optional
import tempfile
import requests
import os
import joblib
import numpy as np
import librosa

from features import extract_features_fast   # FAST MODE


# ================= CONFIG =================
API_KEY = "VOICEAUTH_AI_2026"
MODEL_PATH = "models/voice_model.pkl"
MODEL_VERSION = "1.0-fast"
MAX_DURATION = 8          # seconds (reduced)
AI_THRESHOLD = 0.30
# =========================================


# ---------------- APP INIT ----------------
app = FastAPI(
    title="VoiceAuth AI",
    description="Fast biologically inspired human vs AI voice authentication API",
    version=MODEL_VERSION
)

# Load model ONCE
model = joblib.load(MODEL_PATH)


# ---------------- WARMUP ----------------
@app.on_event("startup")
def warmup():
    dummy = np.zeros((1, model.n_features_in_))
    model.predict_proba(dummy)


# ---------------- AUTH ----------------
def verify_api_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization")

    if authorization.replace("Bearer ", "").strip() != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------- SCHEMAS ----------------
class VoiceRequest(BaseModel):
    audio_url: HttpUrl
    note: Optional[str] = None


class VoiceResult(BaseModel):
    verdict: str
    human_probability: float
    ai_probability: float
    model_version: str


# ---------------- AUDIO FETCH (FAST) ----------------
def load_audio_from_url(url: str):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            raise Exception()

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(r.content)
        tmp.close()

        y, sr = librosa.load(
            tmp.name,
            sr=None,
            mono=True,
            duration=MAX_DURATION
        )

        os.remove(tmp.name)
        return y, sr

    except Exception:
        raise HTTPException(status_code=400, detail="Failed to load audio")


# ---------------- API ENDPOINT ----------------
@app.post("/analyze", response_model=VoiceResult)
def analyze_voice(
    request: VoiceRequest,
    _: None = Depends(verify_api_key)
):
    y, sr = load_audio_from_url(request.audio_url)

    # FAST FEATURE EXTRACTION
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
