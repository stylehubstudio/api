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
AI_THRESHOLD = 0.30

# HARD LIMITS (for hackathon tester)
MAX_BASE64_LEN = 1_500_000     # ~1.1 MB raw
MAX_DURATION_SEC = 5.0         # seconds
TARGET_SR = 16000              # faster decode
MIN_AUDIO_SEC = 1.0
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
def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Header(None),
):
    key = x_api_key or api_key

    if key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    if key != API_KEY:
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


# ---------------- AUDIO DECODER (FAST & SAFE) ----------------
def decode_audio_base64(audio_b64: str):
    try:
        # Sanitize base64 (tester may add spaces/newlines)
        audio_b64 = audio_b64.strip().replace("\n", "").replace(" ", "")

        # 🔥 Fail fast on oversized payloads
        if len(audio_b64) > MAX_BASE64_LEN:
            raise HTTPException(
                status_code=413,
                detail="Audio too large. Use ≤5 seconds MP3."
            )

        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 🔥 FAST decode (resample + duration cap)
        y, sr = librosa.load(
            tmp_path,
            sr=TARGET_SR,
            mono=True,
            offset=0.0,
            duration=MAX_DURATION_SEC
        )

        os.remove(tmp_path)

        if y is None or len(y) < int(sr * MIN_AUDIO_SEC):
            raise HTTPException(status_code=400, detail="Audio too short or silent")

        return y, sr

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio input: {e}")


# ---------------- API ENDPOINT ----------------
@app.post("/analyze", response_model=VoiceResult)
def analyze_voice(
    request: VoiceRequest,
    _: None = Depends(verify_api_key)
):
    # Decode audio
    y, sr = decode_audio_base64(request.audioBase64)

    # FAST feature extraction
    features = extract_features_fast(y, sr)

    # Safety check (prevents 500s)
    if len(features) != model.n_features_in_:
        raise HTTPException(
            status_code=500,
            detail="Model feature mismatch"
        )

    # Predict
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
        "model_version": MODEL_VERSION,
        "features_expected": model.n_features_in_
    }
