from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

from app.model import predict_voice
from app.schemas import VoiceResult

app = FastAPI(title="Human vs AI Voice Detection API")

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze", response_model=VoiceResult)
async def analyze_voice(file: UploadFile = File(...)):

    # Save uploaded file
    temp_name = f"{uuid.uuid4()}.wav"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict
    result = predict_voice(temp_path)

    # Clean up
    os.remove(temp_path)

    # Verdict logic
    if result["human_probability"] >= 0.65:
        verdict = "HUMAN"
    elif result["human_probability"] <= 0.35:
        verdict = "AI"
    else:
        verdict = "UNCERTAIN"

    return VoiceResult(
        human_probability=result["human_probability"],
        ai_probability=result["ai_probability"],
        verdict=verdict
    )
