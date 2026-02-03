from pydantic import BaseModel

class VoiceResult(BaseModel):
    human_probability: float
    ai_probability: float
    verdict: str
