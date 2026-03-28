from pydantic import BaseModel, Field
from typing import List

class AnalysisResponse(BaseModel):
    audience: str = Field(..., description="A detailed description of the target audience.")
    key_messages: List[str] = Field(..., description="A list of the core marketing messages.")
    tone: str = Field(..., description="The suggested tone of voice for the campaign.")
    channels: List[str] = Field(..., description="A list of recommended marketing channels.")
    risks: List[str] = Field(..., description="A list of potential risks or challenges for the campaign.")