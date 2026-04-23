import logging
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from backend.services.inference import ask_mcq

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)


# ==========================
# REQUEST MODEL
# ==========================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Medical question")

    options: Dict[str, str]

    @validator("options")
    def validate_options(cls, v):
        required = {"A", "B", "C", "D"}
        if set(v.keys()) != required:
            raise ValueError("Options must contain A, B, C, D")

        for key, value in v.items():
            if not value.strip():
                raise ValueError(f"Option {key} is empty")

        return v


# ==========================
# RESPONSE MODEL
# ==========================

class AskResponse(BaseModel):
    answer: str
    explanation: str
    raw_output: str


# ==========================
# ROUTE
# ==========================

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = ask_mcq(req.question, req.options)
        return AskResponse(**result)

    except Exception as e:
        logger.exception("/chat/ask failed")
        raise HTTPException(status_code=500, detail=str(e))