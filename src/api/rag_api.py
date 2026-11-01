from typing import Any, Literal
from fastapi import HTTPException, APIRouter, Request, Depends
from pydantic import BaseModel, Field

router = APIRouter(prefix="/generation", tags=["Generate"])

def get_service(request: Request):
    return request.app.state.rag_pipeline

class Input(BaseModel):
    question: str = Field(min_length=1)

class GenerateResult(BaseModel):
    message: str
    status: Literal["success", "failed"]
    answer: Any = None
    error: str | None = None

class Output(BaseModel):
    question: str
    results: GenerateResult

@router.post("/generate", response_model=Output)
def generate(request: Input, service=Depends(get_service)):
    question = request.question
    try:
        answer = service.generate(question)
        return {
            "question": question,
            "results": {
                "message": f"Generate for question successfully",
                "status": "success",
                "answer": answer,
                "error": None
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG generation failed {str(e)}"
        )