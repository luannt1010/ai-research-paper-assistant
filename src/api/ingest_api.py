from typing import List, Literal
from pathlib import Path
from fastapi import HTTPException, APIRouter, Depends, Request
from pydantic import BaseModel, Field


router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

def get_service(request: Request):
    return request.app.state.ingest_pipeline

class PathInput(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=10)

class FileResult(BaseModel):
    path: str
    status: Literal["success", "failed"]
    num_chunks: int | None = None
    error: str | None = None

class PathOutput(BaseModel):
    message: str
    results: List[FileResult]

class DirInput(BaseModel):
    dir_path: str

class DirOutput(BaseModel):
    message: str
    directory: str
    status: Literal["success", "failed"]
    total_chunks: int | None = None
    error: str | None = None

@router.post("/ingest_file", response_model=PathOutput)
def ingest_file(request: PathInput, service=Depends(get_service)):
    results = []
    for path in request.paths:
        path = Path(path)
        try:
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Path not found: {path}")
            if not path.is_file():
                raise HTTPException(status_code=400, detail=f"Path is not file: {path}")
            if path.suffix.lower() != ".pdf":
                raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {path}")

            num_chunks = service.ingest_pdf(path)
            results.append({
                "path": str(path),
                "status": "success",
                "num_chunks": num_chunks
            })
        except Exception as e:
            results.append({
                "path": str(path),
                "status": "failed",
                "error": str(e)
            })
    return {
        "message": "File ingestion completed",
        "results": results
    }

@router.post("/ingest_dir", response_model=DirOutput)
def ingest_dir(request: DirInput, service=Depends(get_service)):
    dir_path = Path(request.dir_path)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {dir_path}")
    try:
        total_chunks = service.ingest_dir(dir_path)
    except Exception as e:
        return {
            "message": "Directory ingestion failed",
            "directory": str(dir_path),
            "status": "failed",
            "error": str(e)
        }
    return {
        "message": "Directory ingestion completed",
        "directory": str(dir_path),
        "status": "success",
        "total_chunks": total_chunks
    }