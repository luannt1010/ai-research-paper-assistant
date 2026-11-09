from typing import List, Literal
from pathlib import Path
from fastapi import HTTPException, APIRouter, Depends, Request
from pydantic import BaseModel, Field


router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

def get_ingest_service(request: Request):
    return request.app.state.ingest_pipeline

def get_rag_service(request: Request):
    return request.app.state.rag_pipeline

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
def ingest_file(request: PathInput, ingest_service=Depends(get_ingest_service), rag_service=Depends(get_rag_service)):
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

            num_chunks = ingest_service.ingest_pdf(path)
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
    is_needed_refresh = any(result["status"]=="success" and result["num_chunks"]>0 for result in results)
    if is_needed_refresh:
        rag_service.refresh_index()
    return {
        "message": "File ingestion completed",
        "results": results
    }

@router.post("/ingest_dir", response_model=DirOutput)
def ingest_dir(request: DirInput, ingest_service=Depends(get_ingest_service), rag_service=Depends(get_rag_service)):
    dir_path = Path(request.dir_path)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {dir_path}")
    try:
        total_chunks = ingest_service.ingest_dir(dir_path)
        if total_chunks > 0:
            rag_service.refresh_index()
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