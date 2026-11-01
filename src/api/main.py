from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from src.pipeline import PipelineFactory
from src.api.ingest_api import router as ingest_router
from src.api.rag_api import router as rag_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    factory = PipelineFactory()
    rag = factory.get_rag_pipeline()
    ingestor = factory.get_ingestion_pipeline()
    print("Model started")
    app.state.factory = factory
    app.state.rag_pipeline = rag
    app.state.ingest_pipeline = ingestor
    yield
    factory.close()
    print("Model closed")


app = FastAPI(title="AI Research Assistant API", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(rag_router)

@app.get("/health")
def health(request: Request):
    return {"message": "AI Research Assistant API",
            "status": "healthy",
            "rag": request.app.state.rag_pipeline is not None,
            "ingestor": request.app.state.ingest_pipeline is not None}