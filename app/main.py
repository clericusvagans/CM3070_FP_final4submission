# app/main.py

from fastapi import FastAPI

from app.routes import (
    ingestion as ingestion_routes,
    documents as documents_routes,
    search as search_routes,
    ui as ui_routes,
)


app = FastAPI(
    title="Legal Document AI Backend",
    version="1.0.0",
    description="Pipeline-driven OCR + NLP backend for structured legal data extraction.",
)

app.include_router(ingestion_routes.router)
app.include_router(documents_routes.router)
app.include_router(search_routes.router)
app.include_router(ui_routes.router)


@app.get("/healthz")
async def health_check():
    return {"status": "ok"}