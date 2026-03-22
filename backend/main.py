from __future__ import annotations

from fastapi import FastAPI

from backend.api.routes.predictions import router as predictions_router
from backend.core.database import Base, engine
from backend.core.settings import settings


app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(predictions_router, prefix=settings.api_prefix)
