"""IRIS Backend — FastAPI + titiler, localhost only."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from titiler.application.main import app as titiler_app

app = FastAPI(title="IRIS Backend", version="0.1.0")

# Allow the Electron frontend (localhost) to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount titiler at /tiles — serves COGs as XYZ map tiles on demand
app.mount("/tiles", titiler_app)


@app.get("/health")
def health():
    return {"status": "ok"}


# Ingestion, catalog, search endpoints go here as routers
# from .api import ingestion, search, change
# app.include_router(ingestion.router)
# app.include_router(search.router)
# app.include_router(change.router)
