"""
Data Service — port 8003
Manages mission lifecycle, detection logging, and export.

Usage:
    cd ~/soil-crack-detection
    python3 -m uvicorn jetson.services.data.main:app --host 0.0.0.0 --port 8003
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routes as _routes

app = FastAPI(title="Data Service", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(_routes.router)


@app.get("/status")
async def status():
    return {"ok": True, "service": "data"}


@app.on_event("startup")
async def startup():
    print("Data service ready.")
