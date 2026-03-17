"""
main.py
-------
FastAPI application for PrivacyShield.
Mounts all routes and configures CORS.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.redact import router as redact_router
from api.routes.unredact import router as unredact_router

app = FastAPI(
    title="PrivacyShield API",
    description="Redact PII from PDFs and restore them with a key.",
    version="1.0.0",
)

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(redact_router)
app.include_router(unredact_router)