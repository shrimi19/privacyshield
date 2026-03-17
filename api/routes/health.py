"""
health.py
---------
GET /health — service health check
"""

from fastapi import APIRouter
from api.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns service status."""
    return HealthResponse(status="ok", version="1.0.0")