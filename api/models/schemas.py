"""
schemas.py
----------
Pydantic request/response models for the PrivacyShield API.
"""

from pydantic import BaseModel
from typing import Optional


class RedactResponse(BaseModel):
    """Response after successful redaction."""
    job_id: str
    original_name: str
    download_name: str
    encryption_key: str
    stats: dict


class RestoreResponse(BaseModel):
    """Response after successful restore."""
    restore_id: str
    download_name: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str