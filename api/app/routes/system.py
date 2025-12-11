"""System API routes."""

from fastapi import APIRouter

from ..schemas.system import HealthResponse, SystemMetricsResponse
from ..services.system_service import system_service

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    health = system_service.get_health()
    return HealthResponse(data=health)


@router.get("/metrics/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system resource metrics."""
    metrics = system_service.get_metrics()
    return SystemMetricsResponse(data=metrics)
