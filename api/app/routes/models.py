"""Models API routes."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas.models import ModelDetailResponse, ModelListData, ModelListResponse
from ..services.models_service import models_service

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all saved models."""
    models = models_service.list_models()
    return ModelListResponse(data=ModelListData(models=models))


@router.get("/{directory:path}", response_model=ModelDetailResponse)
async def get_model_detail(directory: str):
    """Get detailed model information."""
    detail = models_service.get_model_detail(directory)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Model directory {directory} not found")
    return ModelDetailResponse(data=detail)


@router.get("/{directory:path}/download")
async def download_model(directory: str):
    """Download model as ZIP file."""
    buffer = models_service.create_download(directory)
    if not buffer:
        raise HTTPException(status_code=404, detail=f"Model directory {directory} not found")

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=model.zip"},
    )
