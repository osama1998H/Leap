"""Configuration API routes."""

from fastapi import APIRouter, HTTPException

from ..schemas.config import (
    ConfigResponse,
    ConfigTemplateCreateRequest,
    ConfigTemplateCreateResponse,
    ConfigTemplateCreateData,
    ConfigTemplateListData,
    ConfigTemplateResponse,
    ConfigUpdateRequest,
    ConfigValidateRequest,
)
from ..services.config_service import config_service

router = APIRouter(prefix="/config", tags=["config"])


@router.get("", response_model=ConfigResponse)
async def get_config():
    """Get current system configuration."""
    config = config_service.get_config()
    return ConfigResponse(data=config)


@router.put("")
async def update_config(request: ConfigUpdateRequest):
    """Update system configuration."""
    config = config_service.update_config(request)
    return {"data": {"updated": True, "config": config.model_dump(by_alias=True)}}


@router.get("/templates", response_model=ConfigTemplateResponse)
async def list_templates():
    """List configuration templates."""
    templates = config_service.list_templates()
    return ConfigTemplateResponse(
        data=ConfigTemplateListData(templates=templates)
    )


@router.post("/templates", status_code=201)
async def create_template(request: ConfigTemplateCreateRequest):
    """Create a new configuration template."""
    template = config_service.create_template(
        request.name, request.description, request.config
    )
    return {
        "data": {
            "id": template.id,
            "name": template.name,
            "createdAt": template.created_at,
        }
    }


@router.get("/templates/{template_id}")
async def get_template(template_id: str):
    """Get a configuration template."""
    config = config_service.get_template(template_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    return {"data": config}


@router.post("/validate")
async def validate_config(request: ConfigValidateRequest):
    """Validate configuration."""
    errors = config_service.validate_config(request.model_dump(exclude_none=True))
    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "VALIDATION_ERROR",
                "message": "Configuration validation failed",
                "details": errors,
            },
        )
    return {"data": {"valid": True}}
