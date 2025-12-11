"""Common schema definitions."""

from datetime import datetime
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Meta(BaseModel):
    """Response metadata."""

    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    request_id: Optional[str] = Field(None, alias="requestId")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    data: T
    meta: Meta = Field(default_factory=Meta)


class ErrorDetail(BaseModel):
    """Error detail."""

    code: str
    message: str
    details: Optional[dict[str, list[str]]] = None


class ErrorResponse(BaseModel):
    """Error response."""

    error: ErrorDetail
    meta: Meta = Field(default_factory=Meta)


class Pagination(BaseModel):
    """Pagination info."""

    limit: int
    offset: int
    total: int
    has_more: bool = Field(alias="hasMore")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    data: T
    pagination: Pagination
    meta: Meta = Field(default_factory=Meta)
