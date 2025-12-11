"""Logs API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from ..schemas.logs import LogFileData, LogFileListData, LogFileListResponse, LogFileResponse
from ..services.logs_service import logs_service

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("/files", response_model=LogFileListResponse)
async def list_log_files():
    """List available log files."""
    files = logs_service.list_log_files()
    return LogFileListResponse(data=LogFileListData(files=files))


@router.get("/files/{filename}", response_model=LogFileResponse)
async def get_log_file(
    filename: str,
    level: Optional[str] = Query(None, description="Filter by log level"),
    search: Optional[str] = Query(None, description="Text search"),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """Get log file contents."""
    lines, total = logs_service.get_log_file(filename, level, search, limit, offset)
    if not lines and total == 0:
        raise HTTPException(status_code=404, detail=f"Log file {filename} not found")

    return LogFileResponse(
        data=LogFileData(
            filename=filename,
            lines=lines,
            totalLines=total,
        )
    )
