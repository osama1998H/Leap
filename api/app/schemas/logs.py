"""Logging-related schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class LogFileInfo(BaseModel):
    """Log file info."""

    name: str
    size: int
    modified_at: str = Field(alias="modifiedAt")


class LogFileListData(BaseModel):
    """Log file list data."""

    files: list[LogFileInfo]


class LogFileListResponse(BaseModel):
    """Log file list response."""

    data: LogFileListData


class LogLine(BaseModel):
    """Log line."""

    number: int
    timestamp: Optional[str] = None
    level: Optional[str] = None
    logger: Optional[str] = None
    message: str


class LogFileData(BaseModel):
    """Log file data."""

    filename: str
    lines: list[LogLine]
    total_lines: int = Field(alias="totalLines")


class LogFileResponse(BaseModel):
    """Log file response."""

    data: LogFileData
