"""System-related schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class CPUMetrics(BaseModel):
    """CPU metrics."""

    percent: float
    cores: int


class MemoryMetrics(BaseModel):
    """Memory metrics."""

    total: int
    used: int
    percent: float


class GPUMemory(BaseModel):
    """GPU memory metrics."""

    total: int
    used: int
    percent: float


class GPUMetrics(BaseModel):
    """GPU metrics."""

    available: bool
    name: Optional[str] = None
    memory: Optional[GPUMemory] = None


class DiskMetrics(BaseModel):
    """Disk metrics."""

    total: int
    used: int
    percent: float


class SystemMetricsData(BaseModel):
    """System metrics data."""

    cpu: CPUMetrics
    memory: MemoryMetrics
    gpu: GPUMetrics
    disk: DiskMetrics


class SystemMetricsResponse(BaseModel):
    """System metrics response."""

    data: SystemMetricsData


class ComponentStatus(BaseModel):
    """Component status."""

    database: str = "ok"
    mlflow: str = "ok"
    gpu: str = "available"


class HealthData(BaseModel):
    """Health check data."""

    status: str = "healthy"
    version: str = "1.0.0"
    components: ComponentStatus = Field(default_factory=ComponentStatus)


class HealthResponse(BaseModel):
    """Health check response."""

    data: HealthData
