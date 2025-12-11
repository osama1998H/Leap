"""System service."""

import logging
import shutil
from typing import Any

import psutil

from ..schemas.system import (
    ComponentStatus,
    CPUMetrics,
    DiskMetrics,
    GPUMemory,
    GPUMetrics,
    HealthData,
    MemoryMetrics,
    SystemMetricsData,
)

logger = logging.getLogger(__name__)


class SystemService:
    """Service for system monitoring."""

    def get_health(self) -> HealthData:
        """Get system health status."""
        # Check component health
        mlflow_status = self._check_mlflow()
        gpu_status = "available" if self._check_gpu() else "unavailable"

        components = ComponentStatus(
            database="ok",  # No database in current architecture
            mlflow=mlflow_status,
            gpu=gpu_status,
        )

        # Overall status is healthy only if critical components are ok
        overall_status = "healthy" if mlflow_status in ("ok", "unavailable") else "degraded"

        return HealthData(
            status=overall_status,
            version="1.0.0",
            components=components,
        )

    def _check_mlflow(self) -> str:
        """Check MLflow connectivity."""
        try:
            import mlflow
            # Just check if mlflow is importable and configured
            # Full connectivity check would require network call
            return "ok"
        except ImportError:
            return "unavailable"
        except Exception:
            return "error"

    def get_metrics(self) -> SystemMetricsData:
        """Get system metrics."""
        # CPU metrics
        cpu = CPUMetrics(
            percent=psutil.cpu_percent(),
            cores=psutil.cpu_count() or 1,
        )

        # Memory metrics
        mem = psutil.virtual_memory()
        memory = MemoryMetrics(
            total=mem.total,
            used=mem.used,
            percent=mem.percent,
        )

        # GPU metrics
        gpu = self._get_gpu_metrics()

        # Disk metrics
        disk_usage = shutil.disk_usage("/")
        disk = DiskMetrics(
            total=disk_usage.total,
            used=disk_usage.used,
            percent=(disk_usage.used / disk_usage.total) * 100,
        )

        return SystemMetricsData(
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            disk=disk,
        )

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_gpu_metrics(self) -> GPUMetrics:
        """Get GPU metrics."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_total = props.total_memory

                return GPUMetrics(
                    available=True,
                    name=props.name,
                    memory=GPUMemory(
                        total=memory_total,
                        used=memory_allocated,
                        percent=(memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
                    ),
                )
        except Exception as e:
            logger.debug(f"GPU metrics unavailable: {e}")

        return GPUMetrics(
            available=False,
            name=None,
            memory=None,
        )


system_service = SystemService()
