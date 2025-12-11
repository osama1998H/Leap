"""System service."""

import logging
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import psutil

from ..schemas.system import (
    ComponentStatus,
    CPUMetrics,
    DiskMetrics,
    GPUMemory,
    GPUMetrics,
    HealthData,
    MemoryMetrics,
    MLflowLaunchData,
    MLflowStatusData,
    SystemMetricsData,
)
from ..core.config import settings

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

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _find_mlflow_process(self) -> Optional[psutil.Process]:
        """Find running MLflow UI process."""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if any("mlflow" in str(arg).lower() for arg in cmdline):
                    if any("ui" in str(arg).lower() or "server" in str(arg).lower() for arg in cmdline):
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None

    def get_mlflow_status(self, port: int = 5000) -> MLflowStatusData:
        """Get MLflow server status."""
        try:
            import mlflow

            # Get MLflow configuration
            tracking_uri = "sqlite:///mlflow.db"  # Default from settings
            experiment_name = "leap-trading"

            try:
                # Try to load from project config if available
                sys.path.insert(0, str(settings.PROJECT_ROOT))
                from config.settings import MLflowConfig
                mlflow_config = MLflowConfig()
                tracking_uri = mlflow_config.tracking_uri
                experiment_name = mlflow_config.experiment_name
            except Exception:
                pass

            # Check if MLflow server is running
            running = self._is_port_in_use(port)
            url = f"http://localhost:{port}" if running else None

            return MLflowStatusData(
                running=running,
                url=url,
                port=port,
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
            )
        except ImportError:
            return MLflowStatusData(
                running=False,
                url=None,
                port=port,
                tracking_uri="",
                experiment_name="",
            )

    def launch_mlflow(self, port: int = 5000) -> MLflowLaunchData:
        """Launch MLflow UI server."""
        try:
            # Check if already running
            if self._is_port_in_use(port):
                return MLflowLaunchData(
                    success=True,
                    url=f"http://localhost:{port}",
                    message="MLflow UI is already running",
                )

            # Get tracking URI from config
            tracking_uri = "sqlite:///mlflow.db"
            try:
                sys.path.insert(0, str(settings.PROJECT_ROOT))
                from config.settings import MLflowConfig
                mlflow_config = MLflowConfig()
                tracking_uri = mlflow_config.tracking_uri
            except Exception:
                pass

            # Determine the database file path
            db_path = settings.PROJECT_ROOT / "mlflow.db"
            if tracking_uri.startswith("sqlite:///"):
                db_file = tracking_uri.replace("sqlite:///", "")
                if not Path(db_file).is_absolute():
                    db_path = settings.PROJECT_ROOT / db_file

            # Launch MLflow UI in background
            cmd = [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                f"sqlite:///{db_path}",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ]

            logger.info(f"Starting MLflow UI: {' '.join(cmd)}")

            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(settings.PROJECT_ROOT),
                start_new_session=True,
            )

            # Give it a moment to start
            import time
            time.sleep(2)

            # Check if it started successfully
            if process.poll() is None and self._is_port_in_use(port):
                return MLflowLaunchData(
                    success=True,
                    url=f"http://localhost:{port}",
                    message="MLflow UI started successfully",
                )
            else:
                # Try to get error output
                stderr = ""
                try:
                    _, stderr_bytes = process.communicate(timeout=1)
                    stderr = stderr_bytes.decode() if stderr_bytes else ""
                except Exception:
                    pass

                return MLflowLaunchData(
                    success=False,
                    url=f"http://localhost:{port}",
                    message=f"Failed to start MLflow UI: {stderr or 'Unknown error'}",
                )

        except Exception as e:
            logger.exception("Failed to launch MLflow UI")
            return MLflowLaunchData(
                success=False,
                url=f"http://localhost:{port}",
                message=f"Error launching MLflow: {str(e)}",
            )


system_service = SystemService()
