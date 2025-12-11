"""Job manager for handling long-running processes."""

import asyncio
import logging
import subprocess
import sys
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PAUSED = "paused"


class JobType(str, Enum):
    """Job type enumeration."""

    TRAINING = "training"
    BACKTEST = "backtest"
    WALKFORWARD = "walkforward"
    EVALUATE = "evaluate"


class Job:
    """Represents a running or completed job."""

    def __init__(
        self,
        job_id: str,
        job_type: JobType,
        config: dict[str, Any],
        command: list[str],
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.config = config
        self.command = command
        self.status = JobStatus.PENDING
        self.process: Optional[asyncio.subprocess.Process] = None
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self.progress: dict[str, Any] = {}
        self.result: Optional[dict[str, Any]] = None
        self.output_lines: list[str] = []
        self._subscribers: list[Callable] = []

    def to_dict(self) -> dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "jobId": self.job_id,
            "jobType": self.job_type.value,
            "status": self.status.value,
            "config": self.config,
            "progress": self.progress,
            "createdAt": self.created_at.isoformat() + "Z",
            "startedAt": self.started_at.isoformat() + "Z" if self.started_at else None,
            "completedAt": self.completed_at.isoformat() + "Z" if self.completed_at else None,
            "error": self.error,
            "result": self.result,
        }

    def subscribe(self, callback: Callable) -> None:
        """Subscribe to job updates."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from job updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    async def notify_subscribers(self) -> None:
        """Notify all subscribers of state change."""
        for callback in self._subscribers:
            try:
                await callback(self)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")


class JobManager:
    """Manages long-running jobs."""

    _instance: Optional["JobManager"] = None

    def __new__(cls) -> "JobManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.jobs: dict[str, Job] = {}
        self.project_root = Path(__file__).parent.parent.parent.parent
        logger.info(f"JobManager initialized with project root: {self.project_root}")

    def create_job(
        self,
        job_type: JobType,
        config: dict[str, Any],
    ) -> Job:
        """Create a new job."""
        job_id = f"{job_type.value}_{uuid.uuid4().hex[:8]}"
        command = self._build_command(job_type, config)
        job = Job(job_id, job_type, config, command)
        self.jobs[job_id] = job
        logger.info(f"Created job {job_id} of type {job_type.value}")
        return job

    def _build_command(self, job_type: JobType, config: dict[str, Any]) -> list[str]:
        """Build command list for subprocess."""
        main_py = str(self.project_root / "main.py")
        cmd = [sys.executable, main_py]

        if job_type == JobType.TRAINING:
            cmd.append("train")
            if config.get("symbols"):
                cmd.extend(["--symbols"] + config["symbols"])
            elif config.get("symbol"):
                cmd.extend(["--symbol", config["symbol"]])
            if config.get("timeframe"):
                cmd.extend(["--timeframe", config["timeframe"]])
            if config.get("bars"):
                cmd.extend(["--bars", str(config["bars"])])
            if config.get("epochs"):
                cmd.extend(["--epochs", str(config["epochs"])])
            if config.get("timesteps"):
                cmd.extend(["--timesteps", str(config["timesteps"])])
            if config.get("modelDir"):
                cmd.extend(["--model-dir", config["modelDir"]])
            if config.get("multiTimeframe"):
                cmd.append("--multi-timeframe")

        elif job_type == JobType.BACKTEST:
            cmd.append("backtest")
            if config.get("symbol"):
                cmd.extend(["--symbol", config["symbol"]])
            if config.get("timeframe"):
                cmd.extend(["--timeframe", config["timeframe"]])
            if config.get("bars"):
                cmd.extend(["--bars", str(config["bars"])])
            if config.get("modelDir"):
                cmd.extend(["--model-dir", config["modelDir"]])
            if config.get("realisticMode"):
                cmd.append("--realistic")
            if config.get("monteCarlo"):
                cmd.append("--monte-carlo")

        elif job_type == JobType.WALKFORWARD:
            cmd.append("walkforward")
            if config.get("symbol"):
                cmd.extend(["--symbol", config["symbol"]])
            if config.get("timeframe"):
                cmd.extend(["--timeframe", config["timeframe"]])
            if config.get("bars"):
                cmd.extend(["--bars", str(config["bars"])])

        elif job_type == JobType.EVALUATE:
            cmd.append("evaluate")
            if config.get("symbol"):
                cmd.extend(["--symbol", config["symbol"]])
            if config.get("timeframe"):
                cmd.extend(["--timeframe", config["timeframe"]])
            if config.get("bars"):
                cmd.extend(["--bars", str(config["bars"])])
            if config.get("modelDir"):
                cmd.extend(["--model-dir", config["modelDir"]])

        return cmd

    async def start_job(self, job_id: str) -> Job:
        """Start a job."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status != JobStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in pending state")

        job.status = JobStatus.STARTING
        job.started_at = datetime.utcnow()

        try:
            # Start the subprocess
            job.process = await asyncio.create_subprocess_exec(
                *job.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self.project_root),
            )
            job.status = JobStatus.RUNNING
            logger.info(f"Started job {job_id} with PID {job.process.pid}")

            # Start output reader task
            asyncio.create_task(self._read_output(job))

            await job.notify_subscribers()
            return job

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            logger.error(f"Failed to start job {job_id}: {e}")
            await job.notify_subscribers()
            raise

    async def _read_output(self, job: Job) -> None:
        """Read output from job process."""
        if not job.process or not job.process.stdout:
            return

        try:
            while True:
                line = await job.process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                job.output_lines.append(decoded)

                # Parse progress from output
                self._parse_progress(job, decoded)

                # Keep only last 1000 lines
                if len(job.output_lines) > 1000:
                    job.output_lines = job.output_lines[-1000:]

            # Wait for process to complete
            await job.process.wait()

            if job.process.returncode == 0:
                job.status = JobStatus.COMPLETED
            else:
                job.status = JobStatus.FAILED
                job.error = f"Process exited with code {job.process.returncode}"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error(f"Error reading job output: {e}")

        finally:
            job.completed_at = datetime.utcnow()
            await job.notify_subscribers()
            logger.info(f"Job {job.job_id} finished with status {job.status.value}")

    def _parse_progress(self, job: Job, line: str) -> None:
        """Parse progress from output line."""
        # Parse training epoch progress
        if "Epoch" in line and "/" in line:
            try:
                # Example: "Epoch 45/100"
                parts = line.split("Epoch")[1].split()[0].split("/")
                if len(parts) == 2:
                    current = int(parts[0])
                    total = int(parts[1])
                    job.progress["currentEpoch"] = current
                    job.progress["totalEpochs"] = total
                    job.progress["percent"] = int((current / total) * 100)
            except (IndexError, ValueError):
                pass

        # Parse loss values
        if "loss" in line.lower():
            try:
                if "train_loss" in line or "Training loss" in line:
                    # Parse train loss
                    for part in line.split():
                        if part.replace(".", "").replace("-", "").isdigit():
                            job.progress["trainLoss"] = float(part)
                            break
                if "val_loss" in line or "Validation loss" in line:
                    # Parse validation loss
                    for part in line.split():
                        if part.replace(".", "").replace("-", "").isdigit():
                            job.progress["valLoss"] = float(part)
                            break
            except (IndexError, ValueError):
                pass

    async def stop_job(self, job_id: str) -> Job:
        """Stop a running job."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if job.status not in [JobStatus.RUNNING, JobStatus.STARTING]:
            raise ValueError(f"Job {job_id} is not running")

        if job.process:
            job.process.terminate()
            try:
                await asyncio.wait_for(job.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                job.process.kill()
                await job.process.wait()

        job.status = JobStatus.STOPPED
        job.completed_at = datetime.utcnow()
        await job.notify_subscribers()
        logger.info(f"Stopped job {job_id}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Job], int]:
        """List jobs with optional filtering."""
        jobs = list(self.jobs.values())

        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        total = len(jobs)
        jobs = jobs[offset : offset + limit]

        return jobs, total


# Global instance
job_manager = JobManager()
