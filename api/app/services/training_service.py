"""Training service."""

import logging
from datetime import datetime
from typing import Any, Optional

from ..core.job_manager import Job, JobManager, JobStatus, JobType, job_manager
from ..schemas.training import (
    LogEntry,
    TrainingJobData,
    TrainingMetrics,
    TrainingProgress,
    TrainingStartRequest,
    TrainingTiming,
)

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for managing training jobs."""

    def __init__(self):
        self.job_manager = job_manager

    async def start_training(self, request: TrainingStartRequest) -> TrainingJobData:
        """Start a new training job."""
        config = {
            "symbols": request.symbols,
            "symbol": request.symbols[0] if request.symbols else "EURUSD",
            "timeframe": request.timeframe,
            "multiTimeframe": request.multi_timeframe,
            "additionalTimeframes": request.additional_timeframes,
            "bars": request.bars,
            "epochs": request.epochs,
            "timesteps": request.timesteps,
            "modelDir": request.model_dir,
        }

        if request.config:
            if request.config.transformer:
                config["transformer"] = request.config.transformer.model_dump(by_alias=True)
            if request.config.ppo:
                config["ppo"] = request.config.ppo.model_dump(by_alias=True)

        job = self.job_manager.create_job(JobType.TRAINING, config)
        await self.job_manager.start_job(job.job_id)

        return self._job_to_response(job)

    async def list_jobs(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[TrainingJobData], int]:
        """List training jobs."""
        job_status = JobStatus(status) if status else None

        jobs, total = self.job_manager.list_jobs(
            job_type=JobType.TRAINING,
            status=job_status,
            limit=limit,
            offset=offset,
        )

        # Filter by symbol if specified
        if symbol:
            jobs = [j for j in jobs if symbol in j.config.get("symbols", [])]

        return [self._job_to_response(j) for j in jobs], total

    async def get_job(self, job_id: str) -> Optional[TrainingJobData]:
        """Get a specific training job."""
        job = self.job_manager.get_job(job_id)
        if not job or job.job_type != JobType.TRAINING:
            return None
        return self._job_to_response(job)

    async def stop_job(self, job_id: str) -> TrainingJobData:
        """Stop a running training job."""
        job = await self.job_manager.stop_job(job_id)
        return self._job_to_response(job)

    def get_job_logs(
        self,
        job_id: str,
        level: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> tuple[list[LogEntry], bool]:
        """Get logs for a training job.

        Args:
            job_id: The job ID to get logs for
            level: Optional log level filter (INFO, DEBUG, WARNING, ERROR)
            since: Optional timestamp to filter logs (only return logs after this time)
            limit: Maximum number of log entries to return
        """
        job = self.job_manager.get_job(job_id)
        if not job:
            return [], False

        logs = []
        for line in job.output_lines:
            # Parse log line
            entry = self._parse_log_line(line)
            if entry:
                # Filter by level
                if level and entry.level != level:
                    continue

                # Filter by timestamp (since parameter)
                if since and entry.timestamp:
                    try:
                        # Parse the entry timestamp (format: YYYY-MM-DDTHH:MM:SSZ)
                        entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                        since_aware = since.replace(tzinfo=entry_time.tzinfo) if since.tzinfo is None else since
                        if entry_time <= since_aware:
                            continue
                    except (ValueError, AttributeError):
                        # If timestamp parsing fails, include the entry
                        pass

                logs.append(entry)

        # Apply limit after filtering
        has_more = len(logs) > limit
        logs = logs[-limit:]  # Take the most recent entries

        return logs, has_more

    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a log line into structured format."""
        # Simple parser - could be enhanced
        parts = line.split(" ", 3)
        if len(parts) >= 4:
            try:
                timestamp = f"{parts[0]}T{parts[1]}Z"
                level = parts[2]
                message = parts[3]
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    logger="training",
                )
            except Exception:
                pass

        return LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level="INFO",
            message=line,
            logger="training",
        )

    def _job_to_response(self, job: Job) -> TrainingJobData:
        """Convert job to response format."""
        progress = None
        if job.progress:
            progress = TrainingProgress(
                currentEpoch=job.progress.get("currentEpoch"),
                totalEpochs=job.progress.get("totalEpochs"),
                percent=job.progress.get("percent", 0),
                currentTimestep=job.progress.get("currentTimestep"),
                totalTimesteps=job.progress.get("totalTimesteps"),
            )

        metrics = None
        if job.progress:
            metrics = TrainingMetrics(
                trainLoss=job.progress.get("trainLoss"),
                valLoss=job.progress.get("valLoss"),
                learningRate=job.progress.get("learningRate"),
            )

        timing = None
        if job.started_at:
            elapsed = (datetime.utcnow() - job.started_at).total_seconds()
            timing = TrainingTiming(
                startedAt=job.started_at.isoformat() + "Z",
                elapsedSeconds=int(elapsed),
                estimatedRemainingSeconds=None,
            )

        return TrainingJobData(
            jobId=job.job_id,
            status=job.status.value,
            symbols=job.config.get("symbols", [job.config.get("symbol", "EURUSD")]),
            timeframe=job.config.get("timeframe"),
            config=job.config,
            phase="transformer",  # Could be enhanced to track actual phase
            progress=progress,
            metrics=metrics,
            timing=timing,
            createdAt=job.created_at.isoformat() + "Z",
            estimatedDuration=7200,  # Placeholder
        )


training_service = TrainingService()
