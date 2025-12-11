"""WebSocket route for real-time updates."""

import json
import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.job_manager import Job, job_manager
from ..core.websocket_manager import Channel, MessageType, ws_manager

logger = logging.getLogger(__name__)

# Regex pattern to parse log lines (matches common Python logging format)
LOG_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)\s+"
    r"(?:(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+)?"
    r"(?:(?P<logger>[\w.]+)\s+)?"
    r"(?P<message>.*)$"
)

router = APIRouter(tags=["websocket"])


async def job_update_callback(job: Job) -> None:
    """Callback for job updates - broadcasts to WebSocket subscribers."""
    # Determine channel and message type based on job type
    channel = Channel.TRAINING.value if "training" in job.job_type.value else Channel.BACKTEST.value

    # Build progress data
    data = {
        "jobId": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
    }

    # Add job-specific data
    if job.job_type.value == "training":
        data.update({
            "phase": job.progress.get("phase", "transformer"),
            "epoch": job.progress.get("currentEpoch"),
            "totalEpochs": job.progress.get("totalEpochs"),
            "trainLoss": job.progress.get("trainLoss"),
            "valLoss": job.progress.get("valLoss"),
            "learningRate": job.progress.get("learningRate"),
            "patienceCounter": job.progress.get("patienceCounter"),
            "patienceMax": job.progress.get("patienceMax"),
        })
        if job.started_at:
            elapsed = (job.completed_at or datetime.now(timezone.utc)) - job.started_at
            data["elapsedSeconds"] = int(elapsed.total_seconds())
            # Estimate remaining time based on epoch progress
            if job.progress.get("currentEpoch") and job.progress.get("totalEpochs"):
                current = job.progress["currentEpoch"]
                total = job.progress["totalEpochs"]
                if current > 0:
                    time_per_epoch = elapsed.total_seconds() / current
                    remaining_epochs = total - current
                    data["estimatedRemainingSeconds"] = int(time_per_epoch * remaining_epochs)
    else:
        data.update({
            "currentBar": job.progress.get("currentBar"),
            "totalBars": job.progress.get("totalBars"),
            "currentEquity": job.progress.get("currentEquity"),
            "currentDrawdown": job.progress.get("currentDrawdown"),
            "tradesExecuted": job.progress.get("tradesExecuted"),
        })

    # Determine message type based on status
    if job.status.value == "completed":
        msg_type = MessageType.JOB_COMPLETE
        data["result"] = job.result
    elif job.status.value == "failed":
        msg_type = MessageType.JOB_ERROR
        data["error"] = {"message": job.error, "recoverable": False}
    else:
        msg_type = (
            MessageType.TRAINING_PROGRESS
            if channel == Channel.TRAINING.value
            else MessageType.BACKTEST_PROGRESS
        )

    # Broadcast to subscribers
    await ws_manager.broadcast_to_channel(channel, msg_type, data, job.job_id)


async def broadcast_log_entry(job_id: str, line: str, job_type: str = "training") -> None:
    """Broadcast a log entry to WebSocket subscribers."""
    # Parse the log line
    match = LOG_PATTERN.match(line)
    if match:
        data = {
            "jobId": job_id,
            "timestamp": match.group("timestamp"),
            "level": match.group("level") or "INFO",
            "logger": match.group("logger") or "",
            "message": match.group("message"),
        }
    else:
        # Unparseable line - send as plain message
        data = {
            "jobId": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": "INFO",
            "logger": "",
            "message": line,
        }

    # Determine channel
    channel = Channel.LOGS.value

    # Broadcast to subscribers
    await ws_manager.broadcast_to_channel(channel, MessageType.LOG_ENTRY, data, job_id)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await ws_manager.connect(websocket)

    try:
        while True:
            # Receive and parse message
            try:
                data = await websocket.receive_json()
                await ws_manager.handle_message(websocket, data)
            except json.JSONDecodeError:
                await ws_manager.send_personal_message(
                    websocket,
                    MessageType.ERROR,
                    {"message": "Invalid JSON"},
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)


def setup_job_subscribers():
    """Set up subscribers for existing and new jobs."""
    # This function should be called on startup to wire up job notifications
    # Subscribe to all existing jobs
    for job_id, job in job_manager.jobs.items():
        job.subscribe(job_update_callback)
        job.subscribe_logs(broadcast_log_entry)


# Monkey-patch job creation to auto-subscribe new jobs
_original_create_job = job_manager.create_job


def _create_job_with_subscriber(job_type, config):
    """Wrapper to add WebSocket subscriber to new jobs."""
    job = _original_create_job(job_type, config)
    job.subscribe(job_update_callback)
    job.subscribe_logs(broadcast_log_entry)
    return job


job_manager.create_job = _create_job_with_subscriber
