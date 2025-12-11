"""WebSocket route for real-time updates."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.job_manager import Job, job_manager
from ..core.websocket_manager import Channel, MessageType, ws_manager

logger = logging.getLogger(__name__)

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


# Monkey-patch job creation to auto-subscribe new jobs
_original_create_job = job_manager.create_job


def _create_job_with_subscriber(job_type, config):
    """Wrapper to add WebSocket subscriber to new jobs."""
    job = _original_create_job(job_type, config)
    job.subscribe(job_update_callback)
    return job


job_manager.create_job = _create_job_with_subscriber
