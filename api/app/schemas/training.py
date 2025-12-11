"""Training-related schemas."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TransformerConfig(BaseModel):
    """Transformer configuration."""

    d_model: int = Field(128, alias="dModel", ge=32, le=512)
    n_heads: int = Field(8, alias="nHeads", ge=1, le=16)
    n_encoder_layers: int = Field(4, alias="nEncoderLayers", ge=1, le=12)
    dropout: float = Field(0.1, ge=0.0, le=0.5)
    learning_rate: float = Field(0.0001, alias="learningRate", gt=0)


class PPOConfig(BaseModel):
    """PPO agent configuration."""

    learning_rate: float = Field(0.0003, alias="learningRate", gt=0)
    gamma: float = Field(0.99, ge=0.9, le=0.999)
    clip_epsilon: float = Field(0.2, alias="clipEpsilon", ge=0.1, le=0.3)


class TrainingConfig(BaseModel):
    """Training configuration."""

    transformer: Optional[TransformerConfig] = None
    ppo: Optional[PPOConfig] = None


class TrainingStartRequest(BaseModel):
    """Request to start training."""

    symbols: list[str] = Field(default=["EURUSD"])
    timeframe: str = Field("1h")
    multi_timeframe: bool = Field(False, alias="multiTimeframe")
    additional_timeframes: Optional[list[str]] = Field(None, alias="additionalTimeframes")
    bars: int = Field(50000, ge=1000, le=500000)
    epochs: int = Field(100, ge=1, le=1000)
    timesteps: int = Field(1000000, ge=10000, le=10000000)
    model_dir: str = Field("./saved_models", alias="modelDir")
    config: Optional[TrainingConfig] = None


class TrainingProgress(BaseModel):
    """Training progress."""

    current_epoch: Optional[int] = Field(None, alias="currentEpoch")
    total_epochs: Optional[int] = Field(None, alias="totalEpochs")
    percent: int = 0
    current_timestep: Optional[int] = Field(None, alias="currentTimestep")
    total_timesteps: Optional[int] = Field(None, alias="totalTimesteps")


class TrainingMetrics(BaseModel):
    """Training metrics."""

    train_loss: Optional[float] = Field(None, alias="trainLoss")
    val_loss: Optional[float] = Field(None, alias="valLoss")
    learning_rate: Optional[float] = Field(None, alias="learningRate")


class TrainingTiming(BaseModel):
    """Training timing info."""

    started_at: Optional[str] = Field(None, alias="startedAt")
    elapsed_seconds: Optional[int] = Field(None, alias="elapsedSeconds")
    estimated_remaining_seconds: Optional[int] = Field(None, alias="estimatedRemainingSeconds")


class TrainingJobData(BaseModel):
    """Training job data."""

    job_id: str = Field(alias="jobId")
    status: str
    symbols: list[str]
    timeframe: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    phase: Optional[str] = None
    progress: Optional[TrainingProgress] = None
    metrics: Optional[TrainingMetrics] = None
    timing: Optional[TrainingTiming] = None
    created_at: str = Field(alias="createdAt")
    estimated_duration: Optional[int] = Field(None, alias="estimatedDuration")


class TrainingJobResponse(BaseModel):
    """Training job response."""

    data: TrainingJobData


class TrainingJobListData(BaseModel):
    """Training job list data."""

    jobs: list[TrainingJobData]
    total: int
    limit: int
    offset: int


class TrainingJobListResponse(BaseModel):
    """Training job list response."""

    data: TrainingJobListData


class LogEntry(BaseModel):
    """Log entry."""

    timestamp: str
    level: str
    message: str
    logger: Optional[str] = None


class TrainingLogsData(BaseModel):
    """Training logs data."""

    logs: list[LogEntry]
    has_more: bool = Field(alias="hasMore")


class TrainingLogsResponse(BaseModel):
    """Training logs response."""

    data: TrainingLogsData
