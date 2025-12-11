"""Model-related schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictorInfo(BaseModel):
    """Predictor model info."""

    exists: bool
    input_dim: Optional[int] = Field(None, alias="inputDim")
    created_at: Optional[str] = Field(None, alias="createdAt")


class AgentInfo(BaseModel):
    """Agent model info."""

    exists: bool
    state_dim: Optional[int] = Field(None, alias="stateDim")
    action_dim: Optional[int] = Field(None, alias="actionDim")
    created_at: Optional[str] = Field(None, alias="createdAt")


class ModelMetadata(BaseModel):
    """Model metadata."""

    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    trained_at: Optional[str] = Field(None, alias="trainedAt")


class ModelInfo(BaseModel):
    """Model directory info."""

    directory: str
    predictor: PredictorInfo
    agent: AgentInfo
    metadata: ModelMetadata


class ModelListData(BaseModel):
    """Model list data."""

    models: list[ModelInfo]


class ModelListResponse(BaseModel):
    """Model list response."""

    data: ModelListData


class ModelDetailMetadata(BaseModel):
    """Detailed model metadata."""

    predictor: Optional[dict[str, Any]] = None
    agent: Optional[dict[str, Any]] = None
    trained_symbol: Optional[str] = Field(None, alias="trainedSymbol")
    trained_timeframe: Optional[str] = Field(None, alias="trainedTimeframe")
    created_at: Optional[str] = Field(None, alias="createdAt")


class ModelDetailData(BaseModel):
    """Model detail data."""

    directory: str
    files: list[str]
    metadata: ModelDetailMetadata
    config: Optional[dict[str, Any]] = None


class ModelDetailResponse(BaseModel):
    """Model detail response."""

    data: ModelDetailData
