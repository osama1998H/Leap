"""Configuration-related schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data configuration."""

    symbols: list[str] = ["EURUSD", "GBPUSD"]
    primary_timeframe: str = Field("1h", alias="primaryTimeframe")
    additional_timeframes: list[str] = Field(["15m", "4h", "1d"], alias="additionalTimeframes")
    lookback_window: int = Field(120, alias="lookbackWindow")
    prediction_horizon: int = Field(12, alias="predictionHorizon")


class TransformerConfigFull(BaseModel):
    """Full transformer configuration."""

    d_model: int = Field(128, alias="dModel")
    n_heads: int = Field(8, alias="nHeads")
    n_encoder_layers: int = Field(4, alias="nEncoderLayers")
    dropout: float = 0.1
    learning_rate: float = Field(0.0001, alias="learningRate")
    epochs: int = 100
    patience: int = 15


class PPOConfigFull(BaseModel):
    """Full PPO configuration."""

    learning_rate: float = Field(0.0003, alias="learningRate")
    gamma: float = 0.99
    gae_lambda: float = Field(0.95, alias="gaeLambda")
    clip_epsilon: float = Field(0.2, alias="clipEpsilon")
    total_timesteps: int = Field(1000000, alias="totalTimesteps")


class RiskConfig(BaseModel):
    """Risk configuration."""

    max_position_size: float = Field(0.02, alias="maxPositionSize")
    max_daily_loss: float = Field(0.05, alias="maxDailyLoss")
    default_stop_loss_pips: int = Field(50, alias="defaultStopLossPips")
    default_take_profit_pips: int = Field(100, alias="defaultTakeProfitPips")


class BacktestConfigFull(BaseModel):
    """Full backtest configuration."""

    initial_balance: float = Field(10000, alias="initialBalance")
    leverage: int = 100
    spread_pips: float = Field(1.5, alias="spreadPips")
    slippage_pips: float = Field(0.5, alias="slippagePips")
    commission_per_lot: float = Field(7.0, alias="commissionPerLot")
    risk_per_trade: float = Field(0.02, alias="riskPerTrade")
    n_simulations: int = Field(1000, alias="nSimulations", description="Number of Monte Carlo simulations")


class SystemConfigData(BaseModel):
    """Full system configuration."""

    data: DataConfig = Field(default_factory=DataConfig)
    transformer: TransformerConfigFull = Field(default_factory=TransformerConfigFull)
    ppo: PPOConfigFull = Field(default_factory=PPOConfigFull)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfigFull = Field(default_factory=BacktestConfigFull)


class ConfigResponse(BaseModel):
    """Configuration response."""

    data: SystemConfigData


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    data: Optional[DataConfig] = None
    transformer: Optional[TransformerConfigFull] = None
    ppo: Optional[PPOConfigFull] = None
    risk: Optional[RiskConfig] = None
    backtest: Optional[BacktestConfigFull] = None


class ConfigTemplate(BaseModel):
    """Configuration template."""

    id: str
    name: str
    description: Optional[str] = None
    created_at: str = Field(alias="createdAt")


class ConfigTemplateListData(BaseModel):
    """Template list data."""

    templates: list[ConfigTemplate]


class ConfigTemplateResponse(BaseModel):
    """Template list response."""

    data: ConfigTemplateListData


class ConfigTemplateCreateRequest(BaseModel):
    """Request to create template."""

    name: str
    description: Optional[str] = None
    config: dict[str, Any]


class ConfigTemplateCreateData(BaseModel):
    """Created template data."""

    id: str
    name: str
    created_at: str = Field(alias="createdAt")


class ConfigTemplateCreateResponse(BaseModel):
    """Template creation response."""

    data: ConfigTemplateCreateData


class ConfigValidateRequest(BaseModel):
    """Request to validate configuration."""

    transformer: Optional[dict[str, Any]] = None
    ppo: Optional[dict[str, Any]] = None
    risk: Optional[dict[str, Any]] = None
    backtest: Optional[dict[str, Any]] = None
