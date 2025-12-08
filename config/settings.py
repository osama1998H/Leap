"""
Leap Trading System - Configuration Settings
Centralized configuration management for the entire trading system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json
import os


class TimeFrame(Enum):
    """Supported trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class TradingMode(Enum):
    """Trading execution modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "NZDUSD", "USDCAD"
    ])
    primary_timeframe: str = "1h"
    additional_timeframes: List[str] = field(default_factory=lambda: ["15m", "4h", "1d"])
    lookback_window: int = 120  # Number of candles to look back
    prediction_horizon: int = 12  # Number of candles to predict ahead
    train_test_split: float = 0.8
    validation_split: float = 0.1

    # Feature engineering
    use_technical_indicators: bool = True
    use_price_patterns: bool = True
    use_volume_features: bool = True
    normalize_method: str = "robust"  # robust, minmax, standard


@dataclass
class TransformerConfig:
    """Transformer model configuration."""
    d_model: int = 128  # Model dimension
    n_heads: int = 8  # Number of attention heads
    n_encoder_layers: int = 4
    d_ff: int = 512  # Feed-forward dimension
    dropout: float = 0.1
    max_seq_length: int = 120

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15  # Early stopping patience

    # Online learning
    online_learning_rate: float = 1e-5
    online_batch_size: int = 16
    adaptation_frequency: int = 100  # Adapt every N steps


@dataclass
class PPOConfig:
    """PPO Agent configuration."""
    # Network architecture
    actor_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256, 128])
    critic_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256, 128])

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip range
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Training
    n_steps: int = 2048  # Steps per update
    n_epochs: int = 10  # Epochs per update
    batch_size: int = 64
    total_timesteps: int = 1_000_000

    # Online learning
    online_update_frequency: int = 500
    experience_buffer_size: int = 50_000


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size: float = 0.02  # Max 2% of account per trade
    max_daily_loss: float = 0.05  # Max 5% daily loss
    max_drawdown: float = 0.15  # Max 15% drawdown
    default_stop_loss_pips: int = 50
    default_take_profit_pips: int = 100
    risk_reward_ratio: float = 2.0
    max_open_positions: int = 5
    use_trailing_stop: bool = True
    trailing_stop_pips: int = 30


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0  # USD per lot
    slippage_pips: float = 1.0
    spread_pips: float = 1.5
    leverage: int = 100

    # Walk-forward optimization
    train_window_days: int = 180
    test_window_days: int = 30
    walk_forward_steps: int = 12

    # Monte Carlo simulation
    n_simulations: int = 1000
    confidence_level: float = 0.95


@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "sharpe_ratio", "sortino_ratio", "max_drawdown",
        "win_rate", "profit_factor", "calmar_ratio",
        "total_return", "volatility", "var_95"
    ])
    benchmark_symbol: str = "EURUSD"
    risk_free_rate: float = 0.02  # Annual risk-free rate
    trading_days_per_year: int = 252


@dataclass
class AutoTraderConfig:
    """Auto-trader configuration."""
    # Trading settings
    symbols: List[str] = field(default_factory=lambda: ['EURUSD'])
    timeframe: str = '1h'

    # Risk settings
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_positions: int = 3
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.10  # 10% max drawdown

    # Stop loss / Take profit
    default_sl_pips: float = 50.0
    default_tp_pips: float = 100.0

    # Trading hours (UTC)
    trading_start_hour: int = 0  # 24/7 by default
    trading_end_hour: int = 24
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # Execution
    paper_mode: bool = True
    loop_interval: float = 1.0  # Seconds between trading cycles
    bar_interval: int = 3600  # Seconds per bar (for 1h = 3600)

    # Model settings
    min_confidence: float = 0.6
    prediction_threshold: float = 0.001  # Min predicted return for entry

    # Online learning
    enable_online_learning: bool = True
    adaptation_frequency: int = 100  # Trades between adaptations
    max_adaptations_per_day: int = 10

    # MT5 connection (optional - can also login via terminal)
    # Sensitive credentials are loaded from environment variables for security
    mt5_login: Optional[int] = field(
        default_factory=lambda: int(os.environ.get('MT5_LOGIN', '0')) or None
    )
    mt5_password: Optional[str] = field(
        default_factory=lambda: os.environ.get('MT5_PASSWORD')
    )
    mt5_server: Optional[str] = field(
        default_factory=lambda: os.environ.get('MT5_SERVER')
    )
    magic_number: int = 234567

    # Monitoring
    log_trades: bool = True
    heartbeat_interval: float = 60.0  # Seconds between heartbeat logs


@dataclass
class SystemConfig:
    """Main system configuration."""
    # Paths
    base_dir: str = field(default_factory=lambda: os.getcwd())
    models_dir: str = "saved_models"
    logs_dir: str = "logs"
    data_dir: str = "data"
    checkpoints_dir: str = "checkpoints"
    results_dir: str = "results"

    # System settings
    trading_mode: str = "backtest"
    device: str = "auto"  # auto, cpu, cuda
    seed: int = 42
    use_mixed_precision: bool = True
    num_workers: int = 4

    # Logging
    log_level: str = "INFO"
    tensorboard_enabled: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "leap-trading"

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    auto_trader: AutoTraderConfig = field(default_factory=AutoTraderConfig)

    def save(self, path: str):
        """Save configuration to JSON file.

        Note: Sensitive credentials (mt5_password) are excluded from serialization.
        These should be loaded from environment variables.
        """
        # Fields to exclude from serialization (sensitive data)
        sensitive_fields = {'mt5_password'}

        def serialize(obj, exclude_sensitive=False):
            if hasattr(obj, '__dataclass_fields__'):
                result = {}
                for k, v in obj.__dict__.items():
                    # Skip sensitive fields
                    if exclude_sensitive and k in sensitive_fields:
                        continue
                    result[k] = serialize(v, exclude_sensitive=True)
                return result
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [serialize(v, exclude_sensitive=exclude_sensitive) for v in obj]
            return obj

        with open(path, 'w') as f:
            json.dump(serialize(self, exclude_sensitive=True), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SystemConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Recursively convert nested dicts to dataclasses
        config = cls(
            data=DataConfig(**data.get('data', {})),
            transformer=TransformerConfig(**data.get('transformer', {})),
            ppo=PPOConfig(**data.get('ppo', {})),
            risk=RiskConfig(**data.get('risk', {})),
            backtest=BacktestConfig(**data.get('backtest', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            auto_trader=AutoTraderConfig(**data.get('auto_trader', {}))
        )

        # Set top-level attributes
        for key in ['base_dir', 'models_dir', 'logs_dir', 'data_dir',
                    'checkpoints_dir', 'results_dir', 'trading_mode',
                    'device', 'seed', 'use_mixed_precision', 'num_workers',
                    'log_level', 'tensorboard_enabled', 'wandb_enabled', 'wandb_project']:
            if key in data:
                setattr(config, key, data[key])

        return config

    def get_path(self, subdir: str) -> str:
        """Get full path for a subdirectory."""
        path = os.path.join(self.base_dir, getattr(self, f"{subdir}_dir", subdir))
        os.makedirs(path, exist_ok=True)
        return path


# Default configuration instance
def get_config() -> SystemConfig:
    """Get default system configuration."""
    return SystemConfig()
