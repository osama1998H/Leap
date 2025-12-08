# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Leap is an AI-powered forex trading system combining Transformer-based price prediction with PPO reinforcement learning. Features online learning for continuous market adaptation.

## Common Commands

```bash
# Training
python main.py train --symbol EURUSD --timeframe 1h --epochs 100 --timesteps 100000
python main.py train --config config/custom_config.json

# Backtesting
python main.py backtest --symbol EURUSD --bars 50000
python main.py backtest --symbol EURUSD --realistic
python main.py walkforward --symbol EURUSD

# Live Trading
python main.py live --paper           # Paper trading (safe testing)
python main.py autotrade --paper --symbol EURUSD  # Auto-trader with MT5 (Windows only)

# Evaluation
python main.py evaluate --model-dir ./saved_models

# Testing
python -m pytest tests/test_integration.py -v
python -m pytest tests/test_auto_trader.py -v
```

## Architecture

### Key Components

- **`main.py`**: CLI entry point with `LeapTradingSystem` orchestrator class
- **`config/settings.py`**: Dataclass-based configuration (SystemConfig, DataConfig, TransformerConfig, PPOConfig, RiskConfig, etc.)
- **`core/data_pipeline.py`**: Data fetching, feature engineering (100+ features), sequence creation
- **`core/trading_env.py`**: Gymnasium trading environment for backtesting
- **`core/live_trading_env.py`**: Live trading environment for MT5
- **`models/transformer.py`**: Temporal Fusion Transformer for price prediction with quantile outputs
- **`models/ppo_agent.py`**: PPO actor-critic agent (actions: HOLD, BUY, SELL, CLOSE)
- **`training/trainer.py`**: Model training orchestration
- **`training/online_learning.py`**: Online adaptation system (monitors performance, triggers model updates)
- **`evaluation/backtester.py`**: Backtesting engine with walk-forward optimization
- **`core/risk_manager.py`**: Position sizing (fixed/percent/Kelly/volatility), risk limits

### Data Flow

1. Market data (MT5/API) → DataPipeline → Feature Engineering (100+ features)
2. Features → Transformer Predictor → Price prediction + uncertainty quantiles
3. Features + Account state → PPO Agent → Trading action
4. Risk Manager validates trade → Execution
5. Online Learning monitors and adapts models

### Model Architecture

**Transformer Predictor**: Input projection → Positional encoding → N encoder layers → Temporal attention → Gated residual network → Point prediction + Quantile outputs (0.1, 0.5, 0.9)

**PPO Agent**: Shared feature extractor (256→256) → Actor head (128→4 actions) + Critic head (128→1 value)

## Key Patterns

- **Lazy Loading**: Components (data pipeline, models, trainers) initialized only when needed in `LeapTradingSystem`
- **Dataclass Configuration**: Hierarchical configs in `config/settings.py` with JSON serialization
- **Gymnasium Integration**: Trading environments follow Gymnasium standards for RL compatibility
- **Walk-Forward Optimization**: Rolling window train/test splits to prevent overfitting

## Extension Points

- **New Features**: Add computation in `FeatureEngineer._add_*()` methods in `core/data_pipeline.py`
- **New Models**: Implement same interface as existing models, add to `models/__init__.py`
- **New Risk Methods**: Add position sizing method in `RiskManager.calculate_position_size()`
- **New Metrics**: Add calculation in `MetricsCalculator` in `evaluation/metrics.py`

## Tech Stack

- PyTorch (deep learning), Gymnasium (RL environments)
- Pandas/NumPy (data), scikit-learn (ML utilities)
- MetaTrader5 (Windows only, optional for live trading)
- TensorBoard (training visualization)

## Important Notes

- NumPy must be <2.0 for PyTorch compatibility
- MetaTrader5 integration only works on Windows
- Always use `--paper` flag when testing live trading
- Models save to `saved_models/`, checkpoints to `checkpoints/`, results to `results/`
