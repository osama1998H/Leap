# Leap Trading System

An advanced AI-powered forex trading system with online learning and adaptation capabilities. Combines Transformer-based price prediction with PPO reinforcement learning for intelligent trading decisions.

## Features

- **Transformer-based Price Prediction**: Temporal Fusion Transformer for accurate time-series forecasting with uncertainty estimation
- **PPO Reinforcement Learning**: Proximal Policy Optimization agent for optimal trading decisions
- **Online Learning**: Continuous model adaptation to changing market conditions
- **Market Regime Detection**: Automatic detection of market states (trending, ranging, high volatility)
- **Comprehensive Backtesting**: Walk-forward optimization and Monte Carlo simulation
- **Risk Management**: Dynamic position sizing, stop-loss/take-profit, and drawdown limits
- **Multi-timeframe Analysis**: Feature engineering across multiple timeframes

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LEAP TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Market     │───▶│    Data      │───▶│   Feature    │                   │
│  │    Data      │    │   Pipeline   │    │  Engineering │                   │
│  │  (MT5/API)   │    │              │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                           │
│                      ┌───────────────────────────┼───────────────────────┐  │
│                      │                           ▼                       │  │
│                      │  ┌─────────────────────────────────────────────┐ │  │
│                      │  │              AI MODELS LAYER                 │ │  │
│                      │  ├─────────────────────┬───────────────────────┤ │  │
│                      │  │                     │                       │ │  │
│                      │  │  ┌───────────────┐  │  ┌─────────────────┐  │ │  │
│                      │  │  │  Transformer  │  │  │   PPO Agent     │  │ │  │
│                      │  │  │  Predictor    │  │  │  (RL Trading)   │  │ │  │
│                      │  │  │               │  │  │                 │  │ │  │
│                      │  │  │ - Attention   │  │  │ - Actor-Critic  │  │ │  │
│                      │  │  │ - Quantiles   │  │  │ - GAE           │  │ │  │
│                      │  │  │ - Uncertainty │  │  │ - Clipped Loss  │  │ │  │
│                      │  │  └───────┬───────┘  │  └────────┬────────┘  │ │  │
│                      │  │          │          │           │           │ │  │
│                      │  └──────────┼──────────┴───────────┼───────────┘ │  │
│                      │             │                      │             │  │
│                      │             ▼                      ▼             │  │
│                      │  ┌──────────────────────────────────────────┐   │  │
│                      │  │         ONLINE LEARNING MANAGER          │   │  │
│                      │  │                                          │   │  │
│                      │  │  - Performance Monitoring                │   │  │
│                      │  │  - Regime Detection                      │   │  │
│                      │  │  - Adaptive Model Updates                │   │  │
│                      │  │  - Catastrophic Forgetting Prevention    │   │  │
│                      │  └──────────────────────────────────────────┘   │  │
│                      └─────────────────────────────────────────────────┘  │
│                                          │                                 │
│                                          ▼                                 │
│                      ┌─────────────────────────────────────────────────┐  │
│                      │              RISK MANAGEMENT                     │  │
│                      │                                                  │  │
│                      │  - Position Sizing (Kelly/Volatility/Fixed)     │  │
│                      │  - Stop Loss / Take Profit                      │  │
│                      │  - Drawdown Limits                              │  │
│                      │  - Daily/Weekly Loss Limits                     │  │
│                      └─────────────────────────────────────────────────┘  │
│                                          │                                 │
│                                          ▼                                 │
│                      ┌─────────────────────────────────────────────────┐  │
│                      │              EXECUTION LAYER                     │  │
│                      │                                                  │  │
│                      │  - Backtest Engine                              │  │
│                      │  - Paper Trading                                │  │
│                      │  - Live Trading (MT5)                           │  │
│                      └─────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Model Architecture

### Transformer Predictor

```
┌────────────────────────────────────────────────────────────────┐
│                  TEMPORAL FUSION TRANSFORMER                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input: (batch, seq_len, features)                            │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                     │
│   │   Input Projection   │  Linear(features → d_model)         │
│   └──────────┬───────────┘                                     │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                     │
│   │ Positional Encoding  │  Sinusoidal position embedding      │
│   └──────────┬───────────┘                                     │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                     │
│   │  Transformer Encoder │  N × [Multi-Head Attention          │
│   │      (N layers)      │       + Feed Forward + LayerNorm]   │
│   └──────────┬───────────┘                                     │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                     │
│   │  Temporal Attention  │  Interpretable attention weights    │
│   └──────────┬───────────┘                                     │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                     │
│   │ Gated Residual Net   │  Feature gating for selection       │
│   └──────────┬───────────┘                                     │
│              │                                                  │
│      ┌───────┴───────┐                                         │
│      ▼               ▼                                         │
│ ┌─────────┐   ┌───────────┐                                    │
│ │  Point  │   │ Quantile  │  [0.1, 0.5, 0.9] for uncertainty  │
│ │ Predict │   │ Outputs   │                                    │
│ └─────────┘   └───────────┘                                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### PPO Agent

```
┌────────────────────────────────────────────────────────────────┐
│                      PPO ACTOR-CRITIC                           │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   State: (market_features + account_state)                     │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                     │
│   │  Shared Feature      │  MLP: 256 → 256 (LayerNorm)         │
│   │    Extractor         │                                     │
│   └──────────┬───────────┘                                     │
│              │                                                  │
│      ┌───────┴───────┐                                         │
│      ▼               ▼                                         │
│ ┌─────────┐   ┌───────────┐                                    │
│ │  Actor  │   │  Critic   │                                    │
│ │  Head   │   │   Head    │                                    │
│ │         │   │           │                                    │
│ │ 128→4   │   │  128→1    │                                    │
│ └────┬────┘   └─────┬─────┘                                    │
│      │              │                                          │
│      ▼              ▼                                          │
│ ┌─────────┐   ┌───────────┐                                    │
│ │ Action  │   │   State   │                                    │
│ │  Probs  │   │   Value   │                                    │
│ │ [H,B,S,C]│   │    V(s)   │                                    │
│ └─────────┘   └───────────┘                                    │
│                                                                 │
│   Actions: HOLD(0), BUY(1), SELL(2), CLOSE(3)                  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DATA INGESTION                                              │
│  ┌──────────┐    ┌────────────┐    ┌────────────────┐          │
│  │ MetaTrader│───▶│  OHLCV    │───▶│  DataFrame     │          │
│  │    5      │    │  Rates    │    │  with Index    │          │
│  └──────────┘    └────────────┘    └───────┬────────┘          │
│                                             │                    │
│  2. FEATURE ENGINEERING                     ▼                    │
│  ┌────────────────────────────────────────────────────┐         │
│  │                                                    │         │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │         │
│  │  │ Price   │ │Momentum │ │Volatility│ │ Volume │  │         │
│  │  │Features │ │  RSI    │ │  ATR    │ │  OBV   │  │         │
│  │  │         │ │  MACD   │ │  BBands │ │  MFI   │  │         │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │         │
│  │       │           │           │           │        │         │
│  │       └───────────┴─────┬─────┴───────────┘        │         │
│  │                         ▼                          │         │
│  │              ┌─────────────────┐                   │         │
│  │              │  100+ Features  │                   │         │
│  │              └────────┬────────┘                   │         │
│  └───────────────────────┼────────────────────────────┘         │
│                          │                                       │
│  3. SEQUENCE CREATION    ▼                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │  Sliding Window (120 timesteps)             │                │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┐     │                │
│  │  │ t │t+1│t+2│...│   │   │   │   │t+119    │                │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┘     │                │
│  │           │                                 │                │
│  │           ▼                                 │                │
│  │  Shape: (samples, 120, features)           │                │
│  └─────────────────────────────────────────────┘                │
│                          │                                       │
│  4. NORMALIZATION        ▼                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │  RobustScaler (handles outliers)            │                │
│  │  Per-feature normalization                  │                │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Online Learning System

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONLINE LEARNING LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────────────────────────────────────────────────┐      │
│    │                  LIVE MARKET                         │      │
│    └───────────────────────┬─────────────────────────────┘      │
│                            │                                     │
│                            ▼                                     │
│    ┌─────────────────────────────────────────────────────┐      │
│    │              PERFORMANCE MONITOR                     │      │
│    │                                                      │      │
│    │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │      │
│    │  │ Prediction   │  │   Trading    │  │  Regime   │  │      │
│    │  │   Errors     │  │    P&L       │  │ Detection │  │      │
│    │  └──────────────┘  └──────────────┘  └───────────┘  │      │
│    └───────────────────────┬─────────────────────────────┘      │
│                            │                                     │
│                            ▼                                     │
│    ┌─────────────────────────────────────────────────────┐      │
│    │            ADAPTATION TRIGGER CHECK                  │      │
│    │                                                      │      │
│    │   - Error > threshold?                              │      │
│    │   - Drawdown > limit?                               │      │
│    │   - Regime changed?                                 │      │
│    │   - Scheduled update?                               │      │
│    └───────────────────────┬─────────────────────────────┘      │
│                            │                                     │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│    ┌──────────────────┐       ┌──────────────────┐              │
│    │  NO ADAPTATION   │       │  ADAPTATION      │              │
│    │  Continue as-is  │       │  TRIGGERED       │              │
│    └──────────────────┘       └────────┬─────────┘              │
│                                        │                         │
│                                        ▼                         │
│                        ┌───────────────────────────┐            │
│                        │    ONLINE UPDATE          │            │
│                        │                           │            │
│                        │  1. Sample recent data    │            │
│                        │  2. Lower learning rate   │            │
│                        │  3. Update predictor      │            │
│                        │  4. Update RL agent       │            │
│                        │  5. Validate performance  │            │
│                        └───────────────────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Backtesting Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    WALK-FORWARD OPTIMIZATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Full Historical Data                                          │
│   ├───────────────────────────────────────────────────────────┤ │
│   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│   Split 1:  [======TRAIN======][TEST]                           │
│   Split 2:       [======TRAIN======][TEST]                      │
│   Split 3:            [======TRAIN======][TEST]                 │
│   Split 4:                 [======TRAIN======][TEST]            │
│   ...                                                            │
│                                                                  │
│   For each split:                                                │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  1. Train models on TRAIN window                       │    │
│   │  2. Freeze models                                      │    │
│   │  3. Run backtest on TEST window                        │    │
│   │  4. Record out-of-sample performance                   │    │
│   │  5. Slide window forward                               │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
│   Aggregate Results:                                             │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  - Mean return across all TEST periods                 │    │
│   │  - Sharpe ratio consistency                            │    │
│   │  - Worst-case drawdown                                 │    │
│   │  - Profitable folds ratio                              │    │
│   └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
Leap/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
│
├── core/
│   ├── __init__.py
│   ├── data_pipeline.py     # Data fetching & feature engineering
│   ├── trading_env.py       # Gymnasium trading environment
│   └── risk_manager.py      # Position sizing & risk limits
│
├── models/
│   ├── __init__.py
│   ├── transformer.py       # Temporal Fusion Transformer
│   └── ppo_agent.py         # PPO reinforcement learning
│
├── training/
│   ├── __init__.py
│   ├── trainer.py           # Model training orchestration
│   └── online_learning.py   # Online adaptation system
│
├── evaluation/
│   ├── __init__.py
│   ├── backtester.py        # Backtesting engine
│   └── metrics.py           # Performance metrics
│
├── tests/
│   ├── __init__.py
│   └── test_integration.py  # Integration tests
│
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Leap.git
cd Leap

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For MetaTrader 5 integration (Windows only)
pip install MetaTrader5
```

## Usage

### Training

```bash
# Train both prediction and RL models
python main.py train --symbol EURUSD --timeframe 1h --epochs 100 --timesteps 100000

# Train with custom configuration
python main.py train --config config/custom_config.json
```

### Backtesting

```bash
# Run backtest on historical data
python main.py backtest --symbol EURUSD --bars 50000

# Walk-forward optimization
python main.py walkforward --symbol EURUSD
```

### Evaluation

```bash
# Evaluate trained models
python main.py evaluate --model-dir ./saved_models
```

### Live Trading

```bash
# Paper trading (simulated)
python main.py live --paper

# Live trading (requires MT5 connection)
python main.py live
```

### Running Tests

```bash
# Run integration tests
python -m pytest tests/test_integration.py -v

# Or directly
python tests/test_integration.py
```

## Configuration

Key configuration parameters in `config/settings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Transformer model dimension |
| `n_heads` | 8 | Attention heads |
| `n_encoder_layers` | 4 | Transformer encoder layers |
| `lookback_window` | 120 | Input sequence length |
| `prediction_horizon` | 12 | Prediction steps ahead |
| `learning_rate` | 1e-4 | Predictor learning rate |
| `ppo.gamma` | 0.99 | RL discount factor |
| `ppo.clip_epsilon` | 0.2 | PPO clip range |
| `risk.max_drawdown` | 0.15 | Maximum allowed drawdown |

## Performance Metrics

The system calculates comprehensive metrics:

- **Return Metrics**: Total return, CAGR, annualized return
- **Risk Metrics**: Volatility, max drawdown, VaR, CVaR
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Statistics**: Win rate, profit factor, payoff ratio
- **Distribution**: Skewness, kurtosis, tail ratio

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always use paper trading to test strategies before risking real capital.
