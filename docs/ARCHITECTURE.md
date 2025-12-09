# Leap Trading System - Architecture Documentation

## Overview

Leap is designed as a modular, scalable trading system that combines deep learning for price prediction with reinforcement learning for trading decisions. The architecture emphasizes:

1. **Separation of Concerns**: Each component has a single responsibility
2. **Modularity**: Components can be swapped or upgraded independently
3. **Online Learning**: Continuous adaptation to market changes
4. **Risk-First Design**: Risk management at every layer

## System Components

### 1. Data Layer

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DataPipeline                                                │
│  ├── connect()           # Connect to data source           │
│  ├── fetch_historical()  # Get OHLCV data                   │
│  ├── prepare_sequences() # Create training sequences        │
│  └── get_online_batch()  # Stream live data                 │
│                                                              │
│  FeatureEngineer                                             │
│  ├── compute_all_features()                                  │
│  │   ├── _add_price_features()      # Returns, gaps, TR     │
│  │   ├── _add_moving_averages()     # SMA, EMA             │
│  │   ├── _add_momentum_indicators() # RSI, MACD, Stoch     │
│  │   ├── _add_volatility_indicators() # ATR, BB, Keltner   │
│  │   ├── _add_volume_indicators()   # OBV, VPT, MFI        │
│  │   ├── _add_trend_indicators()    # ADX, CCI             │
│  │   ├── _add_candlestick_patterns() # Doji, hammer, etc   │
│  │   └── _add_time_features()       # Hour, day, cyclical  │
│  │                                                          │
│  └── feature_names: List[str]  # 100+ computed features     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Model Layer

#### Transformer Predictor

The Temporal Fusion Transformer (TFT) is designed for multi-horizon time series forecasting with interpretability.

**Key Components**:

- **Positional Encoding**: Sinusoidal encoding for sequence position
- **Multi-Head Attention**: 8 attention heads with d_k = d_model / n_heads
- **Gated Residual Networks**: Feature selection with gating mechanism
- **Quantile Outputs**: Predictions at [0.1, 0.5, 0.9] quantiles for uncertainty

**Forward Pass**:
```
Input (batch, seq_len, features)
    │
    ▼
Input Projection → (batch, seq_len, d_model)
    │
    ▼
Positional Encoding
    │
    ▼
N × Transformer Encoder Layers
    │   ├── Multi-Head Self-Attention
    │   ├── Add & LayerNorm
    │   ├── Feed-Forward Network
    │   └── Add & LayerNorm
    │
    ▼
Temporal Attention (interpretable)
    │
    ▼
Gated Residual Network
    │
    ├──────────┬──────────┐
    ▼          ▼          ▼
Point      Quantile   Attention
Prediction Outputs    Weights
```

#### PPO Agent

Proximal Policy Optimization with:

- **Actor-Critic Architecture**: Shared feature extraction
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Generalized Advantage Estimation (GAE)**: Variance reduction
- **Entropy Bonus**: Encourages exploration

**Action Space**:
- 0: HOLD - No action
- 1: BUY - Open long position
- 2: SELL - Open short position
- 3: CLOSE - Close all positions

**State Space**:
- Market window: (window_size × (OHLCV + features))
- Account state: [balance, equity, position_type, unrealized_pnl, drawdown, total_pnl, etc.]

### 3. Training Layer

#### Offline Training

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Predictor Training                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │  1. Load historical data                           │     │
│  │  2. Feature engineering                            │     │
│  │  3. Create sequences (X, y)                        │     │
│  │  4. Train/Val/Test split (70/15/15)               │     │
│  │  5. Train Transformer with early stopping          │     │
│  │  6. Save best checkpoint                           │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  Phase 2: Agent Training                                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  1. Create trading environment                     │     │
│  │  2. Initialize PPO agent                           │     │
│  │  3. Collect rollouts (n_steps per update)          │     │
│  │  4. Compute advantages with GAE                    │     │
│  │  5. PPO update (n_epochs)                          │     │
│  │  6. Repeat for total_timesteps                     │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Online Learning

The online learning system monitors performance and triggers adaptation when:

1. **Prediction Error Threshold**: Mean error exceeds 5%
2. **Drawdown Threshold**: Drawdown exceeds 10%
3. **Regime Change**: Detected market regime shift
4. **Scheduled Updates**: Every N steps

**Adaptation Process**:
```python
1. Sample recent experiences from buffer
2. Use lower learning rate (1e-5 vs 1e-4)
3. Perform small number of update steps
4. Validate no performance degradation
5. Reset if catastrophic forgetting detected
```

### 4. Risk Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Pre-Trade Checks                                            │
│  ├── Position size within limits?                           │
│  ├── Total exposure within limits?                          │
│  ├── Risk/reward ratio acceptable?                          │
│  ├── Daily loss limit reached?                              │
│  └── Max consecutive losses reached?                        │
│                                                              │
│  Position Sizing Methods                                     │
│  ├── Fixed: Constant lot size                               │
│  ├── Percent: X% of account at risk                         │
│  ├── Kelly: f = (p*b - q) / b                               │
│  └── Volatility: Adjusted for ATR                           │
│                                                              │
│  Dynamic Adjustments                                         │
│  ├── Reduce size in high volatility                         │
│  ├── Reduce size after consecutive losses                   │
│  └── Increase size in favorable regime                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5. Evaluation Layer

#### Metrics Calculated

| Category | Metrics |
|----------|---------|
| Returns | Total return, Annualized return, CAGR |
| Risk | Volatility, Downside vol, Max DD, VaR, CVaR |
| Risk-Adjusted | Sharpe, Sortino, Calmar, Omega, Information ratio |
| Trade Stats | Win rate, Profit factor, Payoff ratio, Expectancy |
| Distribution | Skewness, Kurtosis, Tail ratio |

#### Walk-Forward Optimization

Prevents overfitting by:
1. Training on rolling windows
2. Testing on out-of-sample periods
3. Aggregating performance across all test periods
4. Measuring consistency (profitable folds ratio)

### 6. Live Trading Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE TRADING LAYER                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  MT5BrokerGateway (core/mt5_broker.py)                      │
│  ├── connect() / disconnect()                               │
│  ├── get_account_info()        # Balance, equity, margin    │
│  ├── get_current_tick()        # Real-time prices           │
│  ├── get_symbol_info()         # Symbol specifications      │
│  ├── send_market_order()       # Execute trades             │
│  ├── close_position()          # Close by ticket            │
│  └── modify_position()         # Update SL/TP               │
│                                                              │
│  OrderManager (core/order_manager.py)                       │
│  ├── execute_signal()          # Validate and execute       │
│  ├── validate_signal()         # Risk checks                │
│  ├── calculate_position_size() # Risk-based sizing          │
│  └── close_all_positions()     # Emergency close            │
│                                                              │
│  PositionSynchronizer (core/position_sync.py)               │
│  ├── sync()                    # Sync with broker           │
│  ├── get_positions()           # Current positions          │
│  ├── register_callback()       # Event notifications        │
│  └── Events: OPENED, CLOSED, SL_HIT, TP_HIT, MODIFIED      │
│                                                              │
│  LiveTradingEnvironment (core/live_trading_env.py)          │
│  ├── Gymnasium-compatible interface                         │
│  ├── Real-time observations from MT5                        │
│  ├── open_status control for position entry                 │
│  ├── Paper mode for safe testing                            │
│  └── Extends BaseTradingEnvironment                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7. Auto-Trader Layer

```
┌─────────────────────────────────────────────────────────────┐
│                    AUTO-TRADER SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  AutoTrader (core/auto_trader.py)                           │
│  ├── start() / stop()          # Lifecycle management       │
│  ├── pause() / resume()        # Trading control            │
│  ├── _trading_loop()           # Main autonomous loop       │
│  ├── _generate_signal()        # Combine models             │
│  ├── _check_adaptation()       # Online learning trigger    │
│  └── get_statistics()          # Session metrics            │
│                                                              │
│  Trading Loop (runs every bar):                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Check trading time and daily limits                │ │
│  │  2. Sync positions with broker                         │ │
│  │  3. For each symbol:                                   │ │
│  │     a. Check for new bar                               │ │
│  │     b. Get Transformer prediction                      │ │
│  │     c. Get PPO agent action                            │ │
│  │     d. Combine signals with confidence check           │ │
│  │     e. Validate with risk manager                      │ │
│  │     f. Execute via order manager                       │ │
│  │  4. Check for online learning adaptation               │ │
│  │  5. Sleep until next cycle                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  States: STOPPED → STARTING → RUNNING ⇄ PAUSED → STOPPING  │
│                                    ↓                         │
│                                  ERROR                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8. Trading Environment Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│              TRADING ENVIRONMENT ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  BaseTradingEnvironment (ABC)                               │
│  ├── Shared reward calculation                              │
│  ├── Shared statistics (Sharpe, Sortino, profit factor)     │
│  ├── History tracking (balance, equity, actions)            │
│  └── Abstract methods for subclass implementation           │
│           │                                                  │
│           ├──────────────────┬───────────────────┐          │
│           ▼                  ▼                   ▼          │
│  ┌─────────────────┐ ┌─────────────────┐                    │
│  │TradingEnvironment│ │LiveTradingEnv   │                    │
│  │  (Backtest)      │ │  (Live/Paper)   │                    │
│  ├─────────────────┤ ├─────────────────┤                    │
│  │ Historical data  │ │ MT5 real-time   │                    │
│  │ Simulated exec   │ │ Real execution  │                    │
│  │ Fast iteration   │ │ Position sync   │                    │
│  └─────────────────┘ │ Open status ctrl│                    │
│                       └─────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Diagrams

### Training Flow

```
Historical Data
      │
      ▼
┌─────────────┐
│ DataPipeline│
├─────────────┤
│ fetch_data  │
│ engineer    │
│ sequences   │
└──────┬──────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
┌─────────────┐                 ┌─────────────┐
│ Transformer │                 │   Trading   │
│  Training   │                 │ Environment │
├─────────────┤                 ├─────────────┤
│ X_train     │                 │ OHLCV data  │
│ y_train     │                 │ Features    │
│ epochs      │                 └──────┬──────┘
└──────┬──────┘                        │
       │                               ▼
       │                    ┌─────────────┐
       │                    │    PPO      │
       │                    │  Training   │
       │                    ├─────────────┤
       │                    │ timesteps   │
       │                    └──────┬──────┘
       │                           │
       └───────────┬───────────────┘
                   ▼
           ┌─────────────┐
           │ ModelTrainer│
           ├─────────────┤
           │ save_all()  │
           └─────────────┘
```

### Inference Flow

```
Live Market Data
      │
      ▼
┌─────────────┐
│ DataPipeline│
│ get_online  │
└──────┬──────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
┌─────────────┐                 ┌─────────────┐
│ Transformer │                 │ PPO Agent   │
│  predict()  │                 │ select_act  │
├─────────────┤                 ├─────────────┤
│ prediction  │                 │ action      │
│ uncertainty │                 │ confidence  │
└──────┬──────┘                 └──────┬──────┘
       │                               │
       └───────────┬───────────────────┘
                   ▼
           ┌─────────────┐
           │Risk Manager │
           │ check_trade │
           └──────┬──────┘
                  │
           ┌──────┴──────┐
           ▼             ▼
      Approve        Reject
           │
           ▼
    ┌─────────────┐
    │  Execute    │
    │   Trade     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Online    │
    │  Learning   │
    │   Update    │
    └─────────────┘
```

### Auto-Trading Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-TRADER FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   AutoTrader.start()                                            │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │   Connect   │                                               │
│   │  to MT5     │                                               │
│   └──────┬──────┘                                               │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  TRADING LOOP                            │   │
│   │  ┌─────────────────────────────────────────────────────┐│   │
│   │  │                                                     ││   │
│   │  │  ┌──────────┐    ┌──────────┐    ┌──────────┐      ││   │
│   │  │  │ Position │───▶│  Check   │───▶│   New    │      ││   │
│   │  │  │   Sync   │    │  Limits  │    │   Bar?   │      ││   │
│   │  │  └──────────┘    └──────────┘    └────┬─────┘      ││   │
│   │  │                                        │            ││   │
│   │  │                                        ▼            ││   │
│   │  │  ┌──────────────────────────────────────────────┐  ││   │
│   │  │  │           SIGNAL GENERATION                   │  ││   │
│   │  │  │                                               │  ││   │
│   │  │  │  Observation ──▶ Transformer ──▶ Prediction   │  ││   │
│   │  │  │       │                              │         │  ││   │
│   │  │  │       └──────▶ PPO Agent ───▶ Action │         │  ││   │
│   │  │  │                              │       │         │  ││   │
│   │  │  │                              ▼       ▼         │  ││   │
│   │  │  │                      ┌──────────────────┐      │  ││   │
│   │  │  │                      │ Combine Signals  │      │  ││   │
│   │  │  │                      │ (confidence check)│      │  ││   │
│   │  │  │                      └────────┬─────────┘      │  ││   │
│   │  │  └───────────────────────────────┼───────────────┘  ││   │
│   │  │                                   │                  ││   │
│   │  │                                   ▼                  ││   │
│   │  │                         ┌──────────────────┐        ││   │
│   │  │                         │  Order Manager   │        ││   │
│   │  │                         │  (validate/exec) │        ││   │
│   │  │                         └────────┬─────────┘        ││   │
│   │  │                                   │                  ││   │
│   │  │         ┌─────────────────────────┴──────────────┐  ││   │
│   │  │         ▼                                        ▼  ││   │
│   │  │  ┌──────────────┐                    ┌──────────────┐││   │
│   │  │  │   Execute    │                    │    Reject    │││   │
│   │  │  │    Trade     │                    │   (log why)  │││   │
│   │  │  └──────┬───────┘                    └──────────────┘││   │
│   │  │         │                                            ││   │
│   │  │         ▼                                            ││   │
│   │  │  ┌──────────────┐                                   ││   │
│   │  │  │   Online     │                                   ││   │
│   │  │  │  Learning    │  (after N trades)                 ││   │
│   │  │  │  Adaptation  │                                   ││   │
│   │  │  └──────────────┘                                   ││   │
│   │  │                                                     ││   │
│   │  └─────────────────────────────────────────────────────┘│   │
│   │                           │                              │   │
│   │                     sleep(interval)                      │   │
│   │                           │                              │   │
│   │                     loop until stop                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration System

The configuration uses dataclasses for type safety:

```python
@dataclass
class SystemConfig:
    # Sub-configurations
    data: DataConfig           # Data pipeline settings
    transformer: TransformerConfig  # Transformer model settings
    ppo: PPOConfig             # PPO agent settings
    risk: RiskConfig           # Risk management settings
    backtest: BacktestConfig   # Backtesting settings
    evaluation: EvaluationConfig  # Metrics and evaluation
    logging: LoggingConfig     # Logging configuration
    auto_trader: AutoTraderConfig  # Auto-trader settings

    # System settings
    trading_mode: str = "backtest"
    device: str = "auto"
    seed: int = 42
```

### Configuration Dataclasses

| Config | Purpose |
|--------|---------|
| `DataConfig` | Symbols, timeframes, lookback window, feature engineering |
| `TransformerConfig` | Model dimensions, layers, learning rates |
| `PPOConfig` | Actor-critic architecture, PPO hyperparameters |
| `RiskConfig` | Position sizing, stop loss, drawdown limits |
| `BacktestConfig` | Initial balance, commission, walk-forward settings |
| `EvaluationConfig` | Metrics to calculate, benchmark symbol |
| `LoggingConfig` | Log level, file rotation, format |
| `AutoTraderConfig` | Trading hours, risk per trade, online learning |

Configurations can be saved/loaded as JSON:

```python
config = get_config()
config.save("my_config.json")

loaded_config = SystemConfig.load("my_config.json")
```

### Logging Configuration

```python
@dataclass
class LoggingConfig:
    level: str = "INFO"           # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True      # Enable file logging
    log_to_console: bool = True   # Enable console logging
    max_file_size_mb: int = 10    # Rotation size
    backup_count: int = 5         # Number of backups
    rotation_type: str = "size"   # "size" or "time"
```

### Auto-Trader Configuration

```python
@dataclass
class AutoTraderConfig:
    symbols: List[str] = ['EURUSD']
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_positions: int = 3
    max_daily_loss: float = 0.05  # 5% daily loss limit
    paper_mode: bool = True       # Safe testing mode
    enable_online_learning: bool = True
    trading_days: List[int] = [0, 1, 2, 3, 4]  # Mon-Fri
```

## Extension Points

### Adding New Features

1. Add computation in `FeatureEngineer._add_*()` method
2. Feature will automatically be included in feature_names

### Adding New Models

1. Implement with same interface as existing models
2. Add to `models/__init__.py`
3. Update `LeapTradingSystem.initialize_models()`

### Adding New Risk Methods

1. Add position sizing method in `RiskManager.calculate_position_size()`
2. Update `PositionSizing.method` enum

### Adding New Metrics

1. Add calculation in `MetricsCalculator`
2. Include in `calculate_all()` method

### Adding New Trading Environments

1. Extend `BaseTradingEnvironment`
2. Implement abstract methods:
   - `_get_current_price()`
   - `_get_market_observation()`
   - `_open_position()` / `_close_position()`
   - `_get_open_positions()`
   - `_has_position()` / `_get_unrealized_pnl()`

### Extending Auto-Trader

1. **Custom Signal Sources**: Override `_generate_signal()` in `AutoTrader`
2. **Custom Callbacks**: Register via `trader.register_callback(event, handler)`
   - Events: `on_trade`, `on_signal`, `on_error`, `on_adaptation`, `on_state_change`
3. **Custom Broker**: Implement `MT5BrokerGateway` interface for other brokers
4. **Custom Risk Checks**: Extend `OrderManager.validate_signal()`

## Performance Considerations

- **GPU Acceleration**: Models automatically use CUDA if available
- **Batch Processing**: Data is processed in batches for efficiency
- **Parallel Walk-Forward**: Uses ThreadPoolExecutor for parallel folds
- **Lazy Loading**: Components initialized only when needed
- **Experience Replay**: Efficient sampling from replay buffer
