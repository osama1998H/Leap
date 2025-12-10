# Web UI Requirements

## Goals

The Leap Trading System Web UI aims to:

1. **Accelerate training and testing workflows** - Reduce time from idea to validated model
2. **Reduce CLI complexity** - Provide visual interfaces for common operations with sensible defaults
3. **Provide visual feedback** - Real-time monitoring of long-running processes (training, backtesting)
4. **Enable easier experiment comparison** - Side-by-side model and strategy evaluation
5. **Lower barrier to entry** - Allow users unfamiliar with CLI to effectively use the system

## User Stories

### As a Researcher, I want to...

- **US-R1**: Configure and launch training runs with different hyperparameters without memorizing CLI flags
- **US-R2**: Monitor training progress in real-time with loss curves and metrics visualizations
- **US-R3**: Compare multiple training runs side-by-side to identify best configurations
- **US-R4**: Save and reuse training configurations as templates for reproducible experiments
- **US-R5**: View attention weights and feature importance from the Transformer model
- **US-R6**: Run walk-forward optimization and visualize per-fold performance

### As an Engineer, I want to...

- **US-E1**: Quickly run backtests with different parameters (realistic mode, Monte Carlo)
- **US-E2**: View detailed trade-by-trade analysis with entry/exit visualizations
- **US-E3**: Browse and download model checkpoints for deployment
- **US-E4**: Monitor system resource utilization during training (GPU, memory)
- **US-E5**: View and search through application logs without SSHing to the server
- **US-E6**: Configure risk management parameters and see their impact on backtests

### As a Trader, I want to...

- **US-T1**: Review backtest results with comprehensive performance metrics
- **US-T2**: Understand strategy behavior through trade statistics and equity curves
- **US-T3**: Configure auto-trader settings with clear explanations of each parameter
- **US-T4**: Monitor paper trading sessions in real-time (if running)
- **US-T5**: Export backtest results and reports for external analysis

---

## Functional Requirements

### FR1: Training Management

| ID | Requirement | Priority | CLI Mapping |
|----|-------------|----------|-------------|
| FR1.1 | Configure and launch training runs with all CLI options | High | `train` command |
| FR1.2 | Support multi-symbol training configuration | High | `--symbols` |
| FR1.3 | Support multi-timeframe feature configuration | High | `--multi-timeframe` |
| FR1.4 | Monitor training progress in real-time (loss curves) | High | Log streaming |
| FR1.5 | Display Transformer and PPO training phases separately | Medium | Training pipeline |
| FR1.6 | Pause/resume/stop training runs | Medium | Process control |
| FR1.7 | Compare multiple training runs side-by-side | Medium | MLflow integration |
| FR1.8 | View and download checkpoints | High | `saved_models/` |
| FR1.9 | Save/load training configuration templates | Medium | Config JSON files |
| FR1.10 | View training history and past runs | Medium | MLflow/logs |

### FR2: Testing & Evaluation

| ID | Requirement | Priority | CLI Mapping |
|----|-------------|----------|-------------|
| FR2.1 | Run backtests with configurable parameters | High | `backtest` command |
| FR2.2 | Enable realistic mode constraints | High | `--realistic` |
| FR2.3 | Run Monte Carlo simulation for risk analysis | High | `--monte-carlo` |
| FR2.4 | Display comprehensive backtest metrics | High | `BacktestResult` |
| FR2.5 | Visualize equity curve and drawdown | High | Result data |
| FR2.6 | View trade-by-trade breakdown | High | `trades` list |
| FR2.7 | Run walk-forward optimization | Medium | `walkforward` command |
| FR2.8 | Display per-fold results with aggregated statistics | Medium | WF results |
| FR2.9 | Run model evaluation on test data | Medium | `evaluate` command |
| FR2.10 | Compare backtest results across different configurations | Medium | Multi-select |

### FR3: Configuration Management

| ID | Requirement | Priority | CLI Mapping |
|----|-------------|----------|-------------|
| FR3.1 | Create/edit system configuration | High | `config/*.json` |
| FR3.2 | Configure data settings (symbols, timeframes, features) | High | `DataConfig` |
| FR3.3 | Configure Transformer architecture and training | High | `TransformerConfig` |
| FR3.4 | Configure PPO agent hyperparameters | High | `PPOConfig` |
| FR3.5 | Configure risk management parameters | High | `RiskConfig` |
| FR3.6 | Configure backtest settings | High | `BacktestConfig` |
| FR3.7 | Save configurations as reusable templates | Medium | JSON export |
| FR3.8 | Import/export configurations | Medium | File upload/download |
| FR3.9 | Provide hyperparameter presets (conservative, aggressive) | Low | Pre-built configs |
| FR3.10 | Validate configuration before use | High | Schema validation |

### FR4: Monitoring & Visualization

| ID | Requirement | Priority | CLI Mapping |
|----|-------------|----------|-------------|
| FR4.1 | Display real-time loss curves during training | High | Training logs |
| FR4.2 | Show resource utilization (GPU memory, CPU) | Medium | System metrics |
| FR4.3 | Provide live log viewer with filtering | High | `logs/` directory |
| FR4.4 | Browse saved model artifacts | High | `saved_models/` |
| FR4.5 | Display MLflow experiment tracking data | Medium | MLflow integration |
| FR4.6 | Visualize performance metrics with charts | High | Metrics data |
| FR4.7 | Show regime analysis from backtests | Medium | Regime results |
| FR4.8 | Display Monte Carlo simulation distributions | Medium | MC results |

### FR5: Auto-Trader Management (Future Phase)

| ID | Requirement | Priority | CLI Mapping |
|----|-------------|----------|-------------|
| FR5.1 | Configure auto-trader settings | Low | `AutoTraderConfig` |
| FR5.2 | Start/stop paper trading sessions | Low | `autotrade --paper` |
| FR5.3 | Monitor live trading status | Low | Trader state |
| FR5.4 | View trade history and P&L | Low | Session stats |
| FR5.5 | Configure trading hours and limits | Low | Config settings |

---

## Non-Functional Requirements

### NFR1: Performance

| ID | Requirement | Target |
|----|-------------|--------|
| NFR1.1 | Initial page load time | < 3 seconds |
| NFR1.2 | Real-time metric updates | < 1 second latency |
| NFR1.3 | Chart rendering with 10k+ data points | < 500ms |
| NFR1.4 | API response time for config operations | < 200ms |
| NFR1.5 | Support concurrent training monitoring | 5+ simultaneous runs |

### NFR2: Reliability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR2.1 | WebSocket reconnection | Auto-reconnect within 5s |
| NFR2.2 | Form data persistence | Survive page refresh |
| NFR2.3 | Error recovery | Graceful degradation with error messages |
| NFR2.4 | Training job resilience | Jobs continue if UI disconnects |

### NFR3: Usability

| ID | Requirement | Target |
|----|-------------|--------|
| NFR3.1 | Keyboard navigation | Full keyboard accessibility |
| NFR3.2 | Mobile responsiveness | Functional on tablet (monitoring) |
| NFR3.3 | Dark/light mode | User-selectable theme |
| NFR3.4 | Form validation | Real-time with clear error messages |
| NFR3.5 | Loading states | Skeleton loaders for all async operations |

### NFR4: Browser Support

| Browser | Minimum Version |
|---------|-----------------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

### NFR5: Accessibility

| ID | Requirement | Standard |
|----|-------------|----------|
| NFR5.1 | WCAG compliance | Level AA |
| NFR5.2 | Screen reader support | ARIA labels on all interactive elements |
| NFR5.3 | Color contrast | 4.5:1 minimum for text |
| NFR5.4 | Focus indicators | Visible focus states |

---

## Constraints

1. **Backend API Required**: UI requires a Python backend API to be developed (FastAPI recommended)
2. **WebSocket Support**: Real-time features require WebSocket server implementation
3. **File System Access**: Backend needs access to `saved_models/`, `logs/`, `results/` directories
4. **GPU Monitoring**: Resource monitoring requires additional system dependencies (nvidia-smi)
5. **MLflow Integration**: Experiment tracking features depend on MLflow being enabled
6. **Windows-Only Live Trading**: Auto-trader features only functional on Windows (MT5 dependency)

---

## Out of Scope (v1.0)

1. Live trading controls (beyond paper mode monitoring)
2. Multi-user authentication and authorization
3. Cloud deployment and scaling
4. Model serving/inference API
5. Mobile native applications
6. Automated hyperparameter optimization UI
7. Custom strategy code editor
8. Alerting and notifications system

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Time to launch training run | < 2 minutes (vs 5+ min CLI) |
| Training monitoring setup | 0 additional steps (automatic) |
| Backtest comparison time | < 30 seconds for 5 runs |
| New user onboarding time | < 15 minutes to first successful run |
| Error rate in configuration | < 5% (vs higher with manual JSON) |
