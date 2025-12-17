# ADR 0006: Backtest-Agent Integration

## Status

Accepted

## Context

The backtester was using only Transformer predictions with hardcoded thresholds to make trading decisions:

```python
# Old behavior (main.py backtest)
if pred_return > 0.001:
    return {'action': 'buy', ...}
elif pred_return < -0.001:
    return {'action': 'sell', ...}
```

Meanwhile, live trading via `AutoTrader` uses both the Transformer predictor AND the PPO agent with signal combination logic:

```python
# AutoTrader behavior
signal_type = self._combine_signals(
    predicted_return=predicted_return,
    agent_action=agent_action,  # From PPO agent
    confidence=prediction_confidence,
    symbol=symbol
)
```

This architectural inconsistency meant:
1. **Backtest results did not predict live performance** - Testing one strategy, deploying another
2. **PPO agent was never validated** - Hours of training, but no historical evaluation
3. **Signal combination logic was untested** - Production code never ran against historical data

## Decision

Align the backtester strategy with `AutoTrader._combine_signals()` by:

1. **Passing PPO agent to `backtester.run()`**
2. **Building PPO-compatible observations** from backtest state (market window + account features)
3. **Using same signal combination logic** as AutoTrader:
   - Agent CLOSE → always CLOSE
   - Agent BUY + prediction agrees → BUY
   - Agent BUY + prediction contradicts → HOLD
   - Agent BUY + weak prediction → trust agent (BUY)
   - Same pattern for SELL

The observation structure matches `TradingEnvironment`:
- Market observation: `(window_size * n_features)` normalized
- Account observation: 8 features (balance ratio, equity ratio, positions, PnL, etc.)

## Consequences

### Positive

- **Backtest results now predict live performance** - Same strategy in both contexts
- **PPO agent's learned policy is validated** before deployment
- **Signal combination logic is tested** against historical data
- **Increased confidence** in model performance estimates
- **Clearer logging** indicates which models are being used

### Negative

- **Backtester strategy is more complex** - More code to maintain
- **Requires both models for full backtest** - Transformer-only backtest still works but with degraded behavior
- **Observation building adds overhead** - Minor performance impact

## Alternatives Considered

1. **Create a separate backtest mode for agent-only testing** - Rejected because it doesn't test the combined system
2. **Extract signal combination to a utility function** - Considered but deferred; current duplication is acceptable and keeps modules decoupled
3. **Use TradingEnvironment for backtesting** - Rejected because TradingEnvironment's step semantics don't match backtester's bar-by-bar simulation

## Code References

- `main.py:backtest()` - Strategy function with signal combination (lines 825-964)
- `core/auto_trader.py:_combine_signals()` - Reference implementation (lines 626-674)
- `core/trading_env_base.py:_get_account_observation()` - Account observation format (lines 414-428)
