# ADR 0011: Trading Strategy Pattern

## Status
Accepted

## Context

The system has duplicated signal generation logic in multiple places:

1. **`AutoTrader._combine_signals()`** - Combines predictor + agent outputs for live trading
2. **`cli/system.py` backtest function** - Inline strategy for backtesting
3. **`evaluation/backtester.py`** - Accepts callable strategies with no formal interface

Problems:
- **Code duplication**: Same combination logic exists in multiple files
- **No formal interface**: Strategies are ad-hoc callables with implicit signatures
- **Inconsistent signals**: Live and backtest may diverge due to separate implementations
- **Hard to test**: No isolated strategy unit testing
- **Hard to extend**: Adding new strategies requires understanding multiple files

## Decision

Introduce a `TradingStrategy` Abstract Base Class with formalized signal generation:

### 1. TradingStrategy ABC (`core/strategy.py`)

```python
class TradingStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass

    @abstractmethod
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        positions: List[Trade],
        **kwargs
    ) -> StrategySignal:
        """Generate trading signal from market data."""
        pass

    def on_trade_opened(self, trade: Trade) -> None:
        """Callback when trade opens. Override if needed."""
        pass

    def on_trade_closed(self, trade: Trade) -> None:
        """Callback when trade closes. Override if needed."""
        pass

    def reset(self) -> None:
        """Reset strategy state between sessions."""
        pass
```

### 2. StrategySignal Dataclass

```python
@dataclass
class StrategySignal:
    action: SignalType
    symbol: str = ""
    confidence: float = 1.0
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    risk_percent: Optional[float] = None
    predicted_return: float = 0.0
    agent_action: Optional[Action] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_backtest_dict(self) -> Dict[str, Any]:
        """Convert to backtester-compatible format."""
        ...

    def to_trading_signal(self) -> TradingSignal:
        """Convert to TradingSignal for order execution."""
        ...

    @property
    def is_entry(self) -> bool: ...
    @property
    def is_exit(self) -> bool: ...
    @property
    def is_hold(self) -> bool: ...
```

### 3. CombinedPredictorAgentStrategy

Default implementation combining Transformer + PPO:

```python
class CombinedPredictorAgentStrategy(TradingStrategy):
    def __init__(self, predictor=None, agent=None, config=None, feature_names=None):
        ...

    @property
    def name(self) -> str:
        return "combined_predictor_agent"

    def generate_signal(self, market_data, positions, **kwargs) -> StrategySignal:
        # 1. Get Transformer prediction
        # 2. Get PPO agent action
        # 3. Combine signals with validation
        ...
```

### 4. CallableStrategyAdapter

Backward compatibility for existing callable strategies:

```python
class CallableStrategyAdapter(TradingStrategy):
    def __init__(self, callable_fn: Callable, strategy_name: str = "callable_strategy"):
        self._callable = callable_fn
        self._name = strategy_name

    def generate_signal(self, market_data, positions, **kwargs) -> StrategySignal:
        result = self._callable(market_data, **kwargs)
        # Convert dict result to StrategySignal
        ...
```

### 5. Factory Function

```python
def create_strategy(
    strategy_type: str = 'combined',
    predictor=None,
    agent=None,
    config: Optional[StrategyConfig] = None,
    **kwargs
) -> TradingStrategy:
    """Factory for strategy creation."""
```

## Consequences

### Positive
- **Single source of truth**: Signal combination logic in one place
- **Formal interface**: Clear contract for strategy implementations
- **Testability**: Strategies can be unit tested in isolation
- **Consistency**: Same strategy used in backtest and live trading
- **Extensibility**: Easy to add new strategies (e.g., rule-based, ML-only)
- **Backward compatible**: `CallableStrategyAdapter` wraps existing callables

### Negative
- **Additional abstraction**: More indirection
- **Migration effort**: Need to update Backtester and AutoTrader consumers (future work)
- **Learning curve**: New pattern for contributors to understand

## Implementation Files

| File | Purpose |
|------|---------|
| `core/strategy.py` | ABC + implementations + factory |
| `tests/test_strategy.py` | Strategy unit tests |

## Signal Combination Logic

The `_combine_signals()` method consolidates the decision logic:

```python
def _combine_signals(self, predicted_return, agent_action, confidence, open_status):
    # Check confidence threshold
    if confidence < self.config.min_confidence:
        return SignalType.HOLD

    # Check if new positions allowed
    if not open_status:
        if agent_action == Action.CLOSE:
            return SignalType.CLOSE
        return SignalType.HOLD

    # Close action always passes through
    if agent_action == Action.CLOSE:
        return SignalType.CLOSE

    # BUY signal: check agreement/contradiction
    if agent_action == Action.BUY:
        if predicted_return >= threshold:
            return SignalType.BUY  # Agreement
        elif predicted_return < -threshold:
            return SignalType.HOLD  # Strong contradiction
        else:
            return SignalType.BUY  # Weak prediction, trust agent

    # SELL signal: similar logic
    ...
```

## Migration Path

1. ✅ Create `core/strategy.py` with ABC and implementations
2. ✅ Create unit tests
3. ⏳ Update `Backtester` to accept `TradingStrategy` (future)
4. ⏳ Update `AutoTrader` to use injected strategy (future)
5. ⏳ Deprecate callable-only strategy interface (future)

## Alternatives Considered

### 1. Keep inline functions
- Continues duplication
- No formal interface
- Hard to test

### 2. Simple callable interface with type hints
- Still no structure
- No lifecycle hooks
- No metadata

### 3. Event-driven architecture
- Overkill for current needs
- More complex implementation
- Harder to reason about
