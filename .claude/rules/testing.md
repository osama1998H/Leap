---
paths: tests/**/*.py
---
# Testing Rules

These rules apply when working on test files in the `tests/` directory.

---

## Test File Organization

Test files mirror the source structure:

| Source | Test File |
|--------|-----------|
| `core/trading_env.py` | `tests/test_trading_env.py` |
| `core/risk_manager.py` | `tests/test_risk_manager.py` |
| `core/broker_interface.py` | `tests/test_broker_interface.py` |
| `core/strategy.py` | `tests/test_strategy.py` |
| `cli/commands/*.py` | `tests/test_cli.py` |
| End-to-end pipeline | `tests/test_integration.py` |

---

## Running Tests

### All Tests
```bash
python -m pytest tests/ -v
```

### Specific Test File
```bash
python -m pytest tests/test_trading_env.py -v
```

### Specific Test Function
```bash
python -m pytest tests/test_trading_env.py::test_step_buy -v
```

### Pattern Matching
```bash
python -m pytest tests/ -v -k "risk"  # All tests with "risk" in name
```

### With Coverage
```bash
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## Required Patterns

### Mock External Dependencies

Always mock external services (MT5, data APIs):

```python
from unittest.mock import Mock, patch

@patch('core.mt5_broker.mt5')
def test_broker_connection(mock_mt5):
    mock_mt5.initialize.return_value = True
    mock_mt5.terminal_info.return_value = Mock(connected=True)

    broker = MT5BrokerGateway()
    assert broker.is_connected()
```

### Fixtures for Test Data

Use pytest fixtures for reusable test data:

```python
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    return pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100, 1000, 100),
    })

@pytest.fixture
def trading_env(sample_market_data):
    """Create a trading environment for testing."""
    config = EnvConfig(initial_balance=10000.0)
    return TradingEnvironment(data=sample_market_data, config=config)
```

### Integration Tests

For end-to-end testing:

```python
def test_full_training_pipeline(tmp_path):
    """Test complete training pipeline."""
    # Setup
    config = SystemConfig(...)
    system = LeapTradingSystem(config)

    # Execute
    result = system.train(epochs=1, timesteps=100)

    # Verify
    assert result.success
    assert (tmp_path / 'model.pt').exists()
```

---

## Test Quality Guidelines

### Test Meaningful Behavior

```python
# Good - tests actual behavior
def test_risk_manager_prevents_overleveraging():
    rm = RiskManager(max_leverage=10)
    size = rm.calculate_position_size(
        balance=1000,
        entry_price=100,
        stop_loss_price=95
    )
    assert size * 100 <= 1000 * 10  # Within leverage limit

# Bad - just tests existence
def test_risk_manager_exists():
    rm = RiskManager()
    assert rm is not None
```

### Cover Edge Cases

```python
def test_position_sizing_with_zero_balance():
    rm = RiskManager()
    size = rm.calculate_position_size(balance=0, ...)
    assert size == 0

def test_position_sizing_with_tiny_stop_loss():
    rm = RiskManager()
    size = rm.calculate_position_size(
        stop_loss_price=entry_price - 0.0001  # Very tight stop
    )
    assert size > 0  # Should still calculate something
```

### Test Error Conditions

```python
def test_order_rejected_insufficient_funds():
    broker = PaperBrokerGateway(initial_balance=100)

    with pytest.raises(InsufficientFundsError):
        broker.place_order(Order(size=1000000))  # Way too big
```

---

## Patching Guidelines

### Patch at Import Location

```python
# Correct - patch where it's imported
@patch('main.TransformerPredictor')
def test_train(mock_predictor):
    ...

# Also correct for internal modules
@patch('core.auto_trader.OrderManager')
def test_auto_trader(mock_order_manager):
    ...
```

### Use Context Managers for Complex Mocks

```python
def test_with_multiple_mocks():
    with patch('core.broker_interface.mt5') as mock_mt5, \
         patch('core.data_pipeline.requests') as mock_requests:
        mock_mt5.initialize.return_value = True
        mock_requests.get.return_value.json.return_value = {...}

        # Test code
```

---

## Test Categories

### Unit Tests
- Test single function/method in isolation
- Mock all dependencies
- Fast execution

### Integration Tests
- Test multiple components together
- May use real file system (in tmp_path)
- Medium execution time

### End-to-End Tests
- Test complete workflows
- Minimal mocking
- Slower execution, run less frequently

```python
# Mark slow tests
@pytest.mark.slow
def test_full_backtest():
    ...

# Run excluding slow tests
# python -m pytest tests/ -v -m "not slow"
```

---

## Common Fixtures

Reuse these common fixture patterns:

```python
@pytest.fixture
def env_config():
    return EnvConfig(initial_balance=10000.0, leverage=100)

@pytest.fixture
def paper_broker():
    return PaperBrokerGateway(initial_balance=10000.0)

@pytest.fixture
def mock_predictor():
    predictor = Mock(spec=TransformerPredictor)
    predictor.predict.return_value = (0.5, 0.1)  # prediction, uncertainty
    return predictor

@pytest.fixture
def tmp_model_dir(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
```

---

## DO NOT

- Skip mocking external dependencies
- Write tests that only check existence
- Use real network calls in tests
- Create tests that depend on execution order
- Leave debug print statements in tests
