# ADR 0013: AdaptiveTrainer CLI Integration

## Status
Accepted

## Context

The `AdaptiveTrainer` class in `training/online_learning.py` provides sophisticated online learning capabilities:

- Continuous model adaptation based on market conditions
- Regime detection (trending, ranging, volatile)
- Performance monitoring with automatic adaptation triggers
- Both offline and online training modes

However, there was no way to invoke this functionality from the command line:

1. **No CLI access**: Users must write Python code to use adaptive training
2. **No parameter exposure**: Adaptation thresholds not configurable without code
3. **Discovery gap**: Users may not know the capability exists
4. **Inconsistent UX**: Other features (train, backtest) accessible via CLI

## Decision

Add an `adapt` CLI command to expose AdaptiveTrainer functionality:

### CLI Arguments

```bash
python main.py adapt [options]

Options:
  --mode {offline,online,evaluate}  Adaptation mode (default: offline)
  --symbol SYMBOL                   Trading symbol
  --adapt-bars N                    Recent bars to use (default: 10000)
  --adapt-epochs N                  Predictor training epochs (default: 10)
  --adapt-timesteps N               Agent training timesteps (default: 10000)
  --error-threshold F               Prediction error threshold (default: 0.05)
  --drawdown-threshold F            Drawdown threshold (default: 0.1)
  --adapt-frequency N               Steps between adaptation checks (default: 100)
  --max-adaptations N               Max adaptations per day (default: 10)
  --min-samples N                   Min samples before adaptation (default: 50)
  --model-dir PATH                  Model directory (default: ./saved_models)
  --save                            Save adapted models
```

### Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `offline` | Single adaptation pass with recent data | Quick model update |
| `online` | Continuous adaptation loop | Experimental real-time adaptation |
| `evaluate` | Analyze adaptation performance | Check model drift |

### Implementation Structure

```
cli/
  commands/
    __init__.py       # Register 'adapt': execute_adapt
    adapt.py          # execute_adapt() handler
  parser.py           # Add adapt arguments
```

### Command Handler

```python
def execute_adapt(system, args, config, resolved):
    """Execute the adapt command."""
    # 1. Load existing models
    # 2. Load recent market data
    # 3. Create AdaptationConfig from args
    # 4. Execute based on mode:
    #    - offline: _run_offline_adaptation()
    #    - online: _run_online_adaptation()
    #    - evaluate: _evaluate_adaptation()
```

### Offline Mode Workflow

1. Load predictor and agent from `--model-dir`
2. Fetch recent bars with `--adapt-bars`
3. Prepare training sequences (80/20 train/val split)
4. Create `TradingEnvironment` for agent training
5. Run `AdaptiveTrainer.train_offline()`
6. Print results summary
7. Optionally save adapted models with `--save`

### Online Mode Workflow (Experimental)

1. Load models and create environment
2. Create data stream wrapper
3. Start continuous training loop
4. Monitor for keyboard interrupt (Ctrl+C)
5. Print performance report on exit

### Evaluate Mode Workflow

1. Load models
2. Run prediction on recent data
3. Calculate metrics: MAE, RMSE, correlation, direction accuracy
4. Analyze agent action distribution
5. Print evaluation report

## Consequences

### Positive
- **Discoverability**: Users find adaptation via `python main.py --help`
- **Accessibility**: No Python code needed for basic adaptation
- **Consistency**: Same UX as other commands (train, backtest)
- **Flexibility**: All thresholds configurable via CLI
- **Integration**: Works with existing model directory structure

### Negative
- **Complexity**: Online mode is experimental and needs careful monitoring
- **More arguments**: CLI has many optional parameters
- **Dependencies**: Requires trained models to exist

## Implementation Files

| File | Purpose |
|------|---------|
| `cli/commands/adapt.py` | Command handler with mode implementations |
| `cli/commands/__init__.py` | Register adapt in COMMANDS dict |
| `cli/parser.py` | Add adapt-specific arguments |
| `tests/test_adapt_command.py` | Unit and integration tests |

## Usage Examples

### Quick Offline Adaptation

```bash
# Adapt models to recent EURUSD data
python main.py adapt --symbol EURUSD --adapt-bars 5000 --adapt-epochs 5

# With custom thresholds
python main.py adapt --symbol EURUSD --error-threshold 0.03 --save
```

### Evaluate Model Performance

```bash
# Check prediction accuracy and agent behavior
python main.py adapt --mode evaluate --symbol EURUSD --adapt-bars 2000
```

### Experimental Online Adaptation

```bash
# Run continuous adaptation (press Ctrl+C to stop)
python main.py adapt --mode online --symbol EURUSD --adapt-frequency 50
```

### Multi-Symbol Adaptation

```bash
# Adapt models for multiple symbols
python main.py adapt --symbols EURUSD GBPUSD --adapt-epochs 10 --save
```

## Migration Path

1. ✅ Create `cli/commands/adapt.py` with all mode implementations
2. ✅ Add to `cli/commands/__init__.py` COMMANDS dict
3. ✅ Add arguments to `cli/parser.py`
4. ✅ Create comprehensive tests
5. ⏳ Add to CLI.md documentation (future)

## Alternatives Considered

### 1. Jupyter Notebook
- Pro: Interactive, visual
- Con: Not scriptable, different UX from CLI

### 2. Configuration file only
- Pro: Complex configs easier to manage
- Con: Still needs some CLI; doesn't solve discovery

### 3. Integrated into train command
- Pro: Fewer commands
- Con: Conflates initial training with adaptation; confusing arguments

## Safety Considerations

### Online Mode Warning

The online mode includes explicit warnings:

```python
logger.warning("EXPERIMENTAL: Online adaptation mode")
logger.warning("This will continuously adapt models based on market data.")
logger.warning("Press Ctrl+C to stop.")
```

### Model Saving

- Adapted models saved to timestamped subdirectory: `adapted_{timestamp}/`
- Original models preserved
- `--save` flag required (not automatic)

### Graceful Shutdown

Online mode handles `KeyboardInterrupt` to cleanly stop training and print final report.
