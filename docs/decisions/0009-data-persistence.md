# ADR 0009: Pipeline Data Persistence with --save-data Flag

## Status
Accepted

## Context

Users need to:
1. **Debug model behavior** by examining the exact data used for training/backtesting
2. **Reproduce results** by reusing the same data
3. **Analyze feature distributions** offline without rerunning the pipeline
4. **Share datasets** with collaborators
5. **Archive data** for regulatory/audit purposes

Currently, pipeline data (raw OHLCV and computed features) is only held in memory during execution and discarded after commands complete. This makes it difficult to:
- Verify what data was used for a specific training run
- Compare data across different runs
- Debug feature engineering issues

## Decision

Add a `--save-data` CLI flag that persists pipeline data to CSV files organized by run ID.

### Directory Structure

```
data/
  {run_id}/
    raw.csv         # Raw OHLCV data (timestamp, open, high, low, close, volume)
    features.csv    # Computed features (~100 indicators)
    metadata.json   # Data lineage and reproducibility info
    additional/     # Multi-timeframe raw data (optional)
      15m_raw.csv
      4h_raw.csv
```

### Run ID Format

Run IDs follow the existing pattern: `{command}-{symbol}-{timeframe}-{timestamp}`

Examples:
- `train-EURUSD-1h-20241217_143052`
- `backtest-GBPUSD-4h-20241217_150000`

### Implementation

1. **New utility module**: `utils/data_saver.py` with `save_pipeline_data()` and `generate_run_id()` functions
2. **CLI flag**: `--save-data` added to parser, available for all commands
3. **CSV format**: UTF-8, headers, 8 decimal precision, ISO timestamps
4. **Metadata JSON**: Includes symbol, timeframe, feature names, data source, date range

### Usage

```bash
# Save data during training
python main.py train --symbol EURUSD --save-data

# Save data during backtest
python main.py backtest --symbol EURUSD --save-data

# Data saved to: data/train-EURUSD-1h-20241217_143052/
```

### Supported Commands

- `train` - Saves data after loading, before training
- `backtest` - Saves data after loading, before backtesting
- `walkforward` - Saves data after loading, before analysis
- `evaluate` - Saves data after loading, before evaluation
- `autotrade` - Saves initial data snapshot before trading loop

## Consequences

### Positive
- Full data lineage for debugging and reproducibility
- Offline analysis without rerunning pipelines
- Data sharing and archiving capabilities
- Follows centralized utilities pattern (ADR-0001)
- Correlates with MLflow runs via matching run ID format

### Negative
- Increased disk usage (~10-50MB per run for 50k bars with ~100 features)
- Slight performance overhead when flag is enabled
- Users must manage data directory growth

### Mitigations
- Data saving is opt-in (flag disabled by default)
- Consider future cleanup utility for old runs
- Metadata includes creation timestamp for lifecycle management

## Code References
- See: `utils/data_saver.py`
- See: `cli/parser.py` (--save-data flag)
- See: `cli/commands/*.py` (integration points)
