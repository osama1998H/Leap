# CSV Data Template

This directory contains template files for importing custom OHLCV data into Leap.

## Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Bar timestamp (e.g., "2024-01-01 00:00:00") |
| `open` | float | Opening price |
| `high` | float | Highest price in the period |
| `low` | float | Lowest price in the period |
| `close` | float | Closing price |
| `volume` | float | Trading volume |

## Supported Column Name Variations

The data pipeline automatically recognizes these column name variations:

| Standard | Alternatives |
|----------|--------------|
| timestamp | date, datetime, time |
| open | o |
| high | h |
| low | l |
| close | c |
| volume | v, vol, tick_volume |

## File Naming Convention

When using a directory of CSV files, name your files using one of these patterns:
- `{SYMBOL}_{TIMEFRAME}.csv` (e.g., `EURUSD_1h.csv`)
- `{symbol}_{timeframe}.csv` (e.g., `eurusd_1h.csv`)
- `{SYMBOL}.csv` (e.g., `EURUSD.csv`)
- `{symbol}.csv` (e.g., `eurusd.csv`)

## Usage Examples

### 1. Single CSV File

```python
from core.data_pipeline import DataPipeline

pipeline = DataPipeline()
data = pipeline.fetch_historical_data(
    symbol="EURUSD",
    timeframe="1h",
    n_bars=10000,
    csv_path="/path/to/your/data.csv"
)
```

### 2. Directory of CSV Files

```python
from core.data_pipeline import DataPipeline

# Config with CSV directory
config = {
    'data_source': 'csv',
    'csv_path': '/path/to/csv/directory/'
}

pipeline = DataPipeline(config)
data = pipeline.fetch_historical_data(
    symbol="EURUSD",  # Will look for EURUSD_1h.csv
    timeframe="1h",
    n_bars=10000
)
```

### 3. Using Configuration File

```json
{
  "data": {
    "data_source": "csv",
    "csv_path": "./data/my_data.csv",
    "csv_datetime_column": "timestamp",
    "csv_datetime_format": "%Y-%m-%d %H:%M:%S"
  }
}
```

Then run:
```bash
python main.py train --config config/my_config.json --symbol EURUSD
```

## Data Requirements

1. **Chronological Order**: Data should be sorted by timestamp (oldest first)
2. **OHLCV Validity**: High >= max(Open, Close) and Low <= min(Open, Close)
3. **No Missing Rows**: Ensure continuous data without gaps
4. **Numeric Values**: All OHLCV values must be valid numbers
5. **Minimum Bars**: At least `lookback_window + prediction_horizon` bars (default: 132)

## Sample Data

See `ohlcv_template.csv` for a sample file with the correct format.
