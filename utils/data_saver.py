"""
Leap Trading System - Data Saver Utilities

Save pipeline data (raw OHLCV and computed features) to CSV files
for debugging, reproducibility, and offline analysis.

Usage:
    from utils.data_saver import save_pipeline_data, generate_run_id

    run_id = generate_run_id("train", "EURUSD", "1h")
    save_pipeline_data(run_id, market_data, base_dir="data", command="train")
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['save_pipeline_data', 'generate_run_id']


def save_pipeline_data(
    run_id: str,
    market_data: Any,
    base_dir: str = "data",
    command: str = "unknown",
    n_bars: int = 0,
    data_source: str = "unknown",
    additional_timeframes_data: Optional[Dict[str, pd.DataFrame]] = None
) -> str:
    """
    Save raw OHLCV and computed features to CSV files.

    Creates directory structure:
        {base_dir}/{run_id}/
            raw.csv         - Raw OHLCV data
            features.csv    - Computed features
            metadata.json   - Data lineage info
            additional/     - Multi-timeframe raw data (if any)

    Args:
        run_id: Unique identifier for this run (e.g., "train-EURUSD-1h-20241217_143052")
        market_data: MarketData object with OHLCV and features
        base_dir: Base directory for data storage (default: "data")
        command: CLI command that generated this data
        n_bars: Number of bars requested
        data_source: Data source identifier ("MT5" or "synthetic")
        additional_timeframes_data: Dict mapping timeframe to raw DataFrame

    Returns:
        Path to the created run directory

    Examples:
        >>> from core.data_pipeline import MarketData
        >>> run_id = generate_run_id("train", "EURUSD", "1h")
        >>> run_dir = save_pipeline_data(
        ...     run_id=run_id,
        ...     market_data=market_data,
        ...     command="train",
        ...     n_bars=50000
        ... )
        >>> print(f"Data saved to {run_dir}")
    """
    # Create run directory
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Save raw OHLCV data
    raw_path = os.path.join(run_dir, "raw.csv")
    _save_raw_ohlcv(market_data, raw_path)
    logger.info(f"Saved raw OHLCV data to {raw_path}")

    # Save computed features
    if market_data.features is not None and market_data.feature_names:
        features_path = os.path.join(run_dir, "features.csv")
        _save_features(market_data, features_path)
        logger.info(f"Saved {len(market_data.feature_names)} features to {features_path}")

    # Save additional timeframe data if present
    additional_timeframes = []
    if additional_timeframes_data:
        additional_dir = os.path.join(run_dir, "additional")
        os.makedirs(additional_dir, exist_ok=True)
        for tf_name, tf_df in additional_timeframes_data.items():
            tf_path = os.path.join(additional_dir, f"{tf_name}_raw.csv")
            _save_dataframe(tf_df, tf_path)
            additional_timeframes.append(tf_name)
            logger.info(f"Saved {tf_name} timeframe data to {tf_path}")

    # Save metadata
    metadata_path = os.path.join(run_dir, "metadata.json")
    _save_metadata(
        metadata_path,
        run_id=run_id,
        market_data=market_data,
        command=command,
        n_bars=n_bars,
        data_source=data_source,
        additional_timeframes=additional_timeframes
    )
    logger.info(f"Saved metadata to {metadata_path}")

    return run_dir


def generate_run_id(command: str, symbol: str, timeframe: str) -> str:
    """
    Generate a unique run ID for data saving.

    Args:
        command: CLI command (train, backtest, etc.)
        symbol: Trading symbol
        timeframe: Primary timeframe

    Returns:
        Unique run ID string

    Examples:
        >>> generate_run_id("train", "EURUSD", "1h")
        'train-EURUSD-1h-20241217_143052'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{command}-{symbol}-{timeframe}-{timestamp}"


def _save_raw_ohlcv(market_data: Any, path: str) -> None:
    """Save raw OHLCV data to CSV."""
    df = pd.DataFrame({
        'timestamp': market_data.timestamp,
        'open': market_data.open,
        'high': market_data.high,
        'low': market_data.low,
        'close': market_data.close,
        'volume': market_data.volume
    })
    _save_dataframe(df, path)


def _save_features(market_data: Any, path: str) -> None:
    """Save computed features to CSV."""
    df = pd.DataFrame(
        market_data.features,
        columns=market_data.feature_names
    )
    # Add timestamp as first column for time-series alignment
    df.insert(0, 'timestamp', market_data.timestamp)
    _save_dataframe(df, path)


def _save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV with consistent formatting."""
    df.to_csv(
        path,
        index=False,
        float_format='%.8f',
        date_format='%Y-%m-%d %H:%M:%S'
    )


def _save_metadata(
    path: str,
    run_id: str,
    market_data: Any,
    command: str,
    n_bars: int,
    data_source: str,
    additional_timeframes: List[str]
) -> None:
    """Save metadata JSON file."""
    metadata = {
        'run_id': run_id,
        'command': command,
        'symbol': market_data.symbol,
        'timeframe': market_data.timeframe,
        'additional_timeframes': additional_timeframes,
        'n_bars': n_bars,
        'actual_bars': len(market_data.close),
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'feature_count': len(market_data.feature_names) if market_data.feature_names else 0,
        'feature_names': list(market_data.feature_names) if market_data.feature_names else [],
        'raw_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        'data_source': data_source,
        'date_range': {
            'start': _format_timestamp(market_data.timestamp[0]) if len(market_data.timestamp) > 0 else None,
            'end': _format_timestamp(market_data.timestamp[-1]) if len(market_data.timestamp) > 0 else None
        }
    }

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def _format_timestamp(ts: Any) -> str:
    """Format timestamp to ISO string."""
    if isinstance(ts, (pd.Timestamp, datetime)):
        return ts.isoformat()
    elif isinstance(ts, np.datetime64):
        return pd.Timestamp(ts).isoformat()
    return str(ts)
