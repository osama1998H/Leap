"""
Tests for utils/data_saver.py

Tests the pipeline data persistence functionality including CSV export
of raw OHLCV data and computed features.
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from utils.data_saver import generate_run_id, save_pipeline_data


class MockMarketData:
    """Mock MarketData for testing."""

    def __init__(self, n_bars: int = 100, n_features: int = 5):
        self.symbol = "EURUSD"
        self.timeframe = "1h"
        self.timestamp = pd.date_range("2024-01-01", periods=n_bars, freq="1h").to_numpy()
        self.open = np.random.uniform(1.1, 1.2, n_bars)
        self.high = np.random.uniform(1.1, 1.2, n_bars)
        self.low = np.random.uniform(1.1, 1.2, n_bars)
        self.close = np.random.uniform(1.1, 1.2, n_bars)
        self.volume = np.random.uniform(1000, 10000, n_bars)
        self.features = np.random.randn(n_bars, n_features)
        self.feature_names = [f"feat_{i}" for i in range(n_features)]


class TestGenerateRunId:
    """Tests for generate_run_id function."""

    def test_basic_format(self):
        """Test that run ID follows expected format."""
        run_id = generate_run_id("train", "EURUSD", "1h")
        parts = run_id.split("-")

        assert len(parts) == 4
        assert parts[0] == "train"
        assert parts[1] == "EURUSD"
        assert parts[2] == "1h"
        # Fourth part is timestamp in format YYYYMMDD_HHMMSS
        assert len(parts[3]) == 15  # 8 + 1 + 6

    def test_different_commands(self):
        """Test run ID generation for different commands."""
        for cmd in ["train", "backtest", "walkforward", "evaluate", "autotrade"]:
            run_id = generate_run_id(cmd, "GBPUSD", "4h")
            assert run_id.startswith(f"{cmd}-GBPUSD-4h-")

    def test_unique_ids(self):
        """Test that consecutive calls produce unique IDs (due to timestamp)."""
        import time

        id1 = generate_run_id("train", "EURUSD", "1h")
        time.sleep(1.1)  # Ensure different second
        id2 = generate_run_id("train", "EURUSD", "1h")

        assert id1 != id2


class TestSavePipelineData:
    """Tests for save_pipeline_data function."""

    def test_creates_directory_structure(self):
        """Test that the function creates the expected directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData()
            run_dir = save_pipeline_data(
                run_id="test-run",
                market_data=market_data,
                base_dir=tmpdir
            )

            assert os.path.exists(run_dir)
            assert os.path.exists(os.path.join(run_dir, "raw.csv"))
            assert os.path.exists(os.path.join(run_dir, "features.csv"))
            assert os.path.exists(os.path.join(run_dir, "metadata.json"))

    def test_raw_csv_columns(self):
        """Test that raw.csv has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData(n_bars=50)
            run_dir = save_pipeline_data(
                run_id="test-run",
                market_data=market_data,
                base_dir=tmpdir
            )

            df = pd.read_csv(os.path.join(run_dir, "raw.csv"))
            expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            assert list(df.columns) == expected_columns
            assert len(df) == 50

    def test_features_csv_columns(self):
        """Test that features.csv has correct columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData(n_bars=50, n_features=10)
            run_dir = save_pipeline_data(
                run_id="test-run",
                market_data=market_data,
                base_dir=tmpdir
            )

            df = pd.read_csv(os.path.join(run_dir, "features.csv"))
            # First column is timestamp, then feature columns
            assert df.columns[0] == "timestamp"
            assert len(df.columns) == 11  # timestamp + 10 features
            assert len(df) == 50

    def test_metadata_contents(self):
        """Test that metadata.json has correct contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData(n_bars=100, n_features=5)
            run_dir = save_pipeline_data(
                run_id="test-run",
                market_data=market_data,
                base_dir=tmpdir,
                command="train",
                n_bars=100,
                data_source="MT5"
            )

            with open(os.path.join(run_dir, "metadata.json")) as f:
                metadata = json.load(f)

            assert metadata["run_id"] == "test-run"
            assert metadata["command"] == "train"
            assert metadata["symbol"] == "EURUSD"
            assert metadata["timeframe"] == "1h"
            assert metadata["n_bars"] == 100
            assert metadata["actual_bars"] == 100
            assert metadata["feature_count"] == 5
            assert metadata["data_source"] == "MT5"
            assert len(metadata["feature_names"]) == 5
            assert metadata["raw_columns"] == ["timestamp", "open", "high", "low", "close", "volume"]
            assert "date_range" in metadata
            assert "start" in metadata["date_range"]
            assert "end" in metadata["date_range"]

    def test_returns_run_directory_path(self):
        """Test that function returns the correct path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData()
            run_dir = save_pipeline_data(
                run_id="my-run-id",
                market_data=market_data,
                base_dir=tmpdir
            )

            expected_path = os.path.join(tmpdir, "my-run-id")
            assert run_dir == expected_path

    def test_handles_no_features(self):
        """Test handling of market data without features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData()
            market_data.features = None
            market_data.feature_names = None

            run_dir = save_pipeline_data(
                run_id="test-run",
                market_data=market_data,
                base_dir=tmpdir
            )

            # raw.csv and metadata.json should still exist
            assert os.path.exists(os.path.join(run_dir, "raw.csv"))
            assert os.path.exists(os.path.join(run_dir, "metadata.json"))
            # features.csv should not be created
            assert not os.path.exists(os.path.join(run_dir, "features.csv"))

    def test_float_precision(self):
        """Test that floats are saved with 8 decimal places."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData(n_bars=10)
            # Set a specific value to verify precision
            market_data.close[0] = 1.12345678912345

            run_dir = save_pipeline_data(
                run_id="test-run",
                market_data=market_data,
                base_dir=tmpdir
            )

            df = pd.read_csv(os.path.join(run_dir, "raw.csv"))
            # Value should be rounded to 8 decimal places
            assert abs(df["close"].iloc[0] - 1.12345679) < 1e-8

    def test_nested_run_directory(self):
        """Test that nested base directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            market_data = MockMarketData()
            nested_base = os.path.join(tmpdir, "nested", "data", "dir")

            run_dir = save_pipeline_data(
                run_id="test-run",
                market_data=market_data,
                base_dir=nested_base
            )

            assert os.path.exists(run_dir)
            assert os.path.exists(os.path.join(run_dir, "raw.csv"))


class TestDataSaverIntegration:
    """Integration tests for data saver with realistic data."""

    def test_large_dataset(self):
        """Test with a larger, more realistic dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate ~100 features and 10000 bars
            market_data = MockMarketData(n_bars=10000, n_features=100)

            run_dir = save_pipeline_data(
                run_id="large-test",
                market_data=market_data,
                base_dir=tmpdir,
                command="train",
                n_bars=10000,
                data_source="synthetic"
            )

            # Verify files exist and have correct sizes
            raw_df = pd.read_csv(os.path.join(run_dir, "raw.csv"))
            features_df = pd.read_csv(os.path.join(run_dir, "features.csv"))

            assert len(raw_df) == 10000
            assert len(features_df) == 10000
            assert len(features_df.columns) == 101  # timestamp + 100 features
