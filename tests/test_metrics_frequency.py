import numpy as np
import pandas as pd
import pytest

import importlib.util


def _load_metrics_calculator():
    spec = importlib.util.spec_from_file_location("evaluation.metrics", "evaluation/metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.MetricsCalculator


def _build_equity_curve(periods: int, per_period_return: float):
    # Start at 1.0 and apply constant per-period returns
    returns = np.full(periods, per_period_return)
    equity = 1.0 * np.cumprod(1 + returns)
    return equity


def test_metrics_infer_hourly_frequency():
    MetricsCalculator = _load_metrics_calculator()
    periods = 24 * 30  # 30 days of hourly data
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    per_period_return = 0.001
    equity = _build_equity_curve(periods, per_period_return)

    calc = MetricsCalculator()
    metrics = calc.calculate_all(equity_curve=equity, timestamps=timestamps)

    inferred_ppy = MetricsCalculator.infer_periods_per_year(timestamps)
    expected_total_return = (1 + per_period_return) ** (periods - 1) - 1
    expected_annualized = (1 + expected_total_return) ** (inferred_ppy / periods) - 1

    assert metrics["annualized_return"] == pytest.approx(expected_annualized, rel=1e-6)
    assert inferred_ppy == pytest.approx(252 * 24)


def test_metrics_infer_daily_frequency():
    MetricsCalculator = _load_metrics_calculator()
    periods = 200  # 200 daily observations
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="D")
    per_period_return = 0.002
    equity = _build_equity_curve(periods, per_period_return)

    calc = MetricsCalculator()
    metrics = calc.calculate_all(equity_curve=equity, timestamps=timestamps)

    inferred_ppy = MetricsCalculator.infer_periods_per_year(timestamps)
    expected_total_return = (1 + per_period_return) ** (periods - 1) - 1
    expected_annualized = (1 + expected_total_return) ** (inferred_ppy / periods) - 1

    assert metrics["annualized_return"] == pytest.approx(expected_annualized, rel=1e-6)
    assert inferred_ppy == pytest.approx(252)
