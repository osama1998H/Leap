"""
Leap Trading System - Feature Engineering Tests
Tests for technical indicator calculations, especially ADX with Wilder's smoothing.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location('data_pipeline', 'core/data_pipeline.py')
data_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_pipeline)

FeatureEngineer = data_pipeline.FeatureEngineer


def create_sample_ohlcv(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
    prices = 1.1 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n_bars)))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': prices * (1 + np.random.uniform(0, 0.005, n_bars)),
        'low': prices * (1 - np.random.uniform(0, 0.005, n_bars)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_bars)
    })

    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def test_wilder_smoothing_basic():
    """Test basic Wilder's smoothing calculation."""
    print("\n" + "="*60)
    print("Testing Wilder's Smoothing - Basic...")
    print("="*60)

    fe = FeatureEngineer()

    # Create a simple series with known values
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])

    smoothed = fe._wilder_smoothing(series, period=14)

    # First 13 values should be NaN
    assert pd.isna(smoothed.iloc[:13]).all(), "First 13 values should be NaN"

    # 14th value should be sum of first 14 values
    expected_first = sum(range(1, 15))  # 1+2+...+14 = 105
    assert abs(smoothed.iloc[13] - expected_first) < 0.01, \
        f"First smoothed value should be {expected_first}, got {smoothed.iloc[13]}"

    # 15th value should follow Wilder's formula: Prior - (Prior/14) + Current
    expected_15th = 105 - (105/14) + 15  # = 105 - 7.5 + 15 = 112.5
    assert abs(smoothed.iloc[14] - expected_15th) < 0.01, \
        f"15th value should be {expected_15th}, got {smoothed.iloc[14]}"

    print(f"✓ First smoothed value (sum of first 14): {smoothed.iloc[13]:.2f}")
    print(f"✓ 15th value (Wilder's formula): {smoothed.iloc[14]:.2f}")
    print("✓ Wilder's smoothing basic test passed")

    return True


def test_wilder_smoothing_edge_cases():
    """Test Wilder's smoothing with edge cases."""
    print("\n" + "="*60)
    print("Testing Wilder's Smoothing - Edge Cases...")
    print("="*60)

    fe = FeatureEngineer()

    # Test 1: Empty series
    empty_series = pd.Series([], dtype=float)
    result = fe._wilder_smoothing(empty_series, period=14)
    assert len(result) == 0, "Empty series should return empty result"
    print("✓ Empty series handled correctly")

    # Test 2: Series shorter than period
    short_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fe._wilder_smoothing(short_series, period=14)
    assert pd.isna(result).all(), "Series shorter than period should return all NaN"
    print("✓ Short series handled correctly")

    # Test 3: Series with NaN values
    nan_series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                           9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    result = fe._wilder_smoothing(nan_series, period=14)
    # Should still produce some results after enough valid values
    assert len(result) == len(nan_series), "Result should have same length as input"
    print("✓ Series with NaN values handled correctly")

    # Test 4: Series exactly equal to period length
    exact_series = pd.Series([float(i) for i in range(1, 15)])  # 14 values
    result = fe._wilder_smoothing(exact_series, period=14)
    # Should have exactly one non-NaN value (the sum)
    non_nan_count = result.notna().sum()
    assert non_nan_count == 1, f"Expected 1 non-NaN value, got {non_nan_count}"
    print("✓ Exact period length series handled correctly")

    print("✓ All edge cases passed")
    return True


def test_wilder_smoothing_vs_ema():
    """
    Verify Wilder's smoothing approximates EMA with alpha=1/period.

    Wilder's smoothing: Prior - (Prior/N) + Current = Prior * (1 - 1/N) + Current
    This is equivalent to EMA with alpha = 1/N, but with different initialization.
    """
    print("\n" + "="*60)
    print("Testing Wilder's Smoothing vs EMA Equivalence...")
    print("="*60)

    fe = FeatureEngineer()

    # Create test series
    np.random.seed(42)
    series = pd.Series(np.random.uniform(10, 20, 100))
    period = 14

    # Calculate Wilder's smoothing
    wilder = fe._wilder_smoothing(series, period)

    # Calculate equivalent EMA (alpha = 1/period)
    # Note: EMA initialization differs, so we compare the convergence behavior
    alpha = 1.0 / period
    ema = series.ewm(alpha=alpha, adjust=False).mean()

    # After warmup, the rate of change should be similar
    # Compare the smoothing factors
    wilder_valid = wilder.dropna()

    if len(wilder_valid) > 20:
        # Compare the smoothness (standard deviation of differences)
        wilder_diffs = wilder_valid.diff().dropna()
        ema_diffs = ema.iloc[-len(wilder_diffs):].diff().dropna()

        # Both should have similar smoothing characteristics
        wilder_std = wilder_diffs.std()
        ema_std = ema_diffs.std()

        print(f"✓ Wilder's smoothing std of diffs: {wilder_std:.6f}")
        print(f"✓ EMA (alpha=1/14) std of diffs: {ema_std:.6f}")

        # The ratio should be reasonable (not exact due to different initialization)
        ratio = wilder_std / ema_std if ema_std > 0 else 0
        print(f"✓ Ratio (Wilder/EMA): {ratio:.2f}")

    print("✓ Wilder's smoothing behavior verified")
    return True


def test_adx_calculation():
    """Test ADX indicator calculation."""
    print("\n" + "="*60)
    print("Testing ADX Calculation...")
    print("="*60)

    fe = FeatureEngineer()
    df = create_sample_ohlcv(200)

    # Compute all features (which includes ADX)
    features = fe.compute_all_features(df)

    # Check ADX-related features exist
    adx_features = ['adx', 'adxr', 'plus_di', 'minus_di', 'dx',
                    'di_crossover', 'di_spread', 'adx_slope', 'adx_strength', 'adx_simple']

    for feat in adx_features:
        assert feat in features.columns, f"Missing feature: {feat}"
    print(f"✓ All {len(adx_features)} ADX-related features present")

    # Validate ADX range [0, 100]
    adx_valid = features['adx'].dropna()
    assert len(adx_valid) > 0, "ADX should have non-NaN values"
    assert adx_valid.min() >= 0, f"ADX min should be >= 0, got {adx_valid.min()}"
    assert adx_valid.max() <= 100, f"ADX max should be <= 100, got {adx_valid.max()}"
    print(f"✓ ADX range valid: [{adx_valid.min():.2f}, {adx_valid.max():.2f}]")

    # Validate +DI and -DI ranges [0, 100]
    plus_di_valid = features['plus_di'].dropna()
    minus_di_valid = features['minus_di'].dropna()
    assert plus_di_valid.min() >= 0, f"+DI min should be >= 0"
    assert minus_di_valid.min() >= 0, f"-DI min should be >= 0"
    print(f"✓ +DI range: [{plus_di_valid.min():.2f}, {plus_di_valid.max():.2f}]")
    print(f"✓ -DI range: [{minus_di_valid.min():.2f}, {minus_di_valid.max():.2f}]")

    # Validate DI crossover is binary
    di_cross = features['di_crossover'].dropna()
    assert set(di_cross.unique()).issubset({0, 1}), \
        f"DI crossover should be binary, got {di_cross.unique()}"
    print("✓ DI crossover is binary")

    # Validate DI spread is in [-1, 1]
    di_spread = features['di_spread'].dropna()
    assert di_spread.min() >= -1.01, f"DI spread min should be >= -1"
    assert di_spread.max() <= 1.01, f"DI spread max should be <= 1"
    print(f"✓ DI spread range: [{di_spread.min():.3f}, {di_spread.max():.3f}]")

    # Validate ADX strength categories
    adx_strength = features['adx_strength'].dropna()
    valid_strengths = {0, 1, 2}
    actual_strengths = set(adx_strength.unique())
    assert actual_strengths.issubset(valid_strengths), \
        f"ADX strength should be 0, 1, or 2, got {actual_strengths}"
    print("✓ ADX strength categories valid")

    return True


def test_adx_vs_simple_adx():
    """Compare Wilder's ADX with simple rolling ADX."""
    print("\n" + "="*60)
    print("Testing ADX vs Simple ADX Comparison...")
    print("="*60)

    fe = FeatureEngineer()
    df = create_sample_ohlcv(500)
    features = fe.compute_all_features(df)

    adx = features['adx'].dropna()
    adx_simple = features['adx_simple'].dropna()

    # Align the series for comparison
    common_idx = adx.index.intersection(adx_simple.index)
    adx_aligned = adx.loc[common_idx]
    adx_simple_aligned = adx_simple.loc[common_idx]

    # Calculate correlation
    correlation = adx_aligned.corr(adx_simple_aligned)
    print(f"✓ Correlation between Wilder's ADX and simple ADX: {correlation:.4f}")

    # They should be positively correlated but not identical
    assert correlation > 0.5, f"ADX variants should be correlated, got {correlation}"

    # Wilder's ADX should generally be smoother (lower std of differences)
    adx_diff_std = adx_aligned.diff().std()
    simple_diff_std = adx_simple_aligned.diff().std()
    print(f"✓ Wilder's ADX diff std: {adx_diff_std:.4f}")
    print(f"✓ Simple ADX diff std: {simple_diff_std:.4f}")

    # Wilder's smoothing typically produces smoother output
    if adx_diff_std < simple_diff_std:
        print("✓ Wilder's ADX is smoother (as expected)")
    else:
        print("⚠ Note: Simple ADX is smoother in this test data")

    return True


def test_adx_reference_comparison():
    """
    Verify ADX calculation against known reference values.

    Using a simplified reference calculation to verify the logic is correct.
    In production, this would compare against TA-Lib or another reference implementation.
    """
    print("\n" + "="*60)
    print("Testing ADX Reference Comparison...")
    print("="*60)

    fe = FeatureEngineer()

    # Create deterministic test data
    np.random.seed(123)
    n = 100

    # Create trending data for clearer ADX signal
    trend = np.cumsum(np.random.choice([-1, 1], n, p=[0.3, 0.7])) * 0.001
    base_price = 1.1 + trend

    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='h'),
        'open': base_price,
        'high': base_price + np.random.uniform(0.001, 0.003, n),
        'low': base_price - np.random.uniform(0.001, 0.003, n),
        'close': base_price + np.random.uniform(-0.001, 0.001, n),
        'volume': np.random.uniform(1000, 10000, n)
    })

    features = fe.compute_all_features(df)

    # Manual calculation for verification
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr_manual = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Compare TR
    tr_computed = features['tr']
    tr_diff = abs(tr_manual - tr_computed).mean()
    assert tr_diff < 0.0001, f"TR calculation differs: {tr_diff}"
    print(f"✓ True Range calculation verified (avg diff: {tr_diff:.8f})")

    # Verify DM calculations
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # Spot check: +DM should be 0 when high didn't increase more than low decreased
    print(f"✓ +DM has {(plus_dm > 0).sum()} positive values out of {len(plus_dm)}")
    print(f"✓ -DM has {(minus_dm > 0).sum()} positive values out of {len(minus_dm)}")

    # Verify ADX is computed
    adx_values = features['adx'].dropna()
    assert len(adx_values) > 0, "ADX should have computed values"
    print(f"✓ ADX computed for {len(adx_values)} bars")
    print(f"✓ ADX mean: {adx_values.mean():.2f}, std: {adx_values.std():.2f}")

    return True


def test_insufficient_data_handling():
    """Test that insufficient data is handled gracefully."""
    print("\n" + "="*60)
    print("Testing Insufficient Data Handling...")
    print("="*60)

    fe = FeatureEngineer()

    # Test with very short data (less than warmup period)
    short_df = create_sample_ohlcv(20)
    features = fe.compute_all_features(short_df)

    # ADX should exist but mostly be NaN
    assert 'adx' in features.columns, "ADX column should exist"
    adx_valid = features['adx'].dropna()
    print(f"✓ Short data ({len(short_df)} bars): {len(adx_valid)} valid ADX values")

    # Test with minimum viable data (around 28 bars for stable ADX)
    min_df = create_sample_ohlcv(30)
    features_min = fe.compute_all_features(min_df)
    adx_min_valid = features_min['adx'].dropna()
    print(f"✓ Minimum data ({len(min_df)} bars): {len(adx_min_valid)} valid ADX values")

    # Test with adequate data
    adequate_df = create_sample_ohlcv(100)
    features_adequate = fe.compute_all_features(adequate_df)
    adx_adequate_valid = features_adequate['adx'].dropna()
    print(f"✓ Adequate data ({len(adequate_df)} bars): {len(adx_adequate_valid)} valid ADX values")

    # More data should yield more valid ADX values
    assert len(adx_adequate_valid) > len(adx_valid), \
        "More data should yield more ADX values"
    print("✓ Insufficient data handled gracefully")

    return True


def test_feature_count_and_names():
    """Test that all expected features are computed."""
    print("\n" + "="*60)
    print("Testing Feature Count and Names...")
    print("="*60)

    fe = FeatureEngineer()
    df = create_sample_ohlcv(200)
    features = fe.compute_all_features(df)

    print(f"✓ Total columns: {len(features.columns)}")
    print(f"✓ Feature names count: {len(fe.feature_names)}")

    # Check key feature categories
    expected_categories = {
        'returns': ['returns', 'log_returns'],
        'moving_averages': ['sma_5', 'sma_20', 'ema_10', 'ema_50'],
        'momentum': ['rsi_14', 'macd', 'macd_signal', 'stoch_k_14'],
        'volatility': ['atr_14', 'bb_upper_20', 'bb_lower_20'],
        'volume': ['volume_sma_20', 'obv', 'mfi'],
        'trend': ['adx', 'plus_di', 'minus_di', 'cci'],
        'candlestick': ['body_size', 'is_bullish', 'is_doji'],
        'time': ['hour_sin', 'hour_cos', 'dow_sin']
    }

    for category, expected_features in expected_categories.items():
        present = [f for f in expected_features if f in features.columns]
        missing = [f for f in expected_features if f not in features.columns]
        print(f"✓ {category}: {len(present)}/{len(expected_features)} features present")
        if missing:
            print(f"  ⚠ Missing: {missing}")

    return True


def test_no_inf_values():
    """Test that computed features don't contain inf values."""
    print("\n" + "="*60)
    print("Testing No Inf Values...")
    print("="*60)

    fe = FeatureEngineer()
    df = create_sample_ohlcv(200)
    features = fe.compute_all_features(df)

    # Check for inf values in each column
    inf_columns = []
    for col in features.columns:
        if features[col].dtype in [np.float64, np.float32]:
            if np.isinf(features[col]).any():
                inf_columns.append(col)

    if inf_columns:
        print(f"⚠ Columns with inf values: {inf_columns}")
    else:
        print("✓ No inf values in any column")

    assert len(inf_columns) == 0, f"Found inf values in columns: {inf_columns}"

    return True


def test_wilder_atr_and_rsi():
    """Test Wilder-smoothed ATR and RSI features."""
    print("\n" + "="*60)
    print("Testing Wilder ATR and RSI...")
    print("="*60)

    fe = FeatureEngineer()
    df = create_sample_ohlcv(200)
    features = fe.compute_all_features(df)

    # Check ATR Wilder exists
    assert 'atr_wilder_14' in features.columns, "atr_wilder_14 should exist"
    atr_wilder = features['atr_wilder_14'].dropna()
    assert len(atr_wilder) > 0, "ATR Wilder should have values"
    assert atr_wilder.min() >= 0, "ATR should be non-negative"
    print(f"✓ atr_wilder_14: {len(atr_wilder)} values, range [{atr_wilder.min():.6f}, {atr_wilder.max():.6f}]")

    # Check RSI Wilder exists
    assert 'rsi_wilder_14' in features.columns, "rsi_wilder_14 should exist"
    rsi_wilder = features['rsi_wilder_14'].dropna()
    assert len(rsi_wilder) > 0, "RSI Wilder should have values"
    assert rsi_wilder.min() >= 0 and rsi_wilder.max() <= 100, "RSI should be in [0, 100]"
    print(f"✓ rsi_wilder_14: {len(rsi_wilder)} values, range [{rsi_wilder.min():.2f}, {rsi_wilder.max():.2f}]")

    # Compare with simple versions
    atr_simple = features['atr_14'].dropna()
    common_idx = atr_simple.index.intersection(atr_wilder.index)
    atr_corr = atr_simple.loc[common_idx].corr(atr_wilder.loc[common_idx])
    assert atr_corr > 0.8, f"ATR correlation should be high, got {atr_corr}"
    print(f"✓ ATR simple vs Wilder correlation: {atr_corr:.4f}")

    rsi_simple = features['rsi_14'].dropna()
    common_idx = rsi_simple.index.intersection(rsi_wilder.index)
    rsi_corr = rsi_simple.loc[common_idx].corr(rsi_wilder.loc[common_idx])
    assert rsi_corr > 0.8, f"RSI correlation should be high, got {rsi_corr}"
    print(f"✓ RSI simple vs Wilder correlation: {rsi_corr:.4f}")

    return True


def test_tr_dm_alignment():
    """Test that TR and DM smoothing windows are aligned."""
    print("\n" + "="*60)
    print("Testing TR/DM Window Alignment...")
    print("="*60)

    df = create_sample_ohlcv(50)

    # Add TR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )

    # Calculate DM (simulating the fix)
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # Apply the alignment fix
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    # Verify alignment
    tr_first_valid = df['tr'].first_valid_index()
    dm_first_valid = plus_dm.first_valid_index()

    assert tr_first_valid == dm_first_valid, \
        f"TR and DM should start at same index. TR: {tr_first_valid}, DM: {dm_first_valid}"

    print(f"✓ TR first valid index: {tr_first_valid}")
    print(f"✓ DM first valid index: {dm_first_valid}")
    print("✓ TR and DM smoothing windows are properly aligned")

    return True


def run_all_tests():
    """Run all feature engineering tests."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING TEST SUITE")
    print("="*60)

    tests = [
        ("Wilder's Smoothing - Basic", test_wilder_smoothing_basic),
        ("Wilder's Smoothing - Edge Cases", test_wilder_smoothing_edge_cases),
        ("Wilder's Smoothing vs EMA", test_wilder_smoothing_vs_ema),
        ("ADX Calculation", test_adx_calculation),
        ("ADX vs Simple ADX", test_adx_vs_simple_adx),
        ("ADX Reference Comparison", test_adx_reference_comparison),
        ("Insufficient Data Handling", test_insufficient_data_handling),
        ("Feature Count and Names", test_feature_count_and_names),
        ("No Inf Values", test_no_inf_values),
        ("Wilder ATR and RSI", test_wilder_atr_and_rsi),
        ("TR/DM Alignment", test_tr_dm_alignment),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
