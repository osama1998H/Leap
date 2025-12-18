"""
Tests for feature engineering registry.

Tests the FeatureRegistry singleton, decorator-based registration,
dependency resolution, and feature computation.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List

from core.feature_registry import (
    FeatureRegistry,
    FeatureCategory,
    FeatureSpec,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    FeatureRegistry.reset()
    yield
    FeatureRegistry.reset()


class TestFeatureCategory:
    """Tests for FeatureCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        categories = [
            FeatureCategory.PRICE,
            FeatureCategory.MOVING_AVERAGE,
            FeatureCategory.MOMENTUM,
            FeatureCategory.VOLATILITY,
            FeatureCategory.VOLUME,
            FeatureCategory.TREND,
            FeatureCategory.CANDLESTICK,
            FeatureCategory.TIME,
            FeatureCategory.CUSTOM,
        ]

        for cat in categories:
            assert cat.value is not None

    def test_category_values(self):
        """Test category string values."""
        assert FeatureCategory.PRICE.value == "price"
        assert FeatureCategory.MOMENTUM.value == "momentum"
        assert FeatureCategory.CUSTOM.value == "custom"


class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_create_spec(self):
        """Test creating a feature specification."""
        def dummy_fn(df):
            return df['close']

        spec = FeatureSpec(
            name='test_feature',
            compute_fn=dummy_fn,
            category=FeatureCategory.PRICE,
            dependencies=['close'],
            description='Test description',
            enabled=True,
            is_key_feature=True
        )

        assert spec.name == 'test_feature'
        assert spec.compute_fn is dummy_fn
        assert spec.category == FeatureCategory.PRICE
        assert spec.dependencies == ['close']
        assert spec.description == 'Test description'
        assert spec.enabled is True
        assert spec.is_key_feature is True

    def test_spec_defaults(self):
        """Test feature spec default values."""
        def dummy_fn(df):
            return df['close']

        spec = FeatureSpec(
            name='test',
            compute_fn=dummy_fn,
            category=FeatureCategory.CUSTOM
        )

        assert spec.dependencies == []
        assert spec.description == ""
        assert spec.enabled is True
        assert spec.is_key_feature is False
        assert spec.params == {}


class TestFeatureRegistrySingleton:
    """Tests for FeatureRegistry singleton pattern."""

    def test_singleton_instance(self):
        """Test that registry is a singleton."""
        registry1 = FeatureRegistry()
        registry2 = FeatureRegistry()

        assert registry1 is registry2

    def test_get_instance(self):
        """Test get_instance returns same instance."""
        registry1 = FeatureRegistry.get_instance()
        registry2 = FeatureRegistry.get_instance()

        assert registry1 is registry2

    def test_reset(self):
        """Test registry reset."""
        registry = FeatureRegistry.get_instance()

        # Register a feature
        @FeatureRegistry.register(name='test')
        def test_fn(df):
            return df['close']

        assert 'test' in registry.get_all_features()

        # Reset
        FeatureRegistry.reset()

        # Get new instance (without built-in registration)
        registry2 = FeatureRegistry()
        assert len(registry2.get_all_features()) == 0


class TestFeatureRegistration:
    """Tests for feature registration."""

    def test_register_with_decorator(self):
        """Test registering feature with decorator."""
        @FeatureRegistry.register(
            name='my_feature',
            category=FeatureCategory.MOMENTUM,
            dependencies=['close'],
            description='My custom feature',
            is_key_feature=True
        )
        def compute_my_feature(df: pd.DataFrame) -> pd.Series:
            return df['close'].rolling(10).mean()

        registry = FeatureRegistry()
        spec = registry.get_feature('my_feature')

        assert spec is not None
        assert spec.name == 'my_feature'
        assert spec.category == FeatureCategory.MOMENTUM
        assert spec.dependencies == ['close']
        assert spec.description == 'My custom feature'
        assert spec.is_key_feature is True

    def test_register_default_name(self):
        """Test registration uses function name as default."""
        @FeatureRegistry.register(category=FeatureCategory.PRICE)
        def custom_indicator(df: pd.DataFrame) -> pd.Series:
            return df['close']

        registry = FeatureRegistry()
        assert 'custom_indicator' in registry.get_all_features()

    def test_register_multiple_features(self):
        """Test registering multiple features."""
        @FeatureRegistry.register(name='feat1')
        def fn1(df):
            return df['close']

        @FeatureRegistry.register(name='feat2')
        def fn2(df):
            return df['open']

        @FeatureRegistry.register(name='feat3')
        def fn3(df):
            return df['high']

        registry = FeatureRegistry()
        assert len(registry.get_all_features()) == 3

    def test_add_feature_directly(self):
        """Test adding feature specification directly."""
        registry = FeatureRegistry()

        spec = FeatureSpec(
            name='direct_feature',
            compute_fn=lambda df: df['close'],
            category=FeatureCategory.CUSTOM
        )

        registry.add_feature(spec)
        assert registry.get_feature('direct_feature') is spec

    def test_remove_feature(self):
        """Test removing a feature."""
        @FeatureRegistry.register(name='to_remove')
        def fn(df):
            return df['close']

        registry = FeatureRegistry()
        assert 'to_remove' in registry.get_all_features()

        result = registry.remove_feature('to_remove')
        assert result is True
        assert 'to_remove' not in registry.get_all_features()

    def test_remove_nonexistent_feature(self):
        """Test removing non-existent feature returns False."""
        registry = FeatureRegistry()
        result = registry.remove_feature('does_not_exist')
        assert result is False


class TestFeatureEnableDisable:
    """Tests for enabling/disabling features."""

    def test_enable_feature(self):
        """Test enabling a feature."""
        @FeatureRegistry.register(name='toggleable', enabled=False)
        def fn(df):
            return df['close']

        registry = FeatureRegistry()
        spec = registry.get_feature('toggleable')
        assert spec.enabled is False

        registry.enable_feature('toggleable')
        assert spec.enabled is True

    def test_disable_feature(self):
        """Test disabling a feature."""
        @FeatureRegistry.register(name='toggleable', enabled=True)
        def fn(df):
            return df['close']

        registry = FeatureRegistry()
        registry.disable_feature('toggleable')

        spec = registry.get_feature('toggleable')
        assert spec.enabled is False


class TestFeatureFiltering:
    """Tests for feature filtering methods."""

    @pytest.fixture
    def populated_registry(self):
        """Create registry with various features."""
        @FeatureRegistry.register(
            name='price_feat',
            category=FeatureCategory.PRICE,
            is_key_feature=True
        )
        def fn1(df):
            return df['close']

        @FeatureRegistry.register(
            name='momentum_feat',
            category=FeatureCategory.MOMENTUM,
            is_key_feature=False
        )
        def fn2(df):
            return df['close']

        @FeatureRegistry.register(
            name='disabled_feat',
            category=FeatureCategory.PRICE,
            enabled=False
        )
        def fn3(df):
            return df['close']

        @FeatureRegistry.register(
            name='key_momentum',
            category=FeatureCategory.MOMENTUM,
            is_key_feature=True
        )
        def fn4(df):
            return df['close']

        return FeatureRegistry()

    def test_get_feature_names_all(self, populated_registry):
        """Test getting all enabled feature names."""
        names = populated_registry.get_feature_names(enabled_only=True)

        assert 'price_feat' in names
        assert 'momentum_feat' in names
        assert 'key_momentum' in names
        assert 'disabled_feat' not in names

    def test_get_feature_names_by_category(self, populated_registry):
        """Test filtering by category."""
        names = populated_registry.get_feature_names(
            categories=[FeatureCategory.MOMENTUM]
        )

        assert 'momentum_feat' in names
        assert 'key_momentum' in names
        assert 'price_feat' not in names

    def test_get_feature_names_key_only(self, populated_registry):
        """Test filtering key features only."""
        names = populated_registry.get_feature_names(key_only=True)

        assert 'price_feat' in names
        assert 'key_momentum' in names
        assert 'momentum_feat' not in names

    def test_get_features_by_category(self, populated_registry):
        """Test getting features by category."""
        features = populated_registry.get_features_by_category(
            FeatureCategory.MOMENTUM
        )

        assert len(features) == 2
        names = [f.name for f in features]
        assert 'momentum_feat' in names
        assert 'key_momentum' in names

    def test_get_category_counts(self, populated_registry):
        """Test getting feature count per category."""
        counts = populated_registry.get_category_counts()

        assert counts['price'] == 2
        assert counts['momentum'] == 2


class TestFeatureComputation:
    """Tests for feature computation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        n_bars = 100
        np.random.seed(42)

        close = np.cumsum(np.random.randn(n_bars) * 0.001) + 1.10
        high = close + np.random.uniform(0.0001, 0.001, n_bars)
        low = close - np.random.uniform(0.0001, 0.001, n_bars)
        open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n_bars)

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 5000, n_bars).astype(float),
        })

    def test_compute_single_feature(self, sample_data):
        """Test computing a single feature."""
        @FeatureRegistry.register(
            name='test_returns',
            category=FeatureCategory.PRICE,
            dependencies=['close']
        )
        def compute_returns(df: pd.DataFrame) -> pd.Series:
            return df['close'].pct_change()

        registry = FeatureRegistry()
        result = registry.compute_feature(sample_data, 'test_returns')

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_compute_feature_not_found(self, sample_data):
        """Test computing non-existent feature raises error."""
        registry = FeatureRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.compute_feature(sample_data, 'nonexistent')

    def test_compute_feature_missing_dependency(self, sample_data):
        """Test computing feature with missing dependency."""
        @FeatureRegistry.register(
            name='needs_missing',
            dependencies=['missing_column']
        )
        def fn(df):
            return df['missing_column']

        registry = FeatureRegistry()

        with pytest.raises(ValueError, match="requires column"):
            registry.compute_feature(sample_data, 'needs_missing')

    def test_compute_all_features(self, sample_data):
        """Test computing all features."""
        @FeatureRegistry.register(name='feat1', dependencies=['close'])
        def fn1(df):
            return df['close'].rolling(5).mean()

        @FeatureRegistry.register(name='feat2', dependencies=['close'])
        def fn2(df):
            return df['close'].pct_change()

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data)

        assert 'feat1' in result.columns
        assert 'feat2' in result.columns
        assert 'close' in result.columns  # Original columns preserved

    def test_compute_features_by_category(self, sample_data):
        """Test computing features by category."""
        @FeatureRegistry.register(
            name='price_feat',
            category=FeatureCategory.PRICE,
            dependencies=['close']
        )
        def fn1(df):
            return df['close'].rolling(5).mean()

        @FeatureRegistry.register(
            name='momentum_feat',
            category=FeatureCategory.MOMENTUM,
            dependencies=['close']
        )
        def fn2(df):
            return df['close'].pct_change()

        registry = FeatureRegistry()
        result = registry.compute_all(
            sample_data,
            categories=[FeatureCategory.PRICE]
        )

        assert 'price_feat' in result.columns
        assert 'momentum_feat' not in result.columns

    def test_compute_specific_features(self, sample_data):
        """Test computing specific named features."""
        @FeatureRegistry.register(name='feat1')
        def fn1(df):
            return df['close'].rolling(5).mean()

        @FeatureRegistry.register(name='feat2')
        def fn2(df):
            return df['close'].pct_change()

        @FeatureRegistry.register(name='feat3')
        def fn3(df):
            return df['high'] - df['low']

        registry = FeatureRegistry()
        result = registry.compute_all(
            sample_data,
            feature_names=['feat1', 'feat3']
        )

        assert 'feat1' in result.columns
        assert 'feat3' in result.columns
        assert 'feat2' not in result.columns

    def test_compute_excludes_disabled(self, sample_data):
        """Test disabled features are excluded."""
        @FeatureRegistry.register(name='enabled_feat')
        def fn1(df):
            return df['close'].rolling(5).mean()

        @FeatureRegistry.register(name='disabled_feat', enabled=False)
        def fn2(df):
            return df['close'].pct_change()

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data)

        assert 'enabled_feat' in result.columns
        assert 'disabled_feat' not in result.columns

    def test_compute_includes_disabled_when_requested(self, sample_data):
        """Test disabled features included when requested."""
        @FeatureRegistry.register(name='disabled_feat', enabled=False)
        def fn(df):
            return df['close'].pct_change()

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data, include_disabled=True)

        assert 'disabled_feat' in result.columns


class TestDependencyResolution:
    """Tests for feature dependency resolution."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

    def test_simple_dependency(self, sample_data):
        """Test simple dependency resolution."""
        @FeatureRegistry.register(
            name='base_feature',
            dependencies=['close']
        )
        def compute_base(df):
            return df['close'].rolling(2).mean()

        @FeatureRegistry.register(
            name='derived_feature',
            dependencies=['base_feature']
        )
        def compute_derived(df):
            return df['base_feature'] * 2

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data)

        # base_feature should be computed first
        assert 'base_feature' in result.columns
        assert 'derived_feature' in result.columns

        # Verify derived is computed after base
        # derived should be 2x base (ignoring NaN)
        valid_idx = ~result['base_feature'].isna()
        expected = result.loc[valid_idx, 'base_feature'] * 2
        actual = result.loc[valid_idx, 'derived_feature']
        np.testing.assert_array_almost_equal(
            expected.values,
            actual.values
        )

    def test_multi_level_dependency(self, sample_data):
        """Test multi-level dependency chain."""
        @FeatureRegistry.register(name='level1', dependencies=['close'])
        def fn1(df):
            return df['close'] + 1

        @FeatureRegistry.register(name='level2', dependencies=['level1'])
        def fn2(df):
            return df['level1'] + 1

        @FeatureRegistry.register(name='level3', dependencies=['level2'])
        def fn3(df):
            return df['level2'] + 1

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data)

        # All should be computed
        assert 'level1' in result.columns
        assert 'level2' in result.columns
        assert 'level3' in result.columns

        # Verify chain
        assert (result['level1'] == result['close'] + 1).all()
        assert (result['level2'] == result['level1'] + 1).all()
        assert (result['level3'] == result['level2'] + 1).all()

    def test_multiple_dependencies(self, sample_data):
        """Test feature with multiple dependencies."""
        @FeatureRegistry.register(name='dep1', dependencies=['close'])
        def fn1(df):
            return df['close'] * 2

        @FeatureRegistry.register(name='dep2', dependencies=['volume'])
        def fn2(df):
            return df['volume'] / 100

        @FeatureRegistry.register(
            name='combined',
            dependencies=['dep1', 'dep2']
        )
        def fn3(df):
            return df['dep1'] + df['dep2']

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data)

        assert 'combined' in result.columns
        expected = result['dep1'] + result['dep2']
        np.testing.assert_array_almost_equal(
            result['combined'].values,
            expected.values
        )


class TestBuiltinFeatures:
    """Tests for built-in feature registration."""

    @pytest.fixture
    def registry_with_builtins(self):
        """Get registry with built-in features loaded."""
        return FeatureRegistry.get_instance()

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        n_bars = 100
        np.random.seed(42)

        close = np.cumsum(np.random.randn(n_bars) * 0.001) + 1.10
        high = close + np.random.uniform(0.0001, 0.001, n_bars)
        low = close - np.random.uniform(0.0001, 0.001, n_bars)
        open_ = low + (high - low) * np.random.uniform(0.3, 0.7, n_bars)

        return pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 5000, n_bars).astype(float),
        })

    def test_builtins_registered(self, registry_with_builtins):
        """Test that built-in features are registered."""
        # Should have many features
        assert len(registry_with_builtins) > 50

    def test_returns_feature_exists(self, registry_with_builtins):
        """Test returns feature is registered."""
        spec = registry_with_builtins.get_feature('returns')
        assert spec is not None
        assert spec.category == FeatureCategory.PRICE

    def test_rsi_feature_exists(self, registry_with_builtins):
        """Test RSI feature is registered."""
        spec = registry_with_builtins.get_feature('rsi_14')
        assert spec is not None
        assert spec.category == FeatureCategory.MOMENTUM

    def test_sma_features_exist(self, registry_with_builtins):
        """Test SMA features are registered."""
        sma_features = ['sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200']
        for name in sma_features:
            spec = registry_with_builtins.get_feature(name)
            assert spec is not None, f"Missing SMA feature: {name}"
            assert spec.category == FeatureCategory.MOVING_AVERAGE

    def test_compute_builtin_features(
        self, registry_with_builtins, sample_data
    ):
        """Test computing built-in features."""
        result = registry_with_builtins.compute_all(
            sample_data,
            feature_names=['returns', 'sma_20', 'rsi_14']
        )

        assert 'returns' in result.columns
        assert 'sma_20' in result.columns
        assert 'rsi_14' in result.columns

    def test_key_features(self, registry_with_builtins):
        """Test key features are marked correctly."""
        key_features = registry_with_builtins.get_feature_names(key_only=True)

        # Should have some key features
        assert len(key_features) > 0

        # Returns should be a key feature
        assert 'returns' in key_features

    def test_category_distribution(self, registry_with_builtins):
        """Test features are distributed across categories."""
        counts = registry_with_builtins.get_category_counts()

        # Should have features in multiple categories
        assert counts.get('price', 0) > 0
        assert counts.get('moving_average', 0) > 0
        assert counts.get('momentum', 0) > 0
        assert counts.get('volatility', 0) > 0


class TestRegistryLen:
    """Tests for registry length method."""

    def test_len_empty(self):
        """Test length of empty registry."""
        registry = FeatureRegistry()
        assert len(registry) == 0

    def test_len_after_registration(self):
        """Test length after registering features."""
        @FeatureRegistry.register(name='f1')
        def fn1(df):
            return df['close']

        @FeatureRegistry.register(name='f2')
        def fn2(df):
            return df['open']

        registry = FeatureRegistry()
        assert len(registry) == 2

    def test_len_after_remove(self):
        """Test length after removing feature."""
        @FeatureRegistry.register(name='f1')
        def fn1(df):
            return df['close']

        @FeatureRegistry.register(name='f2')
        def fn2(df):
            return df['open']

        registry = FeatureRegistry()
        assert len(registry) == 2

        registry.remove_feature('f1')
        assert len(registry) == 1


class TestErrorHandling:
    """Tests for error handling in feature computation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

    def test_failed_computation_returns_nan(self, sample_data):
        """Test failed computation returns NaN instead of crashing."""
        @FeatureRegistry.register(name='failing_feature')
        def fn(df):
            raise ValueError("Intentional error")

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data)

        assert 'failing_feature' in result.columns
        assert result['failing_feature'].isna().all()

    def test_partial_failure_continues(self, sample_data):
        """Test computation continues after one feature fails."""
        @FeatureRegistry.register(name='failing_feature')
        def fn1(df):
            raise ValueError("Intentional error")

        @FeatureRegistry.register(name='working_feature')
        def fn2(df):
            return df['close'] * 2

        registry = FeatureRegistry()
        result = registry.compute_all(sample_data)

        # Working feature should be computed
        assert 'working_feature' in result.columns
        assert (result['working_feature'] == sample_data['close'] * 2).all()
