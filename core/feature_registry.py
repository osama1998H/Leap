"""
Leap Trading System - Feature Engineering Registry

Provides a decorator-based registration system for feature computation functions.
Supports dependency resolution, category-based filtering, and custom feature addition.

See ADR-0012 for design rationale.
"""

import logging
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Categories
# =============================================================================

class FeatureCategory(Enum):
    """Feature categories for organization and filtering."""
    PRICE = "price"
    MOVING_AVERAGE = "moving_average"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    CANDLESTICK = "candlestick"
    TIME = "time"
    CUSTOM = "custom"


# =============================================================================
# Feature Specification
# =============================================================================

@dataclass
class FeatureSpec:
    """
    Specification for a registered feature.

    Attributes:
        name: Unique feature identifier
        compute_fn: Function that computes the feature
        category: Feature category for filtering
        dependencies: List of column names this feature depends on
        description: Human-readable description
        enabled: Whether this feature is active
        is_key_feature: Include in multi-timeframe selection
        params: Optional parameters for the compute function
    """
    name: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    category: FeatureCategory
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    enabled: bool = True
    is_key_feature: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Feature Registry (Singleton)
# =============================================================================

class FeatureRegistry:
    """
    Registry for feature computation functions.

    Singleton pattern ensures a single global registry.

    Supports:
    - Decorator-based registration
    - Dependency resolution via topological sort
    - Category-based selection
    - Custom feature addition without modifying core code

    Example:
        >>> @FeatureRegistry.register(
        ...     name='my_indicator',
        ...     category=FeatureCategory.MOMENTUM,
        ...     dependencies=['close'],
        ...     is_key_feature=True
        ... )
        ... def compute_my_indicator(df: pd.DataFrame) -> pd.Series:
        ...     return df['close'].rolling(10).mean()
        ...
        >>> registry = FeatureRegistry.get_instance()
        >>> result = registry.compute_all(df)
    """

    _instance: Optional['FeatureRegistry'] = None
    _features: Dict[str, FeatureSpec] = {}
    _initialized: bool = False

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'FeatureRegistry':
        """
        Get singleton instance.

        Initializes with built-in features on first call.
        """
        if cls._instance is None:
            cls._instance = cls()

        if not cls._initialized:
            _register_builtin_features()
            cls._initialized = True

        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Reset registry to empty state.

        Useful for testing.
        """
        cls._features = {}
        cls._initialized = False

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
        category: FeatureCategory = FeatureCategory.CUSTOM,
        dependencies: Optional[List[str]] = None,
        description: str = "",
        is_key_feature: bool = False,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None
    ) -> Callable:
        """
        Decorator to register a feature computation function.

        Args:
            name: Feature name (defaults to function name)
            category: Feature category
            dependencies: Column names this feature depends on
            description: Human-readable description
            is_key_feature: Include in multi-timeframe selection
            enabled: Whether feature is active
            params: Optional parameters for computation

        Returns:
            Decorator function

        Example:
            >>> @FeatureRegistry.register(
            ...     name='rsi_14',
            ...     category=FeatureCategory.MOMENTUM,
            ...     dependencies=['close'],
            ...     is_key_feature=True
            ... )
            ... def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
            ...     delta = df['close'].diff()
            ...     gain = delta.where(delta > 0, 0).rolling(14).mean()
            ...     loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            ...     rs = gain / (loss + 1e-10)
            ...     return 100 - (100 / (1 + rs))
        """
        def decorator(fn: Callable[[pd.DataFrame], pd.Series]) -> Callable:
            feature_name = name or fn.__name__
            spec = FeatureSpec(
                name=feature_name,
                compute_fn=fn,
                category=category,
                dependencies=dependencies or [],
                description=description or fn.__doc__ or "",
                is_key_feature=is_key_feature,
                enabled=enabled,
                params=params or {}
            )
            cls._features[feature_name] = spec
            logger.debug(f"Registered feature: {feature_name} ({category.value})")
            return fn
        return decorator

    def add_feature(self, spec: FeatureSpec) -> None:
        """
        Add a feature specification directly.

        Args:
            spec: Feature specification
        """
        self._features[spec.name] = spec
        logger.debug(f"Added feature: {spec.name}")

    def remove_feature(self, name: str) -> bool:
        """
        Remove a feature by name.

        Args:
            name: Feature name

        Returns:
            True if feature was removed
        """
        if name in self._features:
            del self._features[name]
            return True
        return False

    def enable_feature(self, name: str) -> None:
        """Enable a feature by name."""
        if name in self._features:
            self._features[name].enabled = True

    def disable_feature(self, name: str) -> None:
        """Disable a feature by name."""
        if name in self._features:
            self._features[name].enabled = False

    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """Get feature specification by name."""
        return self._features.get(name)

    def get_all_features(self) -> Dict[str, FeatureSpec]:
        """Get all registered features."""
        return dict(self._features)

    def get_feature_names(
        self,
        categories: Optional[List[FeatureCategory]] = None,
        key_only: bool = False,
        enabled_only: bool = True
    ) -> List[str]:
        """
        Get list of feature names.

        Args:
            categories: Filter by categories (None for all)
            key_only: Only return key features
            enabled_only: Only return enabled features

        Returns:
            List of feature names
        """
        features = self._features.values()

        if enabled_only:
            features = [f for f in features if f.enabled]

        if categories:
            features = [f for f in features if f.category in categories]

        if key_only:
            features = [f for f in features if f.is_key_feature]

        return [f.name for f in features]

    def get_features_by_category(
        self,
        category: FeatureCategory
    ) -> List[FeatureSpec]:
        """Get all features in a category."""
        return [f for f in self._features.values() if f.category == category]

    def compute_all(
        self,
        df: pd.DataFrame,
        categories: Optional[List[FeatureCategory]] = None,
        include_disabled: bool = False,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute all registered features with dependency resolution.

        Args:
            df: Input DataFrame with OHLCV data
            categories: Compute only features in these categories
            include_disabled: Include disabled features
            feature_names: Specific features to compute (overrides categories)

        Returns:
            DataFrame with computed features
        """
        result = df.copy()
        computed: Set[str] = set(df.columns)

        # Get features to compute
        if feature_names:
            specs_to_compute = [
                self._features[name]
                for name in feature_names
                if name in self._features
            ]
        else:
            specs_to_compute = self._get_features_to_compute(
                categories, include_disabled
            )

        # Topological sort by dependencies
        sorted_specs = self._resolve_dependencies(specs_to_compute, computed)

        # Compute features
        for spec in sorted_specs:
            try:
                result[spec.name] = spec.compute_fn(result)
                computed.add(spec.name)
            except Exception as e:
                logger.warning(f"Failed to compute feature {spec.name}: {e}")
                result[spec.name] = np.nan

        return result

    def compute_feature(
        self,
        df: pd.DataFrame,
        name: str
    ) -> pd.Series:
        """
        Compute a single feature by name.

        Args:
            df: Input DataFrame
            name: Feature name

        Returns:
            Computed feature series

        Raises:
            KeyError: If feature not found
        """
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not registered")

        spec = self._features[name]

        # Check dependencies
        for dep in spec.dependencies:
            if dep not in df.columns:
                raise ValueError(
                    f"Feature '{name}' requires column '{dep}' which is not present"
                )

        return spec.compute_fn(df)

    def _get_features_to_compute(
        self,
        categories: Optional[List[FeatureCategory]],
        include_disabled: bool
    ) -> List[FeatureSpec]:
        """Get list of features to compute based on filters."""
        features = list(self._features.values())

        if not include_disabled:
            features = [f for f in features if f.enabled]

        if categories:
            features = [f for f in features if f.category in categories]

        return features

    def _resolve_dependencies(
        self,
        specs: List[FeatureSpec],
        already_computed: Set[str]
    ) -> List[FeatureSpec]:
        """
        Sort features by dependencies using topological sort.

        Args:
            specs: Features to sort
            already_computed: Set of already available columns

        Returns:
            Sorted list of features
        """
        # Build dependency graph
        name_to_spec = {s.name: s for s in specs}
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        for spec in specs:
            for dep in spec.dependencies:
                if dep in name_to_spec and dep not in already_computed:
                    graph[dep].append(spec.name)
                    in_degree[spec.name] += 1

        # Kahn's algorithm for topological sort
        queue = [s.name for s in specs if in_degree[s.name] == 0]
        sorted_names = []

        while queue:
            name = queue.pop(0)
            sorted_names.append(name)

            for dependent in graph[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(sorted_names) != len(specs):
            missing = set(name_to_spec.keys()) - set(sorted_names)
            logger.warning(f"Circular dependencies detected for: {missing}")
            # Add remaining in original order
            sorted_names.extend(missing)

        return [name_to_spec[name] for name in sorted_names if name in name_to_spec]

    def get_category_counts(self) -> Dict[str, int]:
        """Get count of features per category."""
        counts = defaultdict(int)
        for spec in self._features.values():
            counts[spec.category.value] += 1
        return dict(counts)

    def __len__(self) -> int:
        """Return number of registered features."""
        return len(self._features)


# =============================================================================
# Built-in Feature Registration
# =============================================================================

def _register_builtin_features():
    """Register all built-in features with the registry."""

    # -------------------------------------------------------------------------
    # Price Features
    # -------------------------------------------------------------------------

    @FeatureRegistry.register(
        name='returns',
        category=FeatureCategory.PRICE,
        dependencies=['close'],
        is_key_feature=True,
        description='Simple price returns'
    )
    def compute_returns(df: pd.DataFrame) -> pd.Series:
        return df['close'].pct_change()

    @FeatureRegistry.register(
        name='log_returns',
        category=FeatureCategory.PRICE,
        dependencies=['close'],
        is_key_feature=True,
        description='Log returns'
    )
    def compute_log_returns(df: pd.DataFrame) -> pd.Series:
        return np.log(df['close'] / df['close'].shift(1))

    @FeatureRegistry.register(
        name='hl_ratio',
        category=FeatureCategory.PRICE,
        dependencies=['high', 'low', 'close'],
        description='High-low range relative to close'
    )
    def compute_hl_ratio(df: pd.DataFrame) -> pd.Series:
        return (df['high'] - df['low']) / df['close']

    @FeatureRegistry.register(
        name='oc_ratio',
        category=FeatureCategory.PRICE,
        dependencies=['open', 'close'],
        description='Open-close change ratio'
    )
    def compute_oc_ratio(df: pd.DataFrame) -> pd.Series:
        return (df['close'] - df['open']) / df['open']

    @FeatureRegistry.register(
        name='gap',
        category=FeatureCategory.PRICE,
        dependencies=['open', 'close'],
        description='Gap from previous close'
    )
    def compute_gap(df: pd.DataFrame) -> pd.Series:
        return (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    @FeatureRegistry.register(
        name='tr',
        category=FeatureCategory.PRICE,
        dependencies=['high', 'low', 'close'],
        description='True Range'
    )
    def compute_tr(df: pd.DataFrame) -> pd.Series:
        return np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )

    # -------------------------------------------------------------------------
    # Moving Averages
    # -------------------------------------------------------------------------

    for period in [5, 10, 20, 50, 100, 200]:
        # SMA
        @FeatureRegistry.register(
            name=f'sma_{period}',
            category=FeatureCategory.MOVING_AVERAGE,
            dependencies=['close'],
            is_key_feature=(period == 20),
            description=f'{period}-period Simple Moving Average',
            params={'period': period}
        )
        def compute_sma(df: pd.DataFrame, p=period) -> pd.Series:
            return df['close'].rolling(window=p).mean()

        # EMA
        @FeatureRegistry.register(
            name=f'ema_{period}',
            category=FeatureCategory.MOVING_AVERAGE,
            dependencies=['close'],
            is_key_feature=(period == 20),
            description=f'{period}-period Exponential Moving Average',
            params={'period': period}
        )
        def compute_ema(df: pd.DataFrame, p=period) -> pd.Series:
            return df['close'].ewm(span=p, adjust=False).mean()

        # Price relative to SMA
        @FeatureRegistry.register(
            name=f'close_sma_{period}_ratio',
            category=FeatureCategory.MOVING_AVERAGE,
            dependencies=['close', f'sma_{period}'],
            is_key_feature=(period == 20),
            description=f'Price relative to {period}-SMA',
            params={'period': period}
        )
        def compute_close_sma_ratio(df: pd.DataFrame, p=period) -> pd.Series:
            sma_col = f'sma_{p}'
            if sma_col in df.columns:
                return df['close'] / df[sma_col]
            return df['close'] / df['close'].rolling(window=p).mean()

    # MA Crossovers
    @FeatureRegistry.register(
        name='sma_5_20_cross',
        category=FeatureCategory.MOVING_AVERAGE,
        dependencies=['sma_5', 'sma_20'],
        description='SMA 5/20 crossover signal'
    )
    def compute_sma_5_20_cross(df: pd.DataFrame) -> pd.Series:
        if 'sma_5' in df.columns and 'sma_20' in df.columns:
            return (df['sma_5'] > df['sma_20']).astype(int)
        sma_5 = df['close'].rolling(5).mean()
        sma_20 = df['close'].rolling(20).mean()
        return (sma_5 > sma_20).astype(int)

    @FeatureRegistry.register(
        name='sma_20_50_cross',
        category=FeatureCategory.MOVING_AVERAGE,
        dependencies=['sma_20', 'sma_50'],
        description='SMA 20/50 crossover signal'
    )
    def compute_sma_20_50_cross(df: pd.DataFrame) -> pd.Series:
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            return (df['sma_20'] > df['sma_50']).astype(int)
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        return (sma_20 > sma_50).astype(int)

    # -------------------------------------------------------------------------
    # Momentum Indicators
    # -------------------------------------------------------------------------

    def _compute_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Compute RSI using simple rolling mean."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
        """Wilder's smoothing technique."""
        values = series.to_numpy()
        n = len(values)
        result = np.empty(n)
        result[:] = np.nan

        first_valid = 0
        for i in range(n):
            if not np.isnan(values[i]):
                first_valid = i
                break

        if first_valid + period > n:
            return pd.Series(result, index=series.index)

        result[first_valid + period - 1] = np.nansum(values[first_valid:first_valid + period])
        alpha = 1.0 / period

        for i in range(first_valid + period, n):
            if not np.isnan(result[i - 1]) and not np.isnan(values[i]):
                result[i] = result[i - 1] * (1 - alpha) + values[i]

        return pd.Series(result, index=series.index)

    for period in [7, 14, 21]:
        @FeatureRegistry.register(
            name=f'rsi_{period}',
            category=FeatureCategory.MOMENTUM,
            dependencies=['close'],
            is_key_feature=(period == 14),
            description=f'{period}-period RSI',
            params={'period': period}
        )
        def compute_rsi(df: pd.DataFrame, p=period) -> pd.Series:
            return _compute_rsi(df['close'], p)

    @FeatureRegistry.register(
        name='rsi_wilder_14',
        category=FeatureCategory.MOMENTUM,
        dependencies=['close'],
        is_key_feature=True,
        description='14-period RSI with Wilder smoothing'
    )
    def compute_rsi_wilder_14(df: pd.DataFrame) -> pd.Series:
        period = 14
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = _wilder_smoothing(gain, period) / period
        avg_loss = _wilder_smoothing(loss, period) / period
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    # MACD
    @FeatureRegistry.register(
        name='macd',
        category=FeatureCategory.MOMENTUM,
        dependencies=['close'],
        is_key_feature=True,
        description='MACD line'
    )
    def compute_macd(df: pd.DataFrame) -> pd.Series:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    @FeatureRegistry.register(
        name='macd_signal',
        category=FeatureCategory.MOMENTUM,
        dependencies=['macd'],
        is_key_feature=True,
        description='MACD signal line'
    )
    def compute_macd_signal(df: pd.DataFrame) -> pd.Series:
        if 'macd' in df.columns:
            return df['macd'].ewm(span=9, adjust=False).mean()
        macd = compute_macd(df)
        return macd.ewm(span=9, adjust=False).mean()

    @FeatureRegistry.register(
        name='macd_hist',
        category=FeatureCategory.MOMENTUM,
        dependencies=['macd', 'macd_signal'],
        is_key_feature=True,
        description='MACD histogram'
    )
    def compute_macd_hist(df: pd.DataFrame) -> pd.Series:
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            return df['macd'] - df['macd_signal']
        macd = compute_macd(df)
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    # Stochastic
    @FeatureRegistry.register(
        name='stoch_k_14',
        category=FeatureCategory.MOMENTUM,
        dependencies=['high', 'low', 'close'],
        is_key_feature=True,
        description='Stochastic %K (14-period)'
    )
    def compute_stoch_k_14(df: pd.DataFrame) -> pd.Series:
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        return 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)

    @FeatureRegistry.register(
        name='stoch_d_14',
        category=FeatureCategory.MOMENTUM,
        dependencies=['stoch_k_14'],
        is_key_feature=True,
        description='Stochastic %D (3-period SMA of %K)'
    )
    def compute_stoch_d_14(df: pd.DataFrame) -> pd.Series:
        if 'stoch_k_14' in df.columns:
            return df['stoch_k_14'].rolling(window=3).mean()
        stoch_k = compute_stoch_k_14(df)
        return stoch_k.rolling(window=3).mean()

    # Williams %R
    @FeatureRegistry.register(
        name='williams_r',
        category=FeatureCategory.MOMENTUM,
        dependencies=['high', 'low', 'close'],
        description='Williams %R'
    )
    def compute_williams_r(df: pd.DataFrame) -> pd.Series:
        return -100 * (df['high'].rolling(14).max() - df['close']) / \
               (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-10)

    # Rate of Change
    for period in [5, 10, 20]:
        @FeatureRegistry.register(
            name=f'roc_{period}',
            category=FeatureCategory.MOMENTUM,
            dependencies=['close'],
            description=f'{period}-period Rate of Change',
            params={'period': period}
        )
        def compute_roc(df: pd.DataFrame, p=period) -> pd.Series:
            return (df['close'] - df['close'].shift(p)) / df['close'].shift(p) * 100

    # Momentum
    for period in [10, 20]:
        @FeatureRegistry.register(
            name=f'momentum_{period}',
            category=FeatureCategory.MOMENTUM,
            dependencies=['close'],
            description=f'{period}-period Momentum',
            params={'period': period}
        )
        def compute_momentum(df: pd.DataFrame, p=period) -> pd.Series:
            return df['close'] - df['close'].shift(p)

    # -------------------------------------------------------------------------
    # Volatility Indicators
    # -------------------------------------------------------------------------

    for period in [7, 14, 21]:
        @FeatureRegistry.register(
            name=f'atr_{period}',
            category=FeatureCategory.VOLATILITY,
            dependencies=['tr'],
            is_key_feature=(period == 14),
            description=f'{period}-period ATR',
            params={'period': period}
        )
        def compute_atr(df: pd.DataFrame, p=period) -> pd.Series:
            if 'tr' in df.columns:
                return df['tr'].rolling(window=p).mean()
            tr = compute_tr(df)
            return tr.rolling(window=p).mean()

    @FeatureRegistry.register(
        name='atr_wilder_14',
        category=FeatureCategory.VOLATILITY,
        dependencies=['tr'],
        is_key_feature=True,
        description='14-period ATR with Wilder smoothing'
    )
    def compute_atr_wilder_14(df: pd.DataFrame) -> pd.Series:
        if 'tr' in df.columns:
            tr = df['tr']
        else:
            tr = compute_tr(df)
        tr_wilder_raw = _wilder_smoothing(tr, 14)
        return tr_wilder_raw / 14

    # Bollinger Bands
    @FeatureRegistry.register(
        name='bb_upper_20',
        category=FeatureCategory.VOLATILITY,
        dependencies=['close'],
        description='Upper Bollinger Band (20-period)'
    )
    def compute_bb_upper_20(df: pd.DataFrame) -> pd.Series:
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        return sma + 2 * std

    @FeatureRegistry.register(
        name='bb_lower_20',
        category=FeatureCategory.VOLATILITY,
        dependencies=['close'],
        description='Lower Bollinger Band (20-period)'
    )
    def compute_bb_lower_20(df: pd.DataFrame) -> pd.Series:
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        return sma - 2 * std

    @FeatureRegistry.register(
        name='bb_width_20',
        category=FeatureCategory.VOLATILITY,
        dependencies=['bb_upper_20', 'bb_lower_20'],
        is_key_feature=True,
        description='Bollinger Band Width (20-period)'
    )
    def compute_bb_width_20(df: pd.DataFrame) -> pd.Series:
        if 'bb_upper_20' in df.columns and 'bb_lower_20' in df.columns:
            sma = df['close'].rolling(window=20).mean()
            return (df['bb_upper_20'] - df['bb_lower_20']) / sma
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        return (4 * std) / sma

    @FeatureRegistry.register(
        name='bb_position_20',
        category=FeatureCategory.VOLATILITY,
        dependencies=['close', 'bb_upper_20', 'bb_lower_20'],
        is_key_feature=True,
        description='Position within Bollinger Bands'
    )
    def compute_bb_position_20(df: pd.DataFrame) -> pd.Series:
        if 'bb_upper_20' in df.columns and 'bb_lower_20' in df.columns:
            return (df['close'] - df['bb_lower_20']) / \
                   (df['bb_upper_20'] - df['bb_lower_20'] + 1e-10)
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (df['close'] - lower) / (upper - lower + 1e-10)

    # Keltner Channels
    @FeatureRegistry.register(
        name='keltner_upper',
        category=FeatureCategory.VOLATILITY,
        dependencies=['close', 'tr'],
        description='Upper Keltner Channel'
    )
    def compute_keltner_upper(df: pd.DataFrame) -> pd.Series:
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        if 'tr' in df.columns:
            atr_10 = df['tr'].rolling(window=10).mean()
        else:
            tr = compute_tr(df)
            atr_10 = tr.rolling(window=10).mean()
        return ema_20 + 2 * atr_10

    @FeatureRegistry.register(
        name='keltner_lower',
        category=FeatureCategory.VOLATILITY,
        dependencies=['close', 'tr'],
        description='Lower Keltner Channel'
    )
    def compute_keltner_lower(df: pd.DataFrame) -> pd.Series:
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        if 'tr' in df.columns:
            atr_10 = df['tr'].rolling(window=10).mean()
        else:
            tr = compute_tr(df)
            atr_10 = tr.rolling(window=10).mean()
        return ema_20 - 2 * atr_10

    # Historical Volatility
    for period in [10, 20, 30]:
        @FeatureRegistry.register(
            name=f'volatility_{period}',
            category=FeatureCategory.VOLATILITY,
            dependencies=['log_returns'],
            is_key_feature=(period == 20),
            description=f'{period}-period Historical Volatility',
            params={'period': period}
        )
        def compute_volatility(df: pd.DataFrame, p=period) -> pd.Series:
            if 'log_returns' in df.columns:
                return df['log_returns'].rolling(window=p).std() * np.sqrt(252)
            log_ret = np.log(df['close'] / df['close'].shift(1))
            return log_ret.rolling(window=p).std() * np.sqrt(252)

    # -------------------------------------------------------------------------
    # Volume Indicators
    # -------------------------------------------------------------------------

    @FeatureRegistry.register(
        name='volume_sma_20',
        category=FeatureCategory.VOLUME,
        dependencies=['volume'],
        description='20-period Volume SMA'
    )
    def compute_volume_sma_20(df: pd.DataFrame) -> pd.Series:
        return df['volume'].rolling(window=20).mean()

    @FeatureRegistry.register(
        name='volume_ratio',
        category=FeatureCategory.VOLUME,
        dependencies=['volume', 'volume_sma_20'],
        is_key_feature=True,
        description='Volume relative to 20-SMA'
    )
    def compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
        if 'volume_sma_20' in df.columns:
            return df['volume'] / (df['volume_sma_20'] + 1e-10)
        vol_sma = df['volume'].rolling(window=20).mean()
        return df['volume'] / (vol_sma + 1e-10)

    @FeatureRegistry.register(
        name='obv',
        category=FeatureCategory.VOLUME,
        dependencies=['close', 'volume'],
        description='On-Balance Volume'
    )
    def compute_obv(df: pd.DataFrame) -> pd.Series:
        return (np.sign(df['close'].diff()) * df['volume']).cumsum()

    @FeatureRegistry.register(
        name='vpt',
        category=FeatureCategory.VOLUME,
        dependencies=['close', 'volume'],
        description='Volume Price Trend'
    )
    def compute_vpt(df: pd.DataFrame) -> pd.Series:
        return (df['volume'] * df['close'].pct_change()).cumsum()

    @FeatureRegistry.register(
        name='ad',
        category=FeatureCategory.VOLUME,
        dependencies=['high', 'low', 'close', 'volume'],
        description='Accumulation/Distribution'
    )
    def compute_ad(df: pd.DataFrame) -> pd.Series:
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
              (df['high'] - df['low'] + 1e-10)
        return (clv * df['volume']).cumsum()

    @FeatureRegistry.register(
        name='mfi',
        category=FeatureCategory.VOLUME,
        dependencies=['high', 'low', 'close', 'volume'],
        description='Money Flow Index'
    )
    def compute_mfi(df: pd.DataFrame) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        return 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))

    # -------------------------------------------------------------------------
    # Trend Indicators
    # -------------------------------------------------------------------------

    @FeatureRegistry.register(
        name='trend',
        category=FeatureCategory.TREND,
        dependencies=['close'],
        description='Simple trend direction'
    )
    def compute_trend(df: pd.DataFrame) -> pd.Series:
        return np.where(df['close'] > df['close'].shift(1), 1, -1)

    @FeatureRegistry.register(
        name='trend_strength',
        category=FeatureCategory.TREND,
        dependencies=['trend'],
        is_key_feature=True,
        description='10-period trend strength'
    )
    def compute_trend_strength(df: pd.DataFrame) -> pd.Series:
        if 'trend' in df.columns:
            return df['trend'].rolling(10).sum()
        trend = np.where(df['close'] > df['close'].shift(1), 1, -1)
        return pd.Series(trend, index=df.index).rolling(10).sum()

    @FeatureRegistry.register(
        name='cci',
        category=FeatureCategory.TREND,
        dependencies=['high', 'low', 'close'],
        description='Commodity Channel Index'
    )
    def compute_cci(df: pd.DataFrame) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma_tp) / (0.015 * mad + 1e-10)

    # ADX - Complex indicator with multiple components
    @FeatureRegistry.register(
        name='plus_di',
        category=FeatureCategory.TREND,
        dependencies=['high', 'low', 'tr'],
        is_key_feature=True,
        description='Positive Directional Indicator (+DI)'
    )
    def compute_plus_di(df: pd.DataFrame) -> pd.Series:
        period = 14
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        plus_dm.iloc[0] = np.nan

        if 'tr' in df.columns:
            tr = df['tr']
        else:
            tr = compute_tr(df)

        tr_smoothed = _wilder_smoothing(tr, period)
        plus_dm_smoothed = _wilder_smoothing(plus_dm, period)
        return 100 * (plus_dm_smoothed / (tr_smoothed + 1e-10))

    @FeatureRegistry.register(
        name='minus_di',
        category=FeatureCategory.TREND,
        dependencies=['high', 'low', 'tr'],
        is_key_feature=True,
        description='Negative Directional Indicator (-DI)'
    )
    def compute_minus_di(df: pd.DataFrame) -> pd.Series:
        period = 14
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        minus_dm.iloc[0] = np.nan

        if 'tr' in df.columns:
            tr = df['tr']
        else:
            tr = compute_tr(df)

        tr_smoothed = _wilder_smoothing(tr, period)
        minus_dm_smoothed = _wilder_smoothing(minus_dm, period)
        return 100 * (minus_dm_smoothed / (tr_smoothed + 1e-10))

    @FeatureRegistry.register(
        name='adx',
        category=FeatureCategory.TREND,
        dependencies=['plus_di', 'minus_di'],
        is_key_feature=True,
        description='Average Directional Index'
    )
    def compute_adx(df: pd.DataFrame) -> pd.Series:
        period = 14
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            plus_di = df['plus_di']
            minus_di = df['minus_di']
        else:
            plus_di = compute_plus_di(df)
            minus_di = compute_minus_di(df)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        adx = pd.Series(index=df.index, dtype=float)
        dx_not_na = dx.notna().to_numpy()

        if dx_not_na.any():
            start_pos = int(dx_not_na.argmax())
            if len(dx) >= start_pos + period:
                first_adx_idx = start_pos + period - 1
                adx.iloc[first_adx_idx] = dx.iloc[start_pos:start_pos + period].mean()
                for i in range(first_adx_idx + 1, len(df)):
                    prior_adx = adx.iloc[i - 1]
                    current_dx = dx.iloc[i]
                    if pd.notna(prior_adx) and pd.notna(current_dx):
                        adx.iloc[i] = ((prior_adx * (period - 1)) + current_dx) / period

        return adx

    @FeatureRegistry.register(
        name='dx',
        category=FeatureCategory.TREND,
        dependencies=['plus_di', 'minus_di'],
        description='Directional Movement Index'
    )
    def compute_dx(df: pd.DataFrame) -> pd.Series:
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            plus_di = df['plus_di']
            minus_di = df['minus_di']
        else:
            plus_di = compute_plus_di(df)
            minus_di = compute_minus_di(df)
        return 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    @FeatureRegistry.register(
        name='adxr',
        category=FeatureCategory.TREND,
        dependencies=['adx'],
        description='ADX Rating (smoothed ADX)'
    )
    def compute_adxr(df: pd.DataFrame) -> pd.Series:
        period = 14
        if 'adx' in df.columns:
            adx = df['adx']
        else:
            adx = compute_adx(df)
        return (adx + adx.shift(period)) / 2

    @FeatureRegistry.register(
        name='di_crossover',
        category=FeatureCategory.TREND,
        dependencies=['plus_di', 'minus_di'],
        description='DI Crossover Signal (1 = bullish, 0 = bearish)'
    )
    def compute_di_crossover(df: pd.DataFrame) -> pd.Series:
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            plus_di = df['plus_di']
            minus_di = df['minus_di']
        else:
            plus_di = compute_plus_di(df)
            minus_di = compute_minus_di(df)
        return (plus_di > minus_di).astype(int)

    @FeatureRegistry.register(
        name='di_spread',
        category=FeatureCategory.TREND,
        dependencies=['plus_di', 'minus_di'],
        description='Normalized DI spread [-1, 1]'
    )
    def compute_di_spread(df: pd.DataFrame) -> pd.Series:
        if 'plus_di' in df.columns and 'minus_di' in df.columns:
            plus_di = df['plus_di']
            minus_di = df['minus_di']
        else:
            plus_di = compute_plus_di(df)
            minus_di = compute_minus_di(df)
        return (plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    @FeatureRegistry.register(
        name='adx_slope',
        category=FeatureCategory.TREND,
        dependencies=['adx'],
        description='5-period rate of change of ADX'
    )
    def compute_adx_slope(df: pd.DataFrame) -> pd.Series:
        if 'adx' in df.columns:
            adx = df['adx']
        else:
            adx = compute_adx(df)
        return adx.diff(5)

    @FeatureRegistry.register(
        name='adx_strength',
        category=FeatureCategory.TREND,
        dependencies=['adx'],
        description='ADX strength classification (0=weak, 1=moderate, 2=strong)'
    )
    def compute_adx_strength(df: pd.DataFrame) -> pd.Series:
        if 'adx' in df.columns:
            adx = df['adx']
        else:
            adx = compute_adx(df)
        return pd.cut(
            adx,
            bins=[-np.inf, 20, 40, np.inf],
            labels=[0, 1, 2]
        ).astype(float)

    @FeatureRegistry.register(
        name='adx_simple',
        category=FeatureCategory.TREND,
        dependencies=['high', 'low', 'tr'],
        description='Simple rolling ADX (backward compatible)'
    )
    def compute_adx_simple(df: pd.DataFrame) -> pd.Series:
        period = 14
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        if 'tr' in df.columns:
            tr = df['tr']
        else:
            tr = compute_tr(df)

        tr_14_simple = tr.rolling(period).sum()
        plus_di_simple = 100 * (plus_dm.rolling(period).sum() / (tr_14_simple + 1e-10))
        minus_di_simple = 100 * (minus_dm.rolling(period).sum() / (tr_14_simple + 1e-10))
        dx_simple = 100 * np.abs(plus_di_simple - minus_di_simple) / (plus_di_simple + minus_di_simple + 1e-10)
        return dx_simple.rolling(period).mean()

    # -------------------------------------------------------------------------
    # Candlestick Patterns
    # -------------------------------------------------------------------------

    @FeatureRegistry.register(
        name='body_size',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['open', 'close'],
        description='Candlestick body size'
    )
    def compute_body_size(df: pd.DataFrame) -> pd.Series:
        return np.abs(df['close'] - df['open'])

    @FeatureRegistry.register(
        name='upper_shadow',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['high', 'open', 'close'],
        description='Upper shadow size'
    )
    def compute_upper_shadow(df: pd.DataFrame) -> pd.Series:
        return df['high'] - np.maximum(df['close'], df['open'])

    @FeatureRegistry.register(
        name='lower_shadow',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['low', 'open', 'close'],
        description='Lower shadow size'
    )
    def compute_lower_shadow(df: pd.DataFrame) -> pd.Series:
        return np.minimum(df['close'], df['open']) - df['low']

    @FeatureRegistry.register(
        name='is_bullish',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['open', 'close'],
        description='Bullish candle indicator'
    )
    def compute_is_bullish(df: pd.DataFrame) -> pd.Series:
        return (df['close'] > df['open']).astype(int)

    @FeatureRegistry.register(
        name='is_doji',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['body_size'],
        description='Doji pattern indicator'
    )
    def compute_is_doji(df: pd.DataFrame) -> pd.Series:
        if 'body_size' in df.columns:
            body = df['body_size']
        else:
            body = np.abs(df['close'] - df['open'])
        avg_body = body.rolling(20).mean()
        return (body < avg_body * 0.1).astype(int)

    @FeatureRegistry.register(
        name='is_hammer',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['body_size', 'lower_shadow', 'upper_shadow'],
        description='Hammer pattern indicator'
    )
    def compute_is_hammer(df: pd.DataFrame) -> pd.Series:
        if 'body_size' in df.columns:
            body = df['body_size']
        else:
            body = np.abs(df['close'] - df['open'])

        if 'lower_shadow' in df.columns:
            lower = df['lower_shadow']
        else:
            lower = np.minimum(df['close'], df['open']) - df['low']

        if 'upper_shadow' in df.columns:
            upper = df['upper_shadow']
        else:
            upper = df['high'] - np.maximum(df['close'], df['open'])

        return ((lower > body * 2) & (upper < body * 0.5)).astype(int)

    @FeatureRegistry.register(
        name='is_bullish_engulfing',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['open', 'close', 'is_bullish'],
        description='Bullish engulfing pattern'
    )
    def compute_is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
        if 'is_bullish' in df.columns:
            is_bullish = df['is_bullish']
        else:
            is_bullish = (df['close'] > df['open']).astype(int)
        return (
            (is_bullish == 1) &
            (is_bullish.shift(1) == 0) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)

    @FeatureRegistry.register(
        name='is_bearish_engulfing',
        category=FeatureCategory.CANDLESTICK,
        dependencies=['open', 'close', 'is_bullish'],
        description='Bearish engulfing pattern'
    )
    def compute_is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
        if 'is_bullish' in df.columns:
            is_bullish = df['is_bullish']
        else:
            is_bullish = (df['close'] > df['open']).astype(int)
        return (
            (is_bullish == 0) &
            (is_bullish.shift(1) == 1) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)

    # -------------------------------------------------------------------------
    # Time Features
    # -------------------------------------------------------------------------

    @FeatureRegistry.register(
        name='hour',
        category=FeatureCategory.TIME,
        dependencies=['timestamp'],
        description='Hour of day'
    )
    def compute_hour(df: pd.DataFrame) -> pd.Series:
        if 'timestamp' in df.columns:
            return pd.to_datetime(df['timestamp']).dt.hour
        return pd.Series(0, index=df.index)

    @FeatureRegistry.register(
        name='day_of_week',
        category=FeatureCategory.TIME,
        dependencies=['timestamp'],
        description='Day of week (0=Monday)'
    )
    def compute_day_of_week(df: pd.DataFrame) -> pd.Series:
        if 'timestamp' in df.columns:
            return pd.to_datetime(df['timestamp']).dt.dayofweek
        return pd.Series(0, index=df.index)

    @FeatureRegistry.register(
        name='hour_sin',
        category=FeatureCategory.TIME,
        dependencies=['hour'],
        description='Cyclical encoding of hour (sin)'
    )
    def compute_hour_sin(df: pd.DataFrame) -> pd.Series:
        if 'hour' in df.columns:
            return np.sin(2 * np.pi * df['hour'] / 24)
        if 'timestamp' in df.columns:
            hour = pd.to_datetime(df['timestamp']).dt.hour
            return np.sin(2 * np.pi * hour / 24)
        return pd.Series(0, index=df.index)

    @FeatureRegistry.register(
        name='hour_cos',
        category=FeatureCategory.TIME,
        dependencies=['hour'],
        description='Cyclical encoding of hour (cos)'
    )
    def compute_hour_cos(df: pd.DataFrame) -> pd.Series:
        if 'hour' in df.columns:
            return np.cos(2 * np.pi * df['hour'] / 24)
        if 'timestamp' in df.columns:
            hour = pd.to_datetime(df['timestamp']).dt.hour
            return np.cos(2 * np.pi * hour / 24)
        return pd.Series(0, index=df.index)

    @FeatureRegistry.register(
        name='dow_sin',
        category=FeatureCategory.TIME,
        dependencies=['day_of_week'],
        description='Cyclical encoding of day of week (sin)'
    )
    def compute_dow_sin(df: pd.DataFrame) -> pd.Series:
        if 'day_of_week' in df.columns:
            return np.sin(2 * np.pi * df['day_of_week'] / 7)
        if 'timestamp' in df.columns:
            dow = pd.to_datetime(df['timestamp']).dt.dayofweek
            return np.sin(2 * np.pi * dow / 7)
        return pd.Series(0, index=df.index)

    @FeatureRegistry.register(
        name='dow_cos',
        category=FeatureCategory.TIME,
        dependencies=['day_of_week'],
        description='Cyclical encoding of day of week (cos)'
    )
    def compute_dow_cos(df: pd.DataFrame) -> pd.Series:
        if 'day_of_week' in df.columns:
            return np.cos(2 * np.pi * df['day_of_week'] / 7)
        if 'timestamp' in df.columns:
            dow = pd.to_datetime(df['timestamp']).dt.dayofweek
            return np.cos(2 * np.pi * dow / 7)
        return pd.Series(0, index=df.index)


__all__ = [
    'FeatureRegistry',
    'FeatureSpec',
    'FeatureCategory',
]
