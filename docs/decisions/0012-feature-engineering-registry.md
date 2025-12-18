# ADR 0012: Feature Engineering Registry

## Status
Accepted

## Context

The `FeatureEngineer` class in `core/data_pipeline.py` has several issues:

1. **Monolithic**: Single 500+ line class with all feature computations hardcoded
2. **No customization**: Cannot add custom features without modifying core code
3. **No metadata**: Features lack category, dependencies, or descriptions
4. **Fixed selection**: Hard to select subsets of features for experiments
5. **No validation**: Dependencies not explicitly declared or validated
6. **Scattered logic**: Feature computation mixed with pipeline orchestration

## Decision

Introduce a `FeatureRegistry` singleton with decorator-based registration:

### 1. FeatureCategory Enum

```python
class FeatureCategory(Enum):
    PRICE = "price"
    MOVING_AVERAGE = "moving_average"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    CANDLESTICK = "candlestick"
    TIME = "time"
    CUSTOM = "custom"
```

### 2. FeatureSpec Dataclass

```python
@dataclass
class FeatureSpec:
    name: str
    compute_fn: Callable[[pd.DataFrame], pd.Series]
    category: FeatureCategory
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    enabled: bool = True
    is_key_feature: bool = False  # For multi-timeframe selection
    params: Dict[str, Any] = field(default_factory=dict)
```

### 3. FeatureRegistry Singleton

```python
class FeatureRegistry:
    _instance: Optional['FeatureRegistry'] = None
    _features: Dict[str, FeatureSpec] = {}

    @classmethod
    def get_instance(cls) -> 'FeatureRegistry':
        """Get singleton instance with built-in features loaded."""
        ...

    @classmethod
    def register(
        cls,
        name: Optional[str] = None,
        category: FeatureCategory = FeatureCategory.CUSTOM,
        dependencies: Optional[List[str]] = None,
        is_key_feature: bool = False,
        **kwargs
    ) -> Callable:
        """Decorator to register a feature."""
        ...

    def compute_all(
        self,
        df: pd.DataFrame,
        categories: Optional[List[FeatureCategory]] = None,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compute features with dependency resolution."""
        ...

    def get_feature_names(
        self,
        categories: Optional[List[FeatureCategory]] = None,
        key_only: bool = False,
        enabled_only: bool = True
    ) -> List[str]:
        """Get feature names with filters."""
        ...
```

### 4. Decorator-Based Registration

```python
@FeatureRegistry.register(
    name='rsi_14',
    category=FeatureCategory.MOMENTUM,
    dependencies=['close'],
    is_key_feature=True,
    description='14-period Relative Strength Index'
)
def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))
```

### 5. Dependency Resolution

Topological sort ensures features are computed in correct order:

```python
def _resolve_dependencies(self, specs, already_computed) -> List[FeatureSpec]:
    """Sort features by dependencies using Kahn's algorithm."""
    # Build dependency graph
    # Perform topological sort
    # Return sorted list
```

### 6. Built-in Feature Migration

All ~90 existing features migrated to registry pattern:

| Category | Count | Examples |
|----------|-------|----------|
| PRICE | 5 | returns, log_returns, high_low_range |
| MOVING_AVERAGE | 15 | sma_5/10/20/50/200, ema_12/26, wma, tema |
| MOMENTUM | 15 | rsi_14, macd, stochastic, williams_r, cci |
| VOLATILITY | 10 | atr_14, bb_upper/lower/width, keltner_channels |
| VOLUME | 8 | volume_sma, obv, vwap, force_index |
| TREND | 12 | adx, aroon, parabolic_sar, supertrend |
| CANDLESTICK | 15 | doji, hammer, engulfing, harami, etc. |
| TIME | 6 | hour_sin/cos, day_sin/cos, week_sin/cos |

## Consequences

### Positive
- **Extensibility**: Add custom features without modifying core code
- **Metadata**: Each feature has category, dependencies, description
- **Filtering**: Select features by category or key status
- **Dependency safety**: Automatic resolution ensures correct computation order
- **Testing**: Individual features can be tested in isolation
- **Enable/disable**: Features can be toggled without code changes
- **Discovery**: `get_feature_names()` and `get_category_counts()` for introspection

### Negative
- **Performance**: Slight overhead from registry lookup (negligible)
- **Complexity**: More moving parts than monolithic class
- **Migration effort**: All features need to be refactored (one-time cost)

## Implementation Files

| File | Purpose |
|------|---------|
| `core/feature_registry.py` | Registry singleton + all built-in features |
| `tests/test_feature_registry.py` | Comprehensive unit tests |

## Usage Examples

### Adding Custom Feature

```python
from core.feature_registry import FeatureRegistry, FeatureCategory

@FeatureRegistry.register(
    name='my_custom_indicator',
    category=FeatureCategory.CUSTOM,
    dependencies=['close', 'sma_20'],
    is_key_feature=False
)
def compute_my_indicator(df: pd.DataFrame) -> pd.Series:
    return df['close'] / df['sma_20'] - 1
```

### Selecting Features

```python
registry = FeatureRegistry.get_instance()

# Get all momentum features
momentum = registry.get_feature_names(categories=[FeatureCategory.MOMENTUM])

# Get key features for multi-timeframe
key_features = registry.get_feature_names(key_only=True)

# Compute specific features
result = registry.compute_all(df, feature_names=['rsi_14', 'macd', 'sma_20'])
```

### Disabling Features

```python
registry = FeatureRegistry.get_instance()
registry.disable_feature('volume_sma')  # Exclude from computation
```

## Migration Path

1. ✅ Create `core/feature_registry.py` with registry implementation
2. ✅ Migrate all ~90 built-in features to decorator registration
3. ✅ Create unit tests
4. ⏳ Refactor `FeatureEngineer` to use registry as facade (future)
5. ⏳ Add registry configuration to config files (future)

## Alternatives Considered

### 1. Configuration-based features
- Feature definitions in YAML/JSON
- Pro: No code for simple features
- Con: Complex features need code anyway; two places to look

### 2. Plugin architecture
- External feature packages
- Pro: Maximum extensibility
- Con: Overkill for internal use; dependency management

### 3. Keep monolithic FeatureEngineer
- Continue as-is
- Con: All the problems listed in Context
