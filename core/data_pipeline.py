"""
Leap Trading System - Advanced Data Pipeline
Handles data fetching, preprocessing, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import deque

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler


logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data."""
    symbol: str
    timeframe: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    features: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None


class FeatureEngineer:
    """
    Advanced feature engineering for trading data.
    Computes technical indicators, price patterns, and derived features.
    """

    def __init__(self, config=None):
        self.config = config
        self.feature_names = []

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features from OHLCV data."""
        features = df.copy()

        # Price-based features
        features = self._add_price_features(features)

        # Technical indicators
        features = self._add_moving_averages(features)
        features = self._add_momentum_indicators(features)
        features = self._add_volatility_indicators(features)
        features = self._add_volume_indicators(features)
        features = self._add_trend_indicators(features)

        # Pattern recognition
        features = self._add_candlestick_patterns(features)

        # Time-based features
        features = self._add_time_features(features)

        # Store feature names (excluding OHLCV)
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        self.feature_names = [c for c in features.columns if c not in base_cols]

        return features

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price ratios
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['oc_ratio'] = (df['close'] - df['open']) / df['open']

        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        periods = [5, 10, 20, 50, 100, 200]

        for p in periods:
            if len(df) >= p:
                # Simple Moving Average
                df[f'sma_{p}'] = df['close'].rolling(window=p).mean()

                # Exponential Moving Average
                df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

                # Price relative to MA
                df[f'close_sma_{p}_ratio'] = df['close'] / df[f'sma_{p}']

        # Moving average crossovers
        if len(df) >= 50:
            df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        # RSI - Simple rolling mean (traditional)
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._compute_rsi(df['close'], period)

        # RSI using Wilder's smoothing (original Wilder design, more responsive)
        for period in [14]:
            df[f'rsi_wilder_{period}'] = self._compute_rsi_wilder(df['close'], period)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic Oscillator
        for period in [14]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()

        # Williams %R
        df['williams_r'] = -100 * (df['high'].rolling(14).max() - df['close']) / \
                           (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-10)

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100

        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        # ATR - Simple rolling mean (traditional)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = df['tr'].rolling(window=period).mean()

        # ATR using Wilder's smoothing (more responsive, matches Wilder's original design)
        # Wilder's ATR is the smoothed TR divided by period (to get average)
        for period in [14]:
            tr_wilder_raw = self._wilder_smoothing(df['tr'], period)
            df[f'atr_wilder_{period}'] = tr_wilder_raw / period

        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                          (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)

        # Keltner Channel
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        atr_10 = df['tr'].rolling(window=10).mean()
        df['keltner_upper'] = ema_20 + 2 * atr_10
        df['keltner_lower'] = ema_20 - 2 * atr_10

        # Historical Volatility
        for period in [10, 20, 30]:
            df[f'volatility_{period}'] = df['log_returns'].rolling(window=period).std() * np.sqrt(252)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume Moving Average
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()

        # Volume Price Trend
        df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()

        # Accumulation/Distribution
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['ad'] = (clv * df['volume']).cumsum()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        df['mfi'] = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators including ADX and CCI."""
        # Compute ADX indicators (extracted for maintainability and testability)
        df = self._compute_adx_indicators(df)

        # Parabolic SAR (simplified)
        df['trend'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['trend_strength'] = df['trend'].rolling(10).sum()

        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        return df

    def _compute_adx_indicators(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Compute ADX (Average Directional Index) and related indicators.

        Uses Wilder's smoothing technique for accurate calculation, which is preferred
        over simple moving averages because:
        - Reduces lag while maintaining smoothness
        - More responsive to recent price action
        - Matches the original indicator design by J. Welles Wilder
        - Produces more stable signals in trending markets

        The ADX requires approximately 2 * period bars for stable values due to
        the double smoothing (TR/DM smoothing + DX smoothing).

        Features computed:
        - plus_di: Positive Directional Indicator (+DI)
        - minus_di: Negative Directional Indicator (-DI)
        - dx: Directional Movement Index
        - adx: Average Directional Index (Wilder's smoothing)
        - adxr: ADX Rating (smoothed ADX)
        - di_crossover: Binary signal for +DI > -DI
        - di_spread: Normalized (+DI - -DI)
        - adx_slope: Rate of change of ADX
        - adx_strength: Categorical strength (0=weak, 1=moderate, 2=strong)
        - adx_simple: Backward-compatible simple rolling ADX

        Args:
            df: DataFrame with 'high', 'low', 'close', 'tr' columns
            period: ADX period (default 14, industry standard)

        Returns:
            DataFrame with ADX indicators added
        """
        # Minimum data validation
        min_bars_required = 2 * period  # Need 28 bars for stable 14-period ADX
        if len(df) < min_bars_required:
            logger.warning(
                f"Insufficient data for stable ADX calculation. "
                f"Have {len(df)} bars, recommend at least {min_bars_required}. "
                f"ADX values may be unreliable."
            )

        # ============================================
        # Step 1: Calculate Directional Movement (+DM and -DM)
        # ============================================
        # +DM = current high - previous high (when positive and > -DM)
        # -DM = previous low - current low (when positive and > +DM)
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()  # previous low - current low

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Align DM with TR by setting first value to NaN
        # TR[0] is NaN (requires previous close), so DM[0] should also be NaN
        # This ensures smoothing windows for TR and DM start at the same index
        plus_dm.iloc[0] = np.nan
        minus_dm.iloc[0] = np.nan

        # ============================================
        # Step 2: Apply Wilder's smoothing to TR, +DM, -DM
        # ============================================
        tr_smoothed = self._wilder_smoothing(df['tr'], period)
        plus_dm_smoothed = self._wilder_smoothing(plus_dm, period)
        minus_dm_smoothed = self._wilder_smoothing(minus_dm, period)

        # ============================================
        # Step 3: Calculate +DI and -DI
        # ============================================
        # +DI = 100 * Smoothed +DM / Smoothed TR
        # -DI = 100 * Smoothed -DM / Smoothed TR
        plus_di = 100 * (plus_dm_smoothed / (tr_smoothed + 1e-10))
        minus_di = 100 * (minus_dm_smoothed / (tr_smoothed + 1e-10))

        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # ============================================
        # Step 4: Calculate DX (Directional Movement Index)
        # ============================================
        # DX = 100 * |+DI - -DI| / (+DI + -DI)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['dx'] = dx

        # ============================================
        # Step 5: Calculate ADX using Wilder's smoothing on DX
        # ============================================
        # First ADX = mean of first 'period' DX values
        # Subsequent ADX = ((Prior ADX * (period-1)) + Current DX) / period
        adx = pd.Series(index=df.index, dtype=float)

        # Find position of first valid DX value using numpy (handles non-unique indexes)
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

        df['adx'] = adx

        # ============================================
        # Step 6: Compute derived ADX features
        # ============================================

        # ADXR (ADX Rating) - Average of current ADX and ADX from 'period' bars ago
        df['adxr'] = (adx + adx.shift(period)) / 2

        # DI Crossover Signal (1 = bullish, 0 = bearish)
        df['di_crossover'] = (plus_di > minus_di).astype(int)

        # DI Spread - Normalized difference [-1, 1]
        df['di_spread'] = (plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # ADX Slope - 5-period rate of change
        df['adx_slope'] = adx.diff(5)

        # ADX Strength Classification (industry-standard thresholds)
        # 0 = weak (<20): No clear trend
        # 1 = moderate (20-40): Developing trend
        # 2 = strong (>40): Strong trend
        df['adx_strength'] = pd.cut(
            adx,
            bins=[-np.inf, 20, 40, np.inf],
            labels=[0, 1, 2]
        ).astype(float)

        # ============================================
        # Step 7: Backward-compatible simple ADX
        # ============================================
        # Uses simple rolling sums instead of Wilder's smoothing
        tr_14_simple = df['tr'].rolling(period).sum()
        plus_di_simple = 100 * (plus_dm.rolling(period).sum() / (tr_14_simple + 1e-10))
        minus_di_simple = 100 * (minus_dm.rolling(period).sum() / (tr_14_simple + 1e-10))
        dx_simple = 100 * np.abs(plus_di_simple - minus_di_simple) / (plus_di_simple + minus_di_simple + 1e-10)
        df['adx_simple'] = dx_simple.rolling(period).mean()

        return df

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition."""
        # Body and shadow sizes
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']

        # Bullish/Bearish
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        # Doji (small body)
        avg_body = df['body_size'].rolling(20).mean()
        df['is_doji'] = (df['body_size'] < avg_body * 0.1).astype(int)

        # Hammer (long lower shadow, small body at top)
        df['is_hammer'] = (
            (df['lower_shadow'] > df['body_size'] * 2) &
            (df['upper_shadow'] < df['body_size'] * 0.5)
        ).astype(int)

        # Engulfing pattern
        df['is_bullish_engulfing'] = (
            (df['is_bullish'] == 1) &
            (df['is_bullish'].shift(1) == 0) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)

        df['is_bearish_engulfing'] = (
            (df['is_bullish'] == 0) &
            (df['is_bullish'].shift(1) == 1) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour'] = ts.dt.hour
            df['day_of_week'] = ts.dt.dayofweek
            df['day_of_month'] = ts.dt.day
            df['month'] = ts.dt.month

            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index using simple rolling mean."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _compute_rsi_wilder(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Compute RSI using Wilder's smoothing (original design).

        Wilder's RSI uses exponential smoothing instead of simple moving average,
        making it more responsive to recent price changes while maintaining smoothness.
        This is the original RSI calculation as designed by J. Welles Wilder.

        Args:
            prices: Price series (typically close prices)
            period: RSI period (typically 14)

        Returns:
            RSI values using Wilder's smoothing
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Apply Wilder's smoothing to gains and losses
        avg_gain = self._wilder_smoothing(gain, period) / period
        avg_loss = self._wilder_smoothing(loss, period) / period

        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _wilder_smoothing(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Apply Wilder's smoothing technique (optimized implementation).

        Wilder's smoothing is used in ADX, ATR, and RSI calculations.
        Formula:
            First value = Sum of first 'period' values
            Subsequent = Prior value * (1 - 1/period) + Current value

        This is mathematically equivalent to an EMA with alpha = 1/period,
        but with different initialization (sum instead of first value).

        Performance: Uses numpy arrays for ~10x speedup over pure pandas.
        Note: The recursive nature of this smoothing limits full vectorization.

        Args:
            series: Input series to smooth
            period: Smoothing period (default 14)

        Returns:
            Smoothed series using Wilder's method
        """
        # Handle edge cases
        if series.isna().all():
            return pd.Series(index=series.index, dtype=float)

        # Find position of first valid value using numpy (handles non-unique indexes)
        not_na_mask = series.notna().to_numpy()
        if not not_na_mask.any():
            return pd.Series(index=series.index, dtype=float)
        start_pos = int(not_na_mask.argmax())

        if len(series) < start_pos + period:
            return pd.Series(index=series.index, dtype=float)

        # Convert to numpy for faster operations
        values = series.to_numpy(dtype=float, copy=True)
        n = len(values)
        result = np.full(n, np.nan, dtype=float)

        # First smoothed value = sum of first 'period' values
        first_sum_idx = start_pos + period - 1
        result[first_sum_idx] = np.nansum(values[start_pos:start_pos + period])

        # Decay factor for Wilder's smoothing: (period - 1) / period
        decay = (period - 1.0) / period

        # Apply Wilder's smoothing recursively
        # S[n] = S[n-1] * decay + X[n]
        # Using numpy indexing for speed (avoids pandas overhead)
        for i in range(first_sum_idx + 1, n):
            val = values[i]
            prior = result[i - 1]
            if not np.isnan(prior) and not np.isnan(val):
                result[i] = prior * decay + val
            # NaN propagation is automatic (result stays NaN)

        return pd.Series(result, index=series.index)


class DataPipeline:
    """
    Main data pipeline for fetching, processing, and streaming market data.
    Supports both batch and online (streaming) modes.
    """

    TIMEFRAME_MAP = {
        "1m": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
        "5m": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
        "15m": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
        "30m": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
        "1h": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
        "4h": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
        "1d": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
        "1w": mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 10080,
    }

    def __init__(self, config=None):
        self.config = config or {}
        self.feature_engineer = FeatureEngineer(config)
        self.scalers: Dict[str, RobustScaler] = {}
        self.data_cache: Dict[str, deque] = {}
        self.is_connected = False

    def connect(self) -> bool:
        """Connect to MetaTrader 5."""
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 not available. Using simulation mode.")
            return False

        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        self.is_connected = True
        logger.info("Connected to MetaTrader 5")
        return True

    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        if MT5_AVAILABLE and self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("Disconnected from MetaTrader 5")

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        n_bars: int = 10000,
        end_date: Optional[datetime] = None
    ) -> Optional[MarketData]:
        """Fetch historical OHLCV data."""
        if end_date is None:
            end_date = datetime.now()

        if MT5_AVAILABLE and self.is_connected:
            tf = self.TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from(symbol, tf, end_date, n_bars)

            if rates is None or len(rates) == 0:
                logger.error(f"Failed to fetch data for {symbol}")
                return None

            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')

        else:
            # Generate synthetic data for testing
            logger.info(f"Generating synthetic data for {symbol}")
            df = self._generate_synthetic_data(n_bars, end_date)

        return self._process_raw_data(df, symbol, timeframe)

    def _generate_synthetic_data(self, n_bars: int, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        np.random.seed(42)

        # Generate realistic price series
        returns = np.random.normal(0.0001, 0.01, n_bars)
        prices = 1.1000 * np.exp(np.cumsum(returns))

        # Generate OHLCV
        timestamps = pd.date_range(end=end_date, periods=n_bars, freq='h')

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
            'high': prices * (1 + np.random.uniform(0, 0.005, n_bars)),
            'low': prices * (1 - np.random.uniform(0, 0.005, n_bars)),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_bars)
        })

        # Ensure high > open, close and low < open, close
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        return df

    def _process_raw_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> MarketData:
        """Process raw OHLCV data into MarketData object."""
        # Compute features
        df = self.feature_engineer.compute_all_features(df)

        # Handle missing values
        df = df.ffill().bfill()

        # Get feature columns
        feature_cols = self.feature_engineer.feature_names
        features = df[feature_cols].values if feature_cols else None

        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=df['timestamp'].values,
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            volume=df['volume'].values,
            features=features,
            feature_names=feature_cols
        )

    def prepare_sequences(
        self,
        data: MarketData,
        sequence_length: int = 120,
        prediction_horizon: int = 12,
        target: str = 'returns'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.

        Returns:
            X: Input sequences (samples, seq_len, features)
            y: Target values (samples, prediction_horizon)
            timestamps: Timestamps for each sample
        """
        # Combine OHLCV with features
        ohlcv = np.column_stack([
            data.open, data.high, data.low, data.close, data.volume
        ])

        if data.features is not None:
            features = np.concatenate([ohlcv, data.features], axis=1)
        else:
            features = ohlcv

        # Normalize features
        scaler_key = f"{data.symbol}_{data.timeframe}"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = RobustScaler()
            features = self.scalers[scaler_key].fit_transform(features)
        else:
            features = self.scalers[scaler_key].transform(features)

        # Handle any remaining NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute target (future returns)
        if target == 'returns':
            close_prices = data.close
            targets = []
            for i in range(len(close_prices) - prediction_horizon):
                future_return = (close_prices[i + prediction_horizon] - close_prices[i]) / close_prices[i]
                targets.append(future_return)
            targets = np.array(targets)
        else:
            # Direction prediction
            close_prices = data.close
            targets = (np.roll(close_prices, -prediction_horizon) > close_prices).astype(float)
            targets = targets[:-prediction_horizon]

        # Create sequences
        X, y, timestamps = [], [], []
        n_samples = len(features) - sequence_length - prediction_horizon

        for i in range(n_samples):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length - 1])
            timestamps.append(data.timestamp[i + sequence_length - 1])

        return np.array(X), np.array(y), np.array(timestamps)

    def create_train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train, validation, and test sets."""
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        return {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end]),
            'test': (X[val_end:], y[val_end:])
        }

    def get_online_batch(
        self,
        symbol: str,
        timeframe: str = "1h",
        batch_size: int = 1
    ) -> Optional[MarketData]:
        """
        Get latest data for online learning.
        Fetches new data and returns it for model adaptation.
        """
        # Fetch latest bars
        data = self.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=batch_size + 200  # Extra for feature computation
        )

        if data is None:
            return None

        # Return only the latest batch_size bars
        return MarketData(
            symbol=data.symbol,
            timeframe=data.timeframe,
            timestamp=data.timestamp[-batch_size:],
            open=data.open[-batch_size:],
            high=data.high[-batch_size:],
            low=data.low[-batch_size:],
            close=data.close[-batch_size:],
            volume=data.volume[-batch_size:],
            features=data.features[-batch_size:] if data.features is not None else None,
            feature_names=data.feature_names
        )

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        symbol: str,
        timeframe: str
    ) -> np.ndarray:
        """Inverse transform normalized predictions."""
        scaler_key = f"{symbol}_{timeframe}"
        if scaler_key in self.scalers:
            # Only inverse transform if needed
            return predictions
        return predictions


class StreamingDataBuffer:
    """
    Buffer for streaming data in online learning mode.
    Maintains a rolling window of recent data.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data: deque = deque(maxlen=max_size)
        self.features: deque = deque(maxlen=max_size)

    def add(self, observation: Dict):
        """Add new observation to buffer."""
        self.data.append(observation)

    def get_latest(self, n: int) -> List[Dict]:
        """Get latest n observations."""
        return list(self.data)[-n:]

    def __len__(self) -> int:
        return len(self.data)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough data."""
        return len(self.data) >= min_size
