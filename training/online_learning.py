"""
Leap Trading System - Online Learning and Adaptation
Implements continuous learning and model adaptation for real-time trading.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import logging
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking."""
    timestamp: datetime
    prediction_error: float
    trading_pnl: float
    sharpe_ratio: float
    win_rate: float
    market_regime: str


@dataclass
class AdaptationConfig:
    """Configuration for online adaptation."""
    # Performance thresholds
    error_threshold: float = 0.05  # Trigger adaptation if error > 5%
    drawdown_threshold: float = 0.1  # Trigger adaptation if drawdown > 10%
    performance_window: int = 100  # Number of samples for performance calculation

    # Adaptation parameters
    adaptation_frequency: int = 100  # Adapt every N steps
    min_samples_for_adaptation: int = 50
    learning_rate_decay: float = 0.95
    max_adaptations_per_day: int = 10

    # Regime detection
    regime_detection_enabled: bool = True
    regime_lookback: int = 50
    volatility_threshold_high: float = 0.02
    volatility_threshold_low: float = 0.005


class MarketRegimeDetector:
    """
    Detect market regimes for adaptive model selection.
    Regimes: trending_up, trending_down, ranging, high_volatility
    """

    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.price_history = deque(maxlen=config.regime_lookback)
        self.volatility_history = deque(maxlen=config.regime_lookback)

    def update(self, price: float, returns: float):
        """Update with new price data."""
        self.price_history.append(price)
        self.volatility_history.append(abs(returns))

    def detect_regime(self) -> str:
        """Detect current market regime."""
        if len(self.price_history) < self.config.regime_lookback // 2:
            return "unknown"

        prices = np.array(self.price_history)
        volatility = np.mean(self.volatility_history)

        # Calculate trend
        trend = (prices[-1] - prices[0]) / prices[0]
        sma_short = np.mean(prices[-10:])
        sma_long = np.mean(prices)

        # High volatility regime
        if volatility > self.config.volatility_threshold_high:
            return "high_volatility"

        # Trending regimes
        if trend > 0.02 and sma_short > sma_long:
            return "trending_up"
        elif trend < -0.02 and sma_short < sma_long:
            return "trending_down"

        # Low volatility / ranging
        if volatility < self.config.volatility_threshold_low:
            return "low_volatility"

        return "ranging"


class OnlineLearningManager:
    """
    Manages online learning and model adaptation.

    Features:
    - Continuous performance monitoring
    - Automatic model adaptation triggers
    - Market regime detection
    - Catastrophic forgetting prevention
    """

    def __init__(
        self,
        predictor,  # TransformerPredictor
        agent,      # PPOAgent
        config: Optional[AdaptationConfig] = None
    ):
        self.predictor = predictor
        self.agent = agent
        self.config = config or AdaptationConfig()

        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.prediction_errors: deque = deque(maxlen=self.config.performance_window)
        self.trading_returns: deque = deque(maxlen=self.config.performance_window)

        # Adaptation tracking
        self.adaptations_today = 0
        self.last_adaptation_date = None
        self.total_adaptations = 0

        # Market regime
        self.regime_detector = MarketRegimeDetector(self.config)
        self.current_regime = "unknown"

        # Experience buffer for adaptation
        self.prediction_buffer: deque = deque(maxlen=10000)
        self.experience_buffer: deque = deque(maxlen=50000)

        # Step counter
        self.step_count = 0

        # Callbacks
        self.on_adaptation_callbacks: List[Callable] = []

    def step(
        self,
        market_data: Dict,
        prediction: float,
        actual: float,
        trading_action: int,
        trading_reward: float
    ) -> Dict:
        """
        Process one step of online learning.

        Args:
            market_data: Current market data
            prediction: Model prediction
            actual: Actual outcome
            trading_action: Action taken by agent
            trading_reward: Reward received

        Returns:
            Dictionary with adaptation info
        """
        self.step_count += 1

        # Update regime detector
        if 'close' in market_data and 'returns' in market_data:
            self.regime_detector.update(market_data['close'], market_data['returns'])
            self.current_regime = self.regime_detector.detect_regime()

        # Calculate prediction error
        prediction_error = abs(prediction - actual)
        self.prediction_errors.append(prediction_error)

        # Track trading performance
        self.trading_returns.append(trading_reward)

        # Store in buffers
        self.prediction_buffer.append({
            'features': market_data.get('features'),
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now()
        })

        self.experience_buffer.append({
            'state': market_data.get('state'),
            'action': trading_action,
            'reward': trading_reward,
            'timestamp': datetime.now()
        })

        # Check if adaptation is needed
        adaptation_result = None
        if self._should_adapt():
            adaptation_result = self._perform_adaptation()

        # Calculate current performance
        current_performance = self._calculate_performance()

        # Store metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            prediction_error=np.mean(self.prediction_errors) if self.prediction_errors else 0,
            trading_pnl=sum(self.trading_returns) if self.trading_returns else 0,
            sharpe_ratio=current_performance.get('sharpe_ratio', 0),
            win_rate=current_performance.get('win_rate', 0),
            market_regime=self.current_regime
        )
        self.performance_history.append(metrics)

        return {
            'step': self.step_count,
            'regime': self.current_regime,
            'adaptation_triggered': adaptation_result is not None,
            'adaptation_result': adaptation_result,
            'current_performance': current_performance
        }

    def _should_adapt(self) -> bool:
        """Determine if model adaptation should be triggered."""
        # Check step frequency
        if self.step_count % self.config.adaptation_frequency != 0:
            return False

        # Check minimum samples
        if len(self.prediction_errors) < self.config.min_samples_for_adaptation:
            return False

        # Reset daily counter
        today = datetime.now().date()
        if self.last_adaptation_date != today:
            self.adaptations_today = 0
            self.last_adaptation_date = today

        # Check daily limit
        if self.adaptations_today >= self.config.max_adaptations_per_day:
            return False

        # Check performance thresholds
        avg_error = np.mean(self.prediction_errors)
        if avg_error > self.config.error_threshold:
            logger.info(f"Adaptation triggered: High prediction error ({avg_error:.4f})")
            return True

        # Check drawdown
        if len(self.trading_returns) > 10:
            cumulative = np.cumsum(self.trading_returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / (peak + 1e-10)
            max_drawdown = np.max(drawdown)

            if max_drawdown > self.config.drawdown_threshold:
                logger.info(f"Adaptation triggered: High drawdown ({max_drawdown:.4f})")
                return True

        return False

    def _perform_adaptation(self) -> Dict:
        """Perform model adaptation."""
        logger.info("Performing online adaptation...")

        results = {
            'predictor_updated': False,
            'agent_updated': False,
            'regime': self.current_regime
        }

        # Adapt predictor
        if len(self.prediction_buffer) >= self.config.min_samples_for_adaptation:
            predictor_result = self._adapt_predictor()
            results['predictor_updated'] = predictor_result is not None
            results['predictor_loss'] = predictor_result

        # Adapt agent
        if len(self.experience_buffer) >= self.config.min_samples_for_adaptation:
            agent_result = self._adapt_agent()
            results['agent_updated'] = agent_result is not None
            results['agent_loss'] = agent_result

        # Update counters
        self.adaptations_today += 1
        self.total_adaptations += 1

        # Trigger callbacks
        for callback in self.on_adaptation_callbacks:
            callback(results)

        logger.info(f"Adaptation complete. Total adaptations: {self.total_adaptations}")

        return results

    def _adapt_predictor(self) -> Optional[float]:
        """Adapt the prediction model."""
        # Get recent samples with actual outcomes
        samples = list(self.prediction_buffer)[-self.config.min_samples_for_adaptation:]

        # Filter samples with valid features
        valid_samples = [s for s in samples if s['features'] is not None]

        if len(valid_samples) < 10:
            return None

        # Prepare data
        X = np.array([s['features'] for s in valid_samples])
        y = np.array([s['actual'] for s in valid_samples])

        # Reshape if needed
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        # Perform online update
        loss = self.predictor.online_update(X, y)

        return loss

    def _adapt_agent(self) -> Optional[float]:
        """Adapt the RL agent."""
        result = self.agent.online_update(
            n_samples=min(256, len(self.experience_buffer)),
            n_epochs=3
        )

        return result.get('online_loss') if result else None

    def _calculate_performance(self) -> Dict:
        """Calculate current performance metrics."""
        if len(self.trading_returns) < 2:
            return {}

        returns = np.array(self.trading_returns)

        # Sharpe ratio (simplified)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-10
        sharpe = mean_return / std_return * np.sqrt(252)

        # Win rate
        wins = np.sum(returns > 0)
        win_rate = wins / len(returns)

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0])) + 1e-10
        profit_factor = gross_profit / gross_loss

        return {
            'sharpe_ratio': float(sharpe),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'mean_return': float(mean_return),
            'volatility': float(std_return)
        }

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.performance_history:
            return {}

        recent = list(self.performance_history)[-100:]

        return {
            'total_steps': self.step_count,
            'total_adaptations': self.total_adaptations,
            'current_regime': self.current_regime,
            'avg_prediction_error': float(np.mean([m.prediction_error for m in recent])),
            'total_pnl': float(sum([m.trading_pnl for m in recent])),
            'avg_sharpe': float(np.mean([m.sharpe_ratio for m in recent])),
            'avg_win_rate': float(np.mean([m.win_rate for m in recent])),
            'regime_distribution': self._get_regime_distribution()
        }

    def _get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of market regimes."""
        if not self.performance_history:
            return {}

        regimes = [m.market_regime for m in self.performance_history]
        unique, counts = np.unique(regimes, return_counts=True)

        return {r: c / len(regimes) for r, c in zip(unique, counts)}

    def register_adaptation_callback(self, callback: Callable):
        """Register callback for adaptation events."""
        self.on_adaptation_callbacks.append(callback)


class AdaptiveTrainer:
    """
    High-level trainer with automatic adaptation capabilities.
    Combines prediction model and RL agent training with online adaptation.
    """

    def __init__(
        self,
        predictor,
        agent,
        env,
        config: Optional[Dict] = None
    ):
        self.predictor = predictor
        self.agent = agent
        self.env = env
        self.config = config or {}

        # Online learning manager
        adaptation_config = AdaptationConfig(
            **self.config.get('adaptation', {})
        )
        self.online_manager = OnlineLearningManager(
            predictor, agent, adaptation_config
        )

        # Training state
        self.is_training = False
        self.training_thread = None

    def train_offline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        predictor_epochs: int = 100,
        agent_timesteps: int = 100000
    ) -> Dict:
        """
        Perform offline training of both models.
        """
        results = {}

        # Train predictor
        logger.info("Training prediction model...")
        predictor_results = self.predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=predictor_epochs
        )
        results['predictor'] = predictor_results

        # Train agent
        logger.info("Training RL agent...")
        agent_results = self.agent.train_on_env(
            self.env,
            total_timesteps=agent_timesteps
        )
        results['agent'] = agent_results

        return results

    def start_online_training(
        self,
        data_stream,
        callback: Optional[Callable] = None
    ):
        """
        Start online training loop in background.
        """
        self.is_training = True

        def training_loop():
            while self.is_training:
                try:
                    # Get new data
                    market_data = data_stream.get_latest()

                    if market_data is None:
                        time.sleep(1)
                        continue

                    # Make prediction
                    features = market_data.get('features')
                    if features is not None:
                        prediction = self.predictor.predict(
                            features.reshape(1, -1)
                        )['prediction'][0, 0]
                    else:
                        prediction = 0.0

                    # Get agent action
                    state = market_data.get('state')
                    if state is not None:
                        action, _, _ = self.agent.select_action(state)
                    else:
                        action = 0

                    # Wait for actual outcome
                    actual = market_data.get('actual', prediction)
                    reward = market_data.get('reward', 0.0)

                    # Online learning step
                    result = self.online_manager.step(
                        market_data=market_data,
                        prediction=prediction,
                        actual=actual,
                        trading_action=action,
                        trading_reward=reward
                    )

                    if callback:
                        callback(result)

                except Exception as e:
                    # Use exception() to include full stack trace for debugging
                    logger.exception(f"Online training error: {e}")
                    time.sleep(5)

        self.training_thread = threading.Thread(target=training_loop)
        # daemon=False allows graceful shutdown via stop_online_training()
        # to prevent model corruption during updates
        self.training_thread.daemon = False
        self.training_thread.start()

        logger.info("Online training started")

    def stop_online_training(self):
        """Stop online training loop."""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
        logger.info("Online training stopped")

    def save_models(self, directory: str):
        """Save all models."""
        import os
        os.makedirs(directory, exist_ok=True)

        self.predictor.save(os.path.join(directory, 'predictor.pt'))
        self.agent.save(os.path.join(directory, 'agent.pt'))

        logger.info(f"Models saved to {directory}")

    def load_models(self, directory: str):
        """Load all models."""
        import os

        self.predictor.load(os.path.join(directory, 'predictor.pt'))
        self.agent.load(os.path.join(directory, 'agent.pt'))

        logger.info(f"Models loaded from {directory}")
