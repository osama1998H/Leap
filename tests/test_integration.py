"""
Leap Trading System - Integration Tests
Tests that all components work together correctly.
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_data_pipeline():
    """Test data pipeline and feature engineering."""
    print("\n" + "="*60)
    print("Testing Data Pipeline...")
    print("="*60)

    from core.data_pipeline import DataPipeline, FeatureEngineer

    # Create pipeline
    pipeline = DataPipeline()

    # Fetch synthetic data (no MT5 connection needed)
    market_data = pipeline.fetch_historical_data(
        symbol='EURUSD',
        timeframe='1h',
        n_bars=1000
    )

    assert market_data is not None, "Failed to fetch market data"
    assert len(market_data.close) == 1000, f"Expected 1000 bars, got {len(market_data.close)}"
    assert market_data.features is not None, "Features should be computed"
    assert len(market_data.feature_names) > 0, "Feature names should be populated"

    print(f"✓ Loaded {len(market_data.close)} bars")
    print(f"✓ Computed {len(market_data.feature_names)} features")

    # Test sequence preparation
    X, y, timestamps = pipeline.prepare_sequences(
        data=market_data,
        sequence_length=60,
        prediction_horizon=12
    )

    assert len(X) > 0, "Should create sequences"
    assert X.shape[1] == 60, "Sequence length should be 60"
    assert len(X) == len(y), "X and y should have same length"

    print(f"✓ Created {len(X)} sequences with shape {X.shape}")

    return pipeline, market_data


def test_transformer_predictor(input_dim: int):
    """Test Transformer predictor model."""
    print("\n" + "="*60)
    print("Testing Transformer Predictor...")
    print("="*60)

    from models.transformer import TransformerPredictor

    # Create predictor
    predictor = TransformerPredictor(
        input_dim=input_dim,
        config={
            'd_model': 64,
            'n_heads': 4,
            'n_encoder_layers': 2,
            'd_ff': 128,
            'dropout': 0.1,
            'max_seq_length': 60,
            'learning_rate': 1e-3
        },
        device='cpu'
    )

    print(f"✓ Created TransformerPredictor with input_dim={input_dim}")

    # Create dummy data
    X_train = np.random.randn(100, 60, input_dim).astype(np.float32)
    y_train = np.random.randn(100).astype(np.float32)
    X_val = np.random.randn(20, 60, input_dim).astype(np.float32)
    y_val = np.random.randn(20).astype(np.float32)

    # Quick training test (just 2 epochs)
    results = predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=2,
        batch_size=32,
        verbose=False
    )

    assert 'train_losses' in results, "Training should return losses"
    assert len(results['train_losses']) == 2, "Should have 2 epochs of losses"

    print(f"✓ Training completed with final loss: {results['train_losses'][-1]:.6f}")

    # Test prediction
    predictions = predictor.predict(X_val, return_uncertainty=True)

    assert 'prediction' in predictions, "Should return predictions"
    assert 'quantiles' in predictions, "Should return quantiles"
    assert 'uncertainty' in predictions, "Should return uncertainty"
    assert predictions['prediction'].shape == (20, 1), f"Wrong prediction shape: {predictions['prediction'].shape}"

    print(f"✓ Predictions shape: {predictions['prediction'].shape}")
    print(f"✓ Uncertainty range: [{predictions['uncertainty'].min():.4f}, {predictions['uncertainty'].max():.4f}]")

    # Test online update
    loss = predictor.online_update(X_val[:5], y_val[:5])
    assert isinstance(loss, float), "Online update should return loss"

    print(f"✓ Online update completed with loss: {loss:.6f}")

    return predictor


def test_ppo_agent(state_dim: int):
    """Test PPO reinforcement learning agent."""
    print("\n" + "="*60)
    print("Testing PPO Agent...")
    print("="*60)

    from models.ppo_agent import PPOAgent

    # Create agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=4,
        config={
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'clip_epsilon': 0.2,
            'n_steps': 128,
            'n_epochs': 4,
            'batch_size': 32,
            'hidden_sizes': [64, 64]
        },
        device='cpu'
    )

    print(f"✓ Created PPOAgent with state_dim={state_dim}")

    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action, log_prob, value = agent.select_action(state)

    assert 0 <= action < 4, f"Action should be in [0, 3], got {action}"
    assert isinstance(log_prob, float), "Log prob should be float"
    assert isinstance(value, float), "Value should be float"

    print(f"✓ Selected action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    # Test policy distribution
    probs = agent.get_policy_distribution(state)
    assert len(probs) == 4, f"Should have 4 action probabilities, got {len(probs)}"
    assert abs(sum(probs) - 1.0) < 1e-5, f"Probabilities should sum to 1, got {sum(probs)}"

    print(f"✓ Policy distribution: {probs}")

    return agent


def test_trading_environment():
    """Test trading environment."""
    print("\n" + "="*60)
    print("Testing Trading Environment...")
    print("="*60)

    from core.trading_env import TradingEnvironment

    # Create dummy OHLCV data
    n_steps = 500
    np.random.seed(42)
    prices = 1.1 * np.exp(np.cumsum(np.random.randn(n_steps) * 0.01))

    ohlcv = np.column_stack([
        prices * 0.999,  # open
        prices * 1.002,  # high
        prices * 0.998,  # low
        prices,          # close
        np.random.uniform(1000, 5000, n_steps)  # volume
    ])

    # Create environment
    env = TradingEnvironment(
        data=ohlcv,
        features=None,
        initial_balance=10000.0,
        window_size=60
    )

    print(f"✓ Created TradingEnvironment with {n_steps} steps")
    print(f"✓ Observation space: {env.observation_space.shape}")
    print(f"✓ Action space: {env.action_space.n}")

    # Test reset
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, f"Wrong observation shape: {obs.shape}"

    print(f"✓ Reset successful, initial balance: ${info['balance']:.2f}")

    # Test stepping through environment
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"✓ Ran {i+1} steps, total reward: {total_reward:.4f}")
    print(f"✓ Final equity: ${info['equity']:.2f}, trades: {info['total_trades']}")

    return env.observation_space.shape[0]


def test_risk_manager():
    """Test risk management module."""
    print("\n" + "="*60)
    print("Testing Risk Manager...")
    print("="*60)

    from core.risk_manager import RiskManager, DynamicRiskManager, RiskLimits, PositionSizing

    # Create risk manager
    risk_manager = DynamicRiskManager(
        initial_balance=10000.0,
        limits=RiskLimits(
            max_position_size_pct=0.02,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.15
        ),
        sizing=PositionSizing(
            method='percent',
            percent_risk=0.02
        )
    )

    print(f"✓ Created DynamicRiskManager with ${risk_manager.initial_balance:.2f}")

    # Test position sizing
    entry_price = 1.1000
    stop_loss = 1.0950

    size = risk_manager.calculate_position_size(entry_price, stop_loss)
    assert size > 0, "Position size should be positive"

    print(f"✓ Position size: {size:.4f} lots")

    # Test stop loss/take profit calculation
    sl = risk_manager.calculate_stop_loss(entry_price, 'long', atr=0.005, method='atr')
    tp = risk_manager.calculate_take_profit(entry_price, sl, 'long', risk_reward=2.0)

    assert sl < entry_price, "Stop loss should be below entry for long"
    assert tp > entry_price, "Take profit should be above entry for long"

    print(f"✓ Stop loss: {sl:.5f}, Take profit: {tp:.5f}")

    # Test trade approval
    should_trade, reason = risk_manager.should_take_trade(entry_price, sl, tp, 'long')
    print(f"✓ Trade {'approved' if should_trade else 'rejected'}: {reason}")

    # Test balance update
    risk_manager.update_balance(10500)
    print(f"✓ Updated balance: ${risk_manager.current_balance:.2f}")

    # Test trade recording
    risk_manager.record_trade(100, True)
    risk_manager.record_trade(-50, False)

    report = risk_manager.get_risk_report()
    print(f"✓ Win rate: {report['win_rate']:.1%}, Total trades: {report['total_trades']}")

    return risk_manager


def test_backtester():
    """Test backtesting framework."""
    print("\n" + "="*60)
    print("Testing Backtester...")
    print("="*60)

    import pandas as pd
    from evaluation.backtester import Backtester

    # Create dummy data
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range('2023-01-01', periods=n_bars, freq='H')
    prices = 1.1 * np.exp(np.cumsum(np.random.randn(n_bars) * 0.005))

    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.003,
        'low': prices * 0.997,
        'close': prices,
        'volume': np.random.uniform(1000, 5000, n_bars)
    }, index=dates)

    # Create backtester
    backtester = Backtester(
        initial_balance=10000.0,
        commission_rate=0.0001,
        spread_pips=1.5
    )

    print(f"✓ Created Backtester with ${backtester.initial_balance:.2f}")

    # Simple strategy
    def simple_strategy(data, **kwargs):
        if len(data) < 20:
            return {'action': 'hold'}

        # Simple moving average crossover
        close = data['close']
        sma_fast = close.rolling(5).mean().iloc[-1]
        sma_slow = close.rolling(20).mean().iloc[-1]

        if sma_fast > sma_slow * 1.001:
            return {'action': 'buy', 'stop_loss_pips': 50}
        elif sma_fast < sma_slow * 0.999:
            return {'action': 'sell', 'stop_loss_pips': 50}

        return {'action': 'hold'}

    # Run backtest
    result = backtester.run(df, simple_strategy)

    print(f"✓ Backtest completed:")
    print(f"  - Total return: {result.total_return:.2%}")
    print(f"  - Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"  - Max drawdown: {result.max_drawdown:.2%}")
    print(f"  - Total trades: {result.total_trades}")
    print(f"  - Win rate: {result.win_rate:.1%}")

    return result


def test_metrics_analyzer():
    """Test performance metrics analyzer."""
    print("\n" + "="*60)
    print("Testing Performance Analyzer...")
    print("="*60)

    from evaluation.metrics import MetricsCalculator, PerformanceAnalyzer

    # Create dummy equity curve
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0003  # Slight positive bias
    equity = 10000 * np.cumprod(1 + returns)

    # Test metrics calculator
    calc = MetricsCalculator()

    metrics = calc.calculate_all(equity)

    print(f"✓ Calculated metrics:")
    print(f"  - Total return: {metrics['total_return']:.2%}")
    print(f"  - Annualized return: {metrics['annualized_return']:.2%}")
    print(f"  - Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  - Max drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  - VaR (95%): {metrics['var_95']:.4f}")

    return metrics


def test_online_learning():
    """Test online learning system."""
    print("\n" + "="*60)
    print("Testing Online Learning Manager...")
    print("="*60)

    from training.online_learning import OnlineLearningManager, AdaptationConfig, MarketRegimeDetector

    # Test regime detector
    detector = MarketRegimeDetector(AdaptationConfig())

    for i in range(100):
        price = 1.1 + np.random.randn() * 0.01
        returns = np.random.randn() * 0.01
        detector.update(price, returns)

    regime = detector.detect_regime()
    print(f"✓ Detected market regime: {regime}")

    # Test would require actual models, so just verify import works
    print(f"✓ OnlineLearningManager available")

    return detector


def test_full_integration():
    """Test full system integration."""
    print("\n" + "="*60)
    print("Testing Full System Integration...")
    print("="*60)

    # 1. Create data pipeline and load data
    pipeline, market_data = test_data_pipeline()

    # 2. Test environment and get state dimension
    state_dim = test_trading_environment()

    # 3. Prepare sequences to get input dimension
    X, y, _ = pipeline.prepare_sequences(market_data, sequence_length=60, prediction_horizon=12)
    input_dim = X.shape[2]

    print(f"\n✓ Input dimension: {input_dim}")
    print(f"✓ State dimension: {state_dim}")

    # 4. Create and test predictor
    predictor = test_transformer_predictor(input_dim)

    # 5. Create and test agent
    agent = test_ppo_agent(state_dim)

    # 6. Test risk manager
    risk_manager = test_risk_manager()

    # 7. Test backtester
    backtest_result = test_backtester()

    # 8. Test metrics
    metrics = test_metrics_analyzer()

    # 9. Test online learning
    detector = test_online_learning()

    print("\n" + "="*60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("="*60)

    return True


if __name__ == '__main__':
    try:
        success = test_full_integration()
        if success:
            print("\n✓ All components are properly integrated and working together!")
            sys.exit(0)
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
