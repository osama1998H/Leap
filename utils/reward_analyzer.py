"""
Leap Trading System - Reward Pattern Analyzer

Utility for diagnosing reward signal issues in the PPO trading environment.
Run this to analyze how reward components behave during random/trained agent rollouts.

Usage:
    python -m utils.reward_analyzer --symbol EURUSD --episodes 5
    python -m utils.reward_analyzer --model-dir saved_models --symbol EURUSD
"""

import argparse
import os
import sys
import numpy as np
from typing import Dict, Optional, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_pipeline import DataPipeline
from core.trading_env import TradingEnvironment
from core.trading_types import EnvConfig, Action
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_reward_components(
    symbol: str = "EURUSD",
    timeframe: str = "1h",
    n_bars: int = 5000,
    episodes: int = 3,
    max_steps_per_episode: int = 500,
    model_dir: Optional[str] = None,
    env_config: Optional[EnvConfig] = None
) -> Dict:
    """
    Analyze reward component distributions during environment rollouts.

    Args:
        symbol: Trading symbol to use
        timeframe: Data timeframe
        n_bars: Number of bars to load
        episodes: Number of episodes to run
        max_steps_per_episode: Max steps per episode
        model_dir: Optional model directory to load trained agent
        env_config: Optional custom EnvConfig for testing different parameters

    Returns:
        Dictionary with analysis results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"REWARD COMPONENT ANALYSIS")
    logger.info(f"{'='*60}")

    # Load data
    logger.info(f"\nLoading {symbol} {timeframe} data ({n_bars} bars)...")
    pipeline = DataPipeline()
    df = pipeline.get_historical_data(symbol, timeframe, n_bars)

    if df is None or len(df) == 0:
        logger.error("Failed to load data")
        return {}

    logger.info(f"Loaded {len(df)} bars")

    # Engineer features
    df = pipeline.engineer_features(df)
    features = pipeline.get_feature_columns(df)

    # Create environment with custom or default config
    if env_config is None:
        env_config = EnvConfig()

    logger.info(f"\nEnvironment Config:")
    logger.info(f"  return_scale: {env_config.return_scale}")
    logger.info(f"  drawdown_penalty_scale: {env_config.drawdown_penalty_scale}")
    logger.info(f"  recovery_bonus_scale: {env_config.recovery_bonus_scale}")
    logger.info(f"  holding_cost: {env_config.holding_cost}")
    logger.info(f"  reward_clip: {env_config.reward_clip}")

    env = TradingEnvironment(
        data=df[['open', 'high', 'low', 'close', 'tick_volume']].values,
        features=features.values if features is not None else None,
        config=env_config
    )

    # Load agent if model_dir provided
    agent = None
    if model_dir:
        try:
            from models.ppo_agent import PPOAgent
            import json

            metadata_path = os.path.join(model_dir, 'model_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                agent_metadata = metadata.get('agent', {})
                if agent_metadata.get('exists'):
                    state_dim = agent_metadata['state_dim']
                    action_dim = agent_metadata['action_dim']
                    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
                    agent.load(os.path.join(model_dir, 'agent.pt'))
                    logger.info(f"\nLoaded trained agent from {model_dir}")
        except Exception as e:
            logger.warning(f"Could not load agent: {e}")
            agent = None

    # Run episodes and collect statistics
    all_rewards = []
    all_components = {
        'return_reward': [],
        'drawdown_penalty': [],
        'recovery_bonus': [],
        'holding_cost': [],
        'raw_reward': []
    }
    all_actions = []
    episode_stats = []

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while steps < max_steps_per_episode:
            # Select action (random or from agent)
            if agent:
                action, _, _ = agent.select_action(state)
            else:
                action = np.random.randint(0, 4)  # Random action

            state, reward, terminated, truncated, info = env.step(action)
            all_rewards.append(reward)
            all_actions.append(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        # Get episode statistics
        ep_stats = env.get_episode_stats()
        reward_comp = env.get_reward_component_stats()
        action_dist = env.get_action_distribution()

        episode_stats.append({
            'episode': ep + 1,
            'steps': steps,
            'total_reward': episode_reward,
            'final_equity': ep_stats.get('final_equity', 0),
            'total_return': ep_stats.get('total_return', 0),
            'max_drawdown': ep_stats.get('max_drawdown', 0),
            'total_trades': ep_stats.get('total_trades', 0),
            'win_rate': ep_stats.get('win_rate', 0)
        })

        # Collect component data
        if reward_comp:
            for key, stats in reward_comp.items():
                if isinstance(stats, dict):
                    all_components[key].extend([stats.get('mean', 0)] * steps)

        logger.info(f"\nEpisode {ep + 1}/{episodes}:")
        logger.info(f"  Steps: {steps}, Total Reward: {episode_reward:.4f}")
        logger.info(f"  Final Equity: ${ep_stats.get('final_equity', 0):.2f}")
        logger.info(f"  Total Return: {ep_stats.get('total_return', 0)*100:.2f}%")
        logger.info(f"  Max Drawdown: {ep_stats.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"  Trades: {ep_stats.get('total_trades', 0)}, Win Rate: {ep_stats.get('win_rate', 0)*100:.1f}%")

        if action_dist:
            logger.info(f"  Actions: HOLD={action_dist.get('HOLD', 0):.1f}%, "
                       f"BUY={action_dist.get('BUY', 0):.1f}%, "
                       f"SELL={action_dist.get('SELL', 0):.1f}%, "
                       f"CLOSE={action_dist.get('CLOSE', 0):.1f}%")

        if reward_comp:
            logger.info(f"  Reward Components (mean):")
            for comp, stats in reward_comp.items():
                if isinstance(stats, dict):
                    logger.info(f"    {comp}: mean={stats.get('mean', 0):.6f}, "
                               f"sum={stats.get('sum', 0):.4f}, "
                               f"nonzero={stats.get('nonzero_pct', 0):.1f}%")

    # Aggregate analysis
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE ANALYSIS")
    logger.info(f"{'='*60}")

    rewards_arr = np.array(all_rewards)
    logger.info(f"\nReward Distribution (n={len(rewards_arr)}):")
    logger.info(f"  Mean: {np.mean(rewards_arr):.6f}")
    logger.info(f"  Std:  {np.std(rewards_arr):.6f}")
    logger.info(f"  Min:  {np.min(rewards_arr):.6f}")
    logger.info(f"  Max:  {np.max(rewards_arr):.6f}")
    logger.info(f"  Median: {np.median(rewards_arr):.6f}")
    logger.info(f"  % Positive: {(rewards_arr > 0).sum() / len(rewards_arr) * 100:.1f}%")
    logger.info(f"  % Negative: {(rewards_arr < 0).sum() / len(rewards_arr) * 100:.1f}%")
    logger.info(f"  % Zero: {(rewards_arr == 0).sum() / len(rewards_arr) * 100:.1f}%")

    # Action distribution
    actions_arr = np.array(all_actions)
    logger.info(f"\nAction Distribution (n={len(actions_arr)}):")
    for action_type in [Action.HOLD, Action.BUY, Action.SELL, Action.CLOSE]:
        pct = (actions_arr == action_type).sum() / len(actions_arr) * 100
        logger.info(f"  {action_type.name}: {pct:.1f}%")

    # Diagnose issues
    logger.info(f"\n{'='*60}")
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info(f"{'='*60}")

    issues = []

    # Check for negative bias
    mean_reward = np.mean(rewards_arr)
    if mean_reward < -0.01:
        issues.append(f"NEGATIVE BIAS: Mean reward is {mean_reward:.4f}")

    # Check component balance using episode stats
    if episode_stats:
        avg_trades = np.mean([s['total_trades'] for s in episode_stats])
        avg_return = np.mean([s['total_return'] for s in episode_stats])

        if avg_trades < 1:
            issues.append(f"LOW TRADING: Only {avg_trades:.1f} trades per episode on average")

        if avg_return < -0.05:
            issues.append(f"LOSING MONEY: Average return is {avg_return*100:.2f}%")

    # Check for reward signal being too weak
    if np.std(rewards_arr) < 0.001:
        issues.append("WEAK SIGNAL: Reward std is very low, learning signal may be insufficient")

    # Check for clipping saturation
    clip_val = env_config.reward_clip
    clipped_pct = ((rewards_arr >= clip_val) | (rewards_arr <= -clip_val)).sum() / len(rewards_arr) * 100
    if clipped_pct > 10:
        issues.append(f"CLIPPING: {clipped_pct:.1f}% of rewards hit clip bounds")

    if issues:
        logger.info("\nPotential Issues Found:")
        for i, issue in enumerate(issues, 1):
            logger.info(f"  {i}. {issue}")
    else:
        logger.info("\nNo major issues detected.")

    # Recommendations
    logger.info(f"\n{'='*60}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*60}")

    if mean_reward < -0.01:
        logger.info("\n1. Consider reducing transaction costs or position sizing")
        logger.info("   - Current: commission=0.0001, spread=0.0002, slippage=0.0001")

    if np.mean([s['total_trades'] for s in episode_stats]) < 5:
        logger.info("\n2. Agent may be too passive. Consider:")
        logger.info("   - Reducing holding_cost (currently 0 by default)")
        logger.info("   - Adding small incentive for taking positions")

    return {
        'rewards': rewards_arr,
        'actions': actions_arr,
        'episode_stats': episode_stats,
        'mean_reward': mean_reward,
        'issues': issues
    }


def compare_configs(
    symbol: str = "EURUSD",
    timeframe: str = "1h",
    n_bars: int = 5000,
    episodes: int = 2
) -> None:
    """
    Compare reward behavior across different EnvConfig settings.
    """
    configs = {
        'default': EnvConfig(),
        'symmetric_low': EnvConfig(
            drawdown_penalty_scale=10.0,
            recovery_bonus_scale=10.0,
            holding_cost=0.0
        ),
        'symmetric_high': EnvConfig(
            drawdown_penalty_scale=30.0,
            recovery_bonus_scale=30.0,
            holding_cost=0.0
        ),
        'asymmetric_original': EnvConfig(
            drawdown_penalty_scale=25.0,
            recovery_bonus_scale=12.5,
            holding_cost=0.001
        ),
        'return_focused': EnvConfig(
            return_scale=100.0,
            drawdown_penalty_scale=10.0,
            recovery_bonus_scale=10.0,
            holding_cost=0.0
        )
    }

    logger.info(f"\n{'='*60}")
    logger.info("CONFIG COMPARISON")
    logger.info(f"{'='*60}")

    results = {}
    for name, config in configs.items():
        logger.info(f"\n\n>>> Testing config: {name}")
        res = analyze_reward_components(
            symbol=symbol,
            timeframe=timeframe,
            n_bars=n_bars,
            episodes=episodes,
            env_config=config
        )
        results[name] = res

    # Summary comparison
    logger.info(f"\n\n{'='*60}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"\n{'Config':<20} {'Mean Reward':>12} {'Std':>10} {'Issues':>8}")
    logger.info("-" * 52)

    for name, res in results.items():
        mean_r = res.get('mean_reward', 0)
        std_r = np.std(res.get('rewards', [0]))
        n_issues = len(res.get('issues', []))
        logger.info(f"{name:<20} {mean_r:>12.6f} {std_r:>10.6f} {n_issues:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze reward patterns in trading environment")
    parser.add_argument('--symbol', '-s', default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', '-t', default='1h', help='Timeframe')
    parser.add_argument('--bars', '-b', type=int, default=5000, help='Number of bars')
    parser.add_argument('--episodes', '-e', type=int, default=3, help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--model-dir', '-m', default=None, help='Model directory to load trained agent')
    parser.add_argument('--compare', action='store_true', help='Compare multiple config settings')

    args = parser.parse_args()

    if args.compare:
        compare_configs(
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_bars=args.bars,
            episodes=args.episodes
        )
    else:
        analyze_reward_components(
            symbol=args.symbol,
            timeframe=args.timeframe,
            n_bars=args.bars,
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            model_dir=args.model_dir
        )
