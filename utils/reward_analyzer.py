"""
Leap Trading System - Reward Pattern Analyzer

Utility for diagnosing reward signal issues in the PPO trading environment.
Run this to analyze how reward components behave during random/trained agent rollouts.

Usage:
    python -m utils.reward_analyzer --symbol EURUSD --episodes 5
    python -m utils.reward_analyzer --model-dir saved_models --symbol EURUSD
"""

import argparse
import json
import os
import sys
import numpy as np
from typing import Dict, List, Optional

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
    logger.info("\n" + "=" * 60)
    logger.info("REWARD COMPONENT ANALYSIS")
    logger.info("=" * 60)

    # Load data using correct DataPipeline API
    logger.info(f"\nLoading {symbol} {timeframe} data ({n_bars} bars)...")
    pipeline = DataPipeline()
    market_data = pipeline.fetch_historical_data(symbol, timeframe, n_bars)

    if market_data is None or len(market_data.close) == 0:
        logger.error("Failed to load data")
        return {}

    logger.info(f"Loaded {len(market_data.close)} bars")

    # Convert MarketData to arrays for TradingEnvironment
    data = np.column_stack([
        market_data.open,
        market_data.high,
        market_data.low,
        market_data.close,
        market_data.volume
    ])
    features = market_data.features  # Already numpy array or None

    # Create environment with custom or default config
    if env_config is None:
        env_config = EnvConfig()

    logger.info("\nEnvironment Config:")
    logger.info(f"  return_scale: {env_config.return_scale}")
    logger.info(f"  drawdown_penalty_scale: {env_config.drawdown_penalty_scale}")
    logger.info(f"  recovery_bonus_scale: {env_config.recovery_bonus_scale}")
    logger.info(f"  holding_cost: {env_config.holding_cost}")
    logger.info(f"  reward_clip: {env_config.reward_clip}")

    env = TradingEnvironment(
        data=data,
        features=features,
        config=env_config
    )

    # Load agent if model_dir provided
    agent = _load_agent(model_dir)

    # Run episodes and collect statistics
    all_rewards: List[float] = []
    all_components: Dict[str, List[float]] = {
        'return_reward': [],
        'drawdown_penalty': [],
        'recovery_bonus': [],
        'holding_cost': [],
        'raw_reward': []
    }
    all_actions: List[int] = []
    episode_stats: List[Dict] = []

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        steps = 0

        while steps < max_steps_per_episode:
            # Select action (random or from agent)
            if agent:
                action, _, _ = agent.select_action(state)
            else:
                action = np.random.randint(0, 4)  # Random action

            # Cast to int to handle numpy scalars/tensors from agent
            action = int(action)

            state, reward, terminated, truncated, _info = env.step(action)
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

        # Collect per-step component series (true distributions, not repeated means)
        rc_hist = env.history.get('reward_components', {})
        for key in all_components.keys():
            series = rc_hist.get(key, [])
            if series:
                all_components[key].extend(series)

        logger.info(f"\nEpisode {ep + 1}/{episodes}:")
        logger.info(f"  Steps: {steps}, Total Reward: {episode_reward:.4f}")
        logger.info(f"  Final Equity: ${ep_stats.get('final_equity', 0):.2f}")
        logger.info(f"  Total Return: {ep_stats.get('total_return', 0)*100:.2f}%")
        logger.info(f"  Max Drawdown: {ep_stats.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"  Trades: {ep_stats.get('total_trades', 0)}, "
                   f"Win Rate: {ep_stats.get('win_rate', 0)*100:.1f}%")

        if action_dist:
            logger.info(f"  Actions: HOLD={action_dist.get('HOLD', 0):.1f}%, "
                       f"BUY={action_dist.get('BUY', 0):.1f}%, "
                       f"SELL={action_dist.get('SELL', 0):.1f}%, "
                       f"CLOSE={action_dist.get('CLOSE', 0):.1f}%")

        if reward_comp:
            logger.info("  Reward Components (mean):")
            for comp, stats in reward_comp.items():
                if isinstance(stats, dict):
                    logger.info(f"    {comp}: mean={stats.get('mean', 0):.6f}, "
                               f"sum={stats.get('sum', 0):.4f}, "
                               f"nonzero={stats.get('nonzero_pct', 0):.1f}%")

    # Aggregate analysis
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE ANALYSIS")
    logger.info("=" * 60)

    rewards_arr = np.array(all_rewards)
    actions_arr = np.array(all_actions)

    # Guard against empty arrays (no steps collected)
    if rewards_arr.size == 0:
        logger.error("No rewards collected; cannot compute statistics.")
        logger.error("Check: episodes > 0, max_steps > 0, data window size sufficient")
        return {
            'rewards': rewards_arr,
            'actions': actions_arr,
            'components': all_components,
            'episode_stats': episode_stats,
            'mean_reward': float('nan'),
            'issues': ['NO_DATA: No steps were collected; check episodes/max_steps/data window size.']
        }

    logger.info(f"\nReward Distribution (n={len(rewards_arr)}):")
    logger.info(f"  Mean: {np.mean(rewards_arr):.6f}")
    logger.info(f"  Std:  {np.std(rewards_arr):.6f}")
    logger.info(f"  Min:  {np.min(rewards_arr):.6f}")
    logger.info(f"  Max:  {np.max(rewards_arr):.6f}")
    logger.info(f"  Median: {np.median(rewards_arr):.6f}")
    logger.info(f"  % Positive: {(rewards_arr > 0).sum() / len(rewards_arr) * 100:.1f}%")
    logger.info(f"  % Negative: {(rewards_arr < 0).sum() / len(rewards_arr) * 100:.1f}%")
    logger.info(f"  % Zero: {(rewards_arr == 0).sum() / len(rewards_arr) * 100:.1f}%")

    # Per-component analysis (using actual per-step values)
    logger.info("\nPer-Component Distributions:")
    for comp_name, comp_values in all_components.items():
        if comp_values:
            arr = np.array(comp_values)
            n_values = len(arr)
            logger.info(f"  {comp_name}:")
            logger.info(f"    mean={np.mean(arr):.6f}, std={np.std(arr):.6f}")
            logger.info(f"    min={np.min(arr):.6f}, max={np.max(arr):.6f}")
            logger.info(f"    nonzero={np.count_nonzero(arr) / max(1, n_values) * 100:.1f}%")

    # Action distribution
    logger.info(f"\nAction Distribution (n={len(actions_arr)}):")
    n_actions = max(1, len(actions_arr))  # Avoid division by zero
    for action_type in [Action.HOLD, Action.BUY, Action.SELL, Action.CLOSE]:
        pct = (actions_arr == action_type).sum() / n_actions * 100
        logger.info(f"  {action_type.name}: {pct:.1f}%")

    # Diagnose issues
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)

    issues = _diagnose_issues(rewards_arr, episode_stats, env_config)

    if issues:
        logger.info("\nPotential Issues Found:")
        for i, issue in enumerate(issues, 1):
            logger.info(f"  {i}. {issue}")
    else:
        logger.info("\nNo major issues detected.")

    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)

    mean_reward = float(np.mean(rewards_arr))
    if mean_reward < -0.01:
        logger.info("\n1. Consider reducing transaction costs or position sizing")
        logger.info("   - Current: commission=0.0001, spread=0.0002, slippage=0.0001")

    if episode_stats and np.mean([s['total_trades'] for s in episode_stats]) < 5:
        logger.info("\n2. Agent may be too passive. Consider:")
        logger.info("   - Reducing holding_cost (currently 0 by default)")
        logger.info("   - Adding small incentive for taking positions")

    return {
        'rewards': rewards_arr,
        'actions': actions_arr,
        'components': all_components,
        'episode_stats': episode_stats,
        'mean_reward': mean_reward,
        'issues': issues
    }


def _load_agent(model_dir: Optional[str]):
    """Load trained agent from model directory with proper error handling."""
    if not model_dir:
        return None

    try:
        from models.ppo_agent import PPOAgent

        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if not os.path.exists(metadata_path):
            logger.warning(f"No model metadata found at {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        agent_metadata = metadata.get('agent', {})
        if not agent_metadata.get('exists'):
            logger.warning("Agent metadata indicates no agent exists")
            return None

        state_dim = agent_metadata['state_dim']
        action_dim = agent_metadata['action_dim']
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(os.path.join(model_dir, 'agent.pt'))
        logger.info(f"\nLoaded trained agent from {model_dir}")
        return agent

    except FileNotFoundError as e:
        logger.warning(f"Agent file not found: {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid model metadata JSON: {e}")
    except KeyError as e:
        logger.warning(f"Missing required metadata field: {e}")
    except (OSError, ValueError) as e:
        logger.exception("Error loading agent: %s", e)

    return None


def _diagnose_issues(
    rewards_arr: np.ndarray,
    episode_stats: List[Dict],
    env_config: EnvConfig
) -> List[str]:
    """Diagnose potential issues with reward signal."""
    issues = []

    # Guard against empty array
    if rewards_arr.size == 0:
        return ['NO_DATA: rewards_arr is empty']

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
    n_rewards = max(1, len(rewards_arr))  # Avoid division by zero
    clipped_pct = ((rewards_arr >= clip_val) | (rewards_arr <= -clip_val)).sum() / n_rewards * 100
    if clipped_pct > 10:
        issues.append(f"CLIPPING: {clipped_pct:.1f}% of rewards hit clip bounds")

    return issues


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

    logger.info("\n" + "=" * 60)
    logger.info("CONFIG COMPARISON")
    logger.info("=" * 60)

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
    logger.info("\n\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\n{'Config':<20} {'Mean Reward':>12} {'Std':>10} {'Issues':>8}")
    logger.info("-" * 52)

    for name, res in results.items():
        mean_r = res.get('mean_reward', 0)
        rewards = res.get('rewards', np.array([0]))
        std_r = np.std(rewards) if len(rewards) > 0 else 0
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
