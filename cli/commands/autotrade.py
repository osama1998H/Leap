"""
Auto-trade command implementation.

This module handles the 'autotrade' CLI command for live/paper trading.
"""

import logging
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from config import SystemConfig
    from cli.system import LeapTradingSystem

logger = logging.getLogger(__name__)

# Optional imports for auto-trading (MT5 is Windows-only)
try:
    from core.mt5_broker import MT5BrokerGateway
    from core.auto_trader import AutoTrader
    from config.settings import AutoTraderConfig
    from training.online_learning import OnlineLearningManager
    AUTO_TRADER_AVAILABLE = True
except ImportError:
    AUTO_TRADER_AVAILABLE = False


def execute_autotrade(
    system: 'LeapTradingSystem',
    args: 'argparse.Namespace',
    config: 'SystemConfig',
    resolved: dict
) -> None:
    """
    Execute the autotrade command.

    Args:
        system: LeapTradingSystem instance
        args: Parsed command-line arguments
        config: System configuration
        resolved: Resolved configuration values
    """
    if not AUTO_TRADER_AVAILABLE:
        logger.error("Auto-trader not available. MT5 requires Windows.")
        sys.exit(1)

    symbols = resolved['symbols']
    timeframe = resolved['timeframe']

    logger.info("Starting Auto-Trader...")

    # Load models first
    system.load_models(args.model_dir)

    if system._predictor is None or system._agent is None:
        logger.error("Models not loaded. Please train models first.")
        sys.exit(1)

    # Initialize OnlineLearningManager if online learning is enabled
    if config.auto_trader.enable_online_learning:
        try:
            system._online_manager = OnlineLearningManager(
                predictor=system._predictor,
                agent=system._agent
            )
            logger.info("OnlineLearningManager initialized for adaptive learning")
        except Exception as e:
            logger.warning(f"Failed to initialize OnlineLearningManager: {e}. Online learning will be disabled.")
            system._online_manager = None

    # For autotrade, use config.auto_trader settings as defaults (not config.data)
    # CLI args still take priority: --symbol/--symbols > config.auto_trader > config.data
    if args.symbols:
        autotrade_symbols = args.symbols
    elif args.symbol:
        autotrade_symbols = [args.symbol]
    elif config.auto_trader.symbols:
        autotrade_symbols = config.auto_trader.symbols
    else:
        autotrade_symbols = symbols  # Fall back to resolved value

    autotrade_timeframe = args.timeframe or config.auto_trader.timeframe or timeframe

    # Create broker gateway
    broker = MT5BrokerGateway(
        login=config.auto_trader.mt5_login,
        password=config.auto_trader.mt5_password,
        server=config.auto_trader.mt5_server,
        magic_number=config.auto_trader.magic_number
    )

    # Connect data pipeline to MT5 for real market data
    # This is critical - without this, the system uses synthetic data!
    if not system.data_pipeline.connect():
        logger.warning("DataPipeline failed to connect to MT5. Will use broker connection for data.")
    else:
        logger.info("DataPipeline connected to MT5 for market data")

    # Save initial data snapshot if requested
    if getattr(args, 'save_data', False):
        from utils.data_saver import save_pipeline_data, generate_run_id
        # Load initial data snapshot for the primary symbol
        initial_data = system.load_data(
            symbol=autotrade_symbols[0],
            timeframe=autotrade_timeframe,
            n_bars=1000  # Recent data snapshot
        )
        if initial_data is not None:
            run_id = generate_run_id("autotrade", autotrade_symbols[0], autotrade_timeframe)
            data_source = "MT5" if getattr(system.data_pipeline, 'broker_gateway', None) else "synthetic"
            save_pipeline_data(
                run_id=run_id,
                market_data=initial_data,
                base_dir=config.get_path('data'),
                command="autotrade",
                n_bars=1000,
                data_source=data_source
            )
            logger.info(f"Initial data snapshot saved to {config.get_path('data')}/{run_id}/")

    # Create auto-trader config with model environment dimensions
    trader_config = AutoTraderConfig(
        symbols=autotrade_symbols,
        timeframe=autotrade_timeframe,
        risk_per_trade=config.auto_trader.risk_per_trade,
        max_positions=config.auto_trader.max_positions,
        default_sl_pips=config.auto_trader.default_sl_pips,
        default_tp_pips=config.auto_trader.default_tp_pips,
        paper_mode=args.paper,
        enable_online_learning=config.auto_trader.enable_online_learning,
        # Pass model environment dimensions for live trading compatibility
        model_window_size=system._model_env_config['window_size'],
        model_n_features=system._model_env_config['n_additional_features'],
        model_n_account_features=system._model_env_config['n_account_features']
    )

    # Handle online learning manager (may be None if not initialized)
    online_manager = system._online_manager
    if trader_config.enable_online_learning and online_manager is None:
        logger.warning("Online learning enabled but OnlineLearningManager not initialized. "
                     "Online learning will be disabled.")
        trader_config.enable_online_learning = False

    # Create auto-trader
    auto_trader = AutoTrader(
        broker=broker,
        predictor=system._predictor,
        agent=system._agent,
        risk_manager=system.risk_manager,
        online_manager=online_manager,
        data_pipeline=system.data_pipeline,
        config=trader_config
    )

    # Start trading
    mode = "PAPER" if args.paper else "LIVE"
    print(f"\n{'='*60}")
    print(f"  LEAP AUTO-TRADER - {mode} MODE")
    print(f"  Symbols: {', '.join(autotrade_symbols)}")
    print(f"  Risk per trade: {trader_config.risk_per_trade*100:.1f}%")
    print(f"  Max positions: {trader_config.max_positions}")
    print(f"{'='*60}\n")

    if not args.paper:
        print("WARNING: LIVE TRADING MODE - Real money at risk!")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Aborted.")
            sys.exit(0)

    try:
        auto_trader.start()
        print("Auto-trader running. Press Ctrl+C to stop.")

        # Keep main thread alive
        while auto_trader.state.value == 'running':
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping auto-trader...")
        auto_trader.stop()

    # Print final statistics
    stats = auto_trader.get_statistics()
    print("\n" + "="*60)
    print("  SESSION SUMMARY")
    print("="*60)
    if stats.get('session'):
        session = stats['session']
        print(f"  Duration: {session.get('duration', 'N/A')}")
        print(f"  Total Trades: {session.get('total_trades', 0)}")
        print(f"  Win Rate: {session.get('win_rate', 0)*100:.1f}%")
        print(f"  P&L: ${session.get('pnl', 0):.2f} ({session.get('pnl_percent', 0):.2f}%)")
        print(f"  Max Drawdown: {session.get('max_drawdown', 0)*100:.1f}%")
    print("="*60 + "\n")
