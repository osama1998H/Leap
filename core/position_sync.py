"""
Leap Trading System - Position Synchronizer
Keeps internal state in sync with MT5 broker positions.
"""

import logging
from typing import Optional, List, Dict, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock

from core.mt5_broker import MT5BrokerGateway, MT5Position

logger = logging.getLogger(__name__)


class PositionEvent(Enum):
    """Position-related events."""
    OPENED = "opened"
    CLOSED = "closed"
    MODIFIED = "modified"
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"
    EXTERNAL_OPEN = "external_open"
    EXTERNAL_CLOSE = "external_close"


@dataclass
class PositionChange:
    """Represents a change in position state."""
    event: PositionEvent
    position: Optional[MT5Position]
    ticket: int
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncedPosition:
    """Internal representation of a synced position."""
    mt5_position: MT5Position
    opened_by_us: bool = True
    last_sync: datetime = field(default_factory=datetime.now)
    original_sl: float = 0.0
    original_tp: float = 0.0


class PositionSynchronizer:
    """
    Synchronizes internal state with MT5 broker positions.

    Responsibilities:
    - Track all open positions
    - Detect externally opened/closed positions
    - Detect SL/TP hits
    - Notify callbacks of position events
    - Maintain position state consistency
    """

    def __init__(
        self,
        broker: MT5BrokerGateway,
        magic_number: Optional[int] = None,
        sync_interval: float = 1.0
    ):
        """
        Initialize position synchronizer.

        Args:
            broker: MT5 broker gateway
            magic_number: Filter positions by magic number (None for all)
            sync_interval: Minimum seconds between syncs
        """
        self.broker = broker
        self.magic_number = magic_number
        self.sync_interval = sync_interval

        # Position tracking
        self._positions: Dict[int, SyncedPosition] = {}
        self._known_tickets: Set[int] = set()
        self._lock = Lock()

        # Event callbacks
        self._callbacks: Dict[PositionEvent, List[Callable]] = {
            event: [] for event in PositionEvent
        }

        # Sync state
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0

        # Position change history
        self._change_history: List[PositionChange] = []
        self._max_history = 1000

    def register_callback(self, event: PositionEvent, callback: Callable):
        """
        Register a callback for position events.

        Args:
            event: Event type to listen for
            callback: Function to call (receives PositionChange)
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def unregister_callback(self, event: PositionEvent, callback: Callable):
        """Unregister a callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def sync(self, force: bool = False) -> List[PositionChange]:
        """
        Synchronize positions with broker.

        Args:
            force: Force sync even if within sync interval

        Returns:
            List of position changes detected
        """
        if not force and self._last_sync:
            elapsed = (datetime.now() - self._last_sync).total_seconds()
            if elapsed < self.sync_interval:
                return []

        if not self.broker.is_connected:
            logger.warning("Cannot sync: broker not connected")
            return []

        changes = []

        with self._lock:
            # Get current positions from broker
            broker_positions = self.broker.get_positions()

            # Filter by magic number if specified
            if self.magic_number is not None:
                broker_positions = [
                    p for p in broker_positions
                    if p.magic == self.magic_number
                ]

            # Create lookup by ticket
            broker_tickets = {p.ticket: p for p in broker_positions}

            # Detect closed positions (were in our list, not in broker)
            closed_tickets = set(self._positions.keys()) - set(broker_tickets.keys())
            for ticket in closed_tickets:
                synced = self._positions.pop(ticket)
                self._known_tickets.discard(ticket)

                # Determine close reason
                if synced.mt5_position.sl > 0 and self._was_sl_hit(synced):
                    event = PositionEvent.SL_HIT
                elif synced.mt5_position.tp > 0 and self._was_tp_hit(synced):
                    event = PositionEvent.TP_HIT
                elif synced.opened_by_us:
                    event = PositionEvent.CLOSED
                else:
                    event = PositionEvent.EXTERNAL_CLOSE

                change = PositionChange(
                    event=event,
                    position=synced.mt5_position,
                    ticket=ticket,
                    symbol=synced.mt5_position.symbol,
                    details={
                        'profit': synced.mt5_position.profit,
                        'opened_by_us': synced.opened_by_us
                    }
                )
                changes.append(change)
                self._record_change(change)
                logger.info(f"Position closed: {ticket} ({event.value})")

            # Detect new positions (in broker, not in our list)
            new_tickets = set(broker_tickets.keys()) - set(self._positions.keys())
            for ticket in new_tickets:
                pos = broker_tickets[ticket]

                # Determine if opened by us (based on magic number)
                opened_by_us = (
                    self.magic_number is None or
                    pos.magic == self.magic_number
                )

                # Check if truly new position BEFORE adding to known tickets
                is_new_position = ticket not in self._known_tickets

                synced = SyncedPosition(
                    mt5_position=pos,
                    opened_by_us=opened_by_us,
                    original_sl=pos.sl,
                    original_tp=pos.tp
                )
                self._positions[ticket] = synced
                self._known_tickets.add(ticket)

                if is_new_position:
                    # Truly new position
                    event = PositionEvent.OPENED if opened_by_us else PositionEvent.EXTERNAL_OPEN
                    change = PositionChange(
                        event=event,
                        position=pos,
                        ticket=ticket,
                        symbol=pos.symbol,
                        details={'opened_by_us': opened_by_us}
                    )
                    changes.append(change)
                    self._record_change(change)
                    logger.info(f"Position opened: {ticket} ({event.value})")

            # Update existing positions and check for modifications
            for ticket, pos in broker_tickets.items():
                if ticket in self._positions:
                    synced = self._positions[ticket]
                    old_pos = synced.mt5_position

                    # Check for modifications
                    if old_pos.sl != pos.sl or old_pos.tp != pos.tp:
                        change = PositionChange(
                            event=PositionEvent.MODIFIED,
                            position=pos,
                            ticket=ticket,
                            symbol=pos.symbol,
                            details={
                                'old_sl': old_pos.sl,
                                'new_sl': pos.sl,
                                'old_tp': old_pos.tp,
                                'new_tp': pos.tp
                            }
                        )
                        changes.append(change)
                        self._record_change(change)
                        logger.debug(f"Position modified: {ticket}")

                    # Update synced position
                    synced.mt5_position = pos
                    synced.last_sync = datetime.now()

            self._last_sync = datetime.now()
            self._sync_count += 1

        # Fire callbacks
        for change in changes:
            self._fire_callbacks(change)

        return changes

    def get_positions(self, symbol: Optional[str] = None) -> List[MT5Position]:
        """
        Get current positions.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of positions
        """
        with self._lock:
            positions = [sp.mt5_position for sp in self._positions.values()]

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        return positions

    def get_position(self, ticket: int) -> Optional[MT5Position]:
        """Get position by ticket."""
        with self._lock:
            synced = self._positions.get(ticket)
            return synced.mt5_position if synced else None

    def get_positions_count(self, symbol: Optional[str] = None) -> int:
        """Get count of positions."""
        return len(self.get_positions(symbol))

    def has_position(self, symbol: str, direction: Optional[str] = None) -> bool:
        """
        Check if we have a position.

        Args:
            symbol: Symbol to check
            direction: 'long', 'short', or None for any

        Returns:
            True if position exists
        """
        positions = self.get_positions(symbol)

        if direction == 'long':
            return any(p.is_long for p in positions)
        elif direction == 'short':
            return any(p.is_short for p in positions)
        else:
            return len(positions) > 0

    def get_unrealized_pnl(self, symbol: Optional[str] = None) -> float:
        """Get total unrealized PnL."""
        positions = self.get_positions(symbol)
        return sum(p.unrealized_pnl for p in positions)

    def get_total_volume(self, symbol: Optional[str] = None) -> float:
        """Get total position volume."""
        positions = self.get_positions(symbol)
        return sum(p.volume for p in positions)

    def mark_position_as_ours(self, ticket: int):
        """Mark a position as opened by us."""
        with self._lock:
            if ticket in self._positions:
                self._positions[ticket].opened_by_us = True
            self._known_tickets.add(ticket)

    def get_change_history(self, n: int = 100) -> List[PositionChange]:
        """Get recent position changes."""
        return self._change_history[-n:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        with self._lock:
            positions = list(self._positions.values())

        return {
            'total_positions': len(positions),
            'positions_by_us': sum(1 for p in positions if p.opened_by_us),
            'external_positions': sum(1 for p in positions if not p.opened_by_us),
            'sync_count': self._sync_count,
            'last_sync': self._last_sync,
            'total_unrealized_pnl': sum(p.mt5_position.unrealized_pnl for p in positions),
            'total_volume': sum(p.mt5_position.volume for p in positions)
        }

    def reset(self):
        """Reset synchronizer state."""
        with self._lock:
            self._positions.clear()
            self._known_tickets.clear()
            self._change_history.clear()
            self._last_sync = None
            self._sync_count = 0

    def _was_sl_hit(self, synced: SyncedPosition) -> bool:
        """Check if position was likely closed by SL."""
        pos = synced.mt5_position
        if pos.sl == 0:
            return False

        if pos.is_long:
            return pos.price_current <= pos.sl
        else:
            return pos.price_current >= pos.sl

    def _was_tp_hit(self, synced: SyncedPosition) -> bool:
        """Check if position was likely closed by TP."""
        pos = synced.mt5_position
        if pos.tp == 0:
            return False

        if pos.is_long:
            return pos.price_current >= pos.tp
        else:
            return pos.price_current <= pos.tp

    def _fire_callbacks(self, change: PositionChange):
        """Fire registered callbacks for an event."""
        callbacks = self._callbacks.get(change.event, [])
        for callback in callbacks:
            try:
                callback(change)
            except Exception:
                logger.exception(f"Callback error for {change.event}")

    def _record_change(self, change: PositionChange):
        """Record a change in history."""
        self._change_history.append(change)
        if len(self._change_history) > self._max_history:
            self._change_history = self._change_history[-self._max_history:]


class PositionTracker:
    """
    Lightweight position tracker for internal use.
    Tracks positions without MT5 dependency (for backtesting compatibility).
    """

    @dataclass
    class TrackedPosition:
        ticket: int
        symbol: str
        type: str  # 'long' or 'short'
        volume: float
        entry_price: float
        sl: float = 0.0
        tp: float = 0.0
        open_time: datetime = field(default_factory=datetime.now)
        unrealized_pnl: float = 0.0

        @property
        def is_long(self) -> bool:
            return self.type == 'long'

        @property
        def is_short(self) -> bool:
            return self.type == 'short'

    def __init__(self):
        self._positions: Dict[int, 'PositionTracker.TrackedPosition'] = {}
        self._next_ticket = 1

    def open_position(
        self,
        symbol: str,
        position_type: str,
        volume: float,
        entry_price: float,
        sl: float = 0.0,
        tp: float = 0.0
    ) -> int:
        """Open a new tracked position. Returns ticket."""
        ticket = self._next_ticket
        self._next_ticket += 1

        self._positions[ticket] = self.TrackedPosition(
            ticket=ticket,
            symbol=symbol,
            type=position_type,
            volume=volume,
            entry_price=entry_price,
            sl=sl,
            tp=tp
        )

        return ticket

    def close_position(self, ticket: int) -> Optional['PositionTracker.TrackedPosition']:
        """Close a position. Returns the closed position."""
        return self._positions.pop(ticket, None)

    def update_position(
        self,
        ticket: int,
        current_price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ):
        """Update position with current price and optionally modify SL/TP."""
        if ticket not in self._positions:
            return

        pos = self._positions[ticket]

        # Update unrealized PnL using contract size
        contract_size = pos.contract_size if hasattr(pos, 'contract_size') and pos.contract_size else 100000
        if pos.is_long:
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.volume * contract_size
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.volume * contract_size

        # Update SL/TP if provided
        if sl is not None:
            pos.sl = sl
        if tp is not None:
            pos.tp = tp

    def get_positions(self, symbol: Optional[str] = None) -> List['PositionTracker.TrackedPosition']:
        """Get all positions, optionally filtered by symbol."""
        positions = list(self._positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_position(self, ticket: int) -> Optional['PositionTracker.TrackedPosition']:
        """Get position by ticket."""
        return self._positions.get(ticket)

    def has_position(self, symbol: str, direction: Optional[str] = None) -> bool:
        """Check if we have a position."""
        positions = self.get_positions(symbol)
        if direction == 'long':
            return any(p.is_long for p in positions)
        elif direction == 'short':
            return any(p.is_short for p in positions)
        return len(positions) > 0

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL across all positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    def clear(self):
        """Clear all positions."""
        self._positions.clear()
