"""WebSocket connection manager for real-time updates."""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"

    # Server -> Client
    TRAINING_PROGRESS = "training_progress"
    BACKTEST_PROGRESS = "backtest_progress"
    LOG_ENTRY = "log_entry"
    SYSTEM_METRICS = "system_metrics"
    JOB_COMPLETE = "job_complete"
    JOB_ERROR = "job_error"
    PONG = "pong"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    ERROR = "error"


class Channel(str, Enum):
    """Subscription channels."""

    TRAINING = "training"
    BACKTEST = "backtest"
    LOGS = "logs"
    SYSTEM = "system"


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""

    _instance: Optional["WebSocketManager"] = None

    def __new__(cls) -> "WebSocketManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Active connections: websocket -> set of (channel, job_id) tuples
        self.connections: dict[WebSocket, set[tuple[str, Optional[str]]]] = {}
        # Channel subscribers: (channel, job_id) -> set of websockets
        self.channel_subscribers: dict[tuple[str, Optional[str]], set[WebSocket]] = {}
        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("WebSocketManager initialized")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.connections[websocket] = set()
        logger.info(f"WebSocket connected: {id(websocket)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            # Unsubscribe from all channels
            if websocket in self.connections:
                subscriptions = self.connections[websocket].copy()
                for channel, job_id in subscriptions:
                    key = (channel, job_id)
                    if key in self.channel_subscribers:
                        self.channel_subscribers[key].discard(websocket)
                        if not self.channel_subscribers[key]:
                            del self.channel_subscribers[key]
                del self.connections[websocket]
        logger.info(f"WebSocket disconnected: {id(websocket)}")

    async def subscribe(
        self, websocket: WebSocket, channel: str, job_id: Optional[str] = None
    ) -> None:
        """Subscribe a WebSocket to a channel."""
        async with self._lock:
            if websocket not in self.connections:
                return

            key = (channel, job_id)
            self.connections[websocket].add(key)

            if key not in self.channel_subscribers:
                self.channel_subscribers[key] = set()
            self.channel_subscribers[key].add(websocket)

        logger.info(f"WebSocket {id(websocket)} subscribed to {channel}:{job_id}")

        # Send confirmation
        await self.send_personal_message(
            websocket,
            MessageType.SUBSCRIBED,
            {"channel": channel, "jobId": job_id},
        )

    async def unsubscribe(
        self, websocket: WebSocket, channel: str, job_id: Optional[str] = None
    ) -> None:
        """Unsubscribe a WebSocket from a channel."""
        async with self._lock:
            if websocket not in self.connections:
                return

            key = (channel, job_id)
            self.connections[websocket].discard(key)

            if key in self.channel_subscribers:
                self.channel_subscribers[key].discard(websocket)
                if not self.channel_subscribers[key]:
                    del self.channel_subscribers[key]

        logger.info(f"WebSocket {id(websocket)} unsubscribed from {channel}:{job_id}")

        # Send confirmation
        await self.send_personal_message(
            websocket,
            MessageType.UNSUBSCRIBED,
            {"channel": channel, "jobId": job_id},
        )

    async def send_personal_message(
        self, websocket: WebSocket, message_type: MessageType, data: Any
    ) -> None:
        """Send a message to a specific WebSocket."""
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")

    async def broadcast_to_channel(
        self,
        channel: str,
        message_type: MessageType,
        data: Any,
        job_id: Optional[str] = None,
    ) -> None:
        """Broadcast a message to all subscribers of a channel."""
        message = {
            "type": message_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Get subscribers for both specific job and general channel
        subscribers = set()
        async with self._lock:
            # Subscribers to specific job
            if job_id:
                key = (channel, job_id)
                if key in self.channel_subscribers:
                    subscribers.update(self.channel_subscribers[key])
            # Subscribers to all jobs on this channel
            general_key = (channel, None)
            if general_key in self.channel_subscribers:
                subscribers.update(self.channel_subscribers[general_key])

        # Send to all subscribers
        disconnected = []
        for websocket in subscribers:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected sockets
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def handle_message(self, websocket: WebSocket, message: dict) -> None:
        """Handle incoming WebSocket message."""
        msg_type = message.get("type")
        channel = message.get("channel")
        job_id = message.get("jobId")

        if msg_type == MessageType.SUBSCRIBE.value:
            if channel:
                await self.subscribe(websocket, channel, job_id)
            else:
                await self.send_personal_message(
                    websocket,
                    MessageType.ERROR,
                    {"message": "Channel is required for subscribe"},
                )

        elif msg_type == MessageType.UNSUBSCRIBE.value:
            if channel:
                await self.unsubscribe(websocket, channel, job_id)

        elif msg_type == MessageType.PING.value:
            await self.send_personal_message(websocket, MessageType.PONG, {})

        else:
            await self.send_personal_message(
                websocket,
                MessageType.ERROR,
                {"message": f"Unknown message type: {msg_type}"},
            )

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.connections)

    def get_channel_subscriber_count(
        self, channel: str, job_id: Optional[str] = None
    ) -> int:
        """Get the number of subscribers for a channel."""
        key = (channel, job_id)
        return len(self.channel_subscribers.get(key, set()))


# Global instance
ws_manager = WebSocketManager()
