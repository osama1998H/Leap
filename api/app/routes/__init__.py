# API Routes
from .training import router as training_router
from .backtest import router as backtest_router
from .config import router as config_router
from .models import router as models_router
from .logs import router as logs_router
from .system import router as system_router
from .websocket import router as websocket_router

__all__ = [
    "training_router",
    "backtest_router",
    "config_router",
    "models_router",
    "logs_router",
    "system_router",
    "websocket_router",
]
