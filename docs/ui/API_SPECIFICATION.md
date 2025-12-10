# Backend API Specification

This document specifies the REST API and WebSocket interfaces required for the Leap Trading System Web UI.

---

## Base Configuration

```
Base URL: http://localhost:8000/api/v1
WebSocket: ws://localhost:8000/ws
Content-Type: application/json
```

---

## Authentication

**Phase 1 (MVP)**: No authentication required (localhost only)

**Future**: Bearer token authentication
```
Authorization: Bearer <token>
```

---

## Common Response Formats

### Success Response

```json
{
  "data": { ... },
  "meta": {
    "timestamp": "2024-01-15T14:30:00Z",
    "requestId": "req_abc123"
  }
}
```

### Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid configuration provided",
    "details": {
      "epochs": ["Must be between 1 and 1000"]
    }
  },
  "meta": {
    "timestamp": "2024-01-15T14:30:00Z",
    "requestId": "req_abc123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `JOB_RUNNING` | 409 | Cannot modify running job |
| `INTERNAL_ERROR` | 500 | Server error |
| `MODEL_NOT_FOUND` | 404 | Model files not found |

---

## Training Endpoints

### Start Training

**POST** `/training/start`

Start a new training job.

**Request Body:**
```json
{
  "symbols": ["EURUSD", "GBPUSD"],
  "timeframe": "1h",
  "multiTimeframe": true,
  "additionalTimeframes": ["15m", "4h", "1d"],
  "bars": 50000,
  "epochs": 100,
  "timesteps": 1000000,
  "modelDir": "./saved_models",
  "config": {
    "transformer": {
      "dModel": 128,
      "nHeads": 8,
      "nEncoderLayers": 4,
      "dropout": 0.1,
      "learningRate": 0.0001
    },
    "ppo": {
      "learningRate": 0.0003,
      "gamma": 0.99,
      "clipEpsilon": 0.2
    }
  }
}
```

**Response (201 Created):**
```json
{
  "data": {
    "jobId": "train_abc123",
    "status": "starting",
    "symbols": ["EURUSD", "GBPUSD"],
    "createdAt": "2024-01-15T14:30:00Z",
    "estimatedDuration": 7200
  }
}
```

---

### List Training Jobs

**GET** `/training/jobs`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status (running, completed, failed) |
| `symbol` | string | Filter by symbol |
| `limit` | int | Max results (default 20) |
| `offset` | int | Pagination offset |

**Response (200 OK):**
```json
{
  "data": {
    "jobs": [
      {
        "jobId": "train_abc123",
        "status": "running",
        "symbols": ["EURUSD"],
        "phase": "transformer",
        "progress": {
          "currentEpoch": 45,
          "totalEpochs": 100,
          "percent": 45
        },
        "createdAt": "2024-01-15T14:30:00Z"
      }
    ],
    "total": 15,
    "limit": 20,
    "offset": 0
  }
}
```

---

### Get Training Job

**GET** `/training/jobs/{jobId}`

**Response (200 OK):**
```json
{
  "data": {
    "jobId": "train_abc123",
    "status": "running",
    "symbols": ["EURUSD"],
    "timeframe": "1h",
    "config": { ... },
    "phase": "transformer",
    "progress": {
      "currentEpoch": 45,
      "totalEpochs": 100,
      "percent": 45,
      "currentTimestep": null,
      "totalTimesteps": 1000000
    },
    "metrics": {
      "trainLoss": 0.00234,
      "valLoss": 0.00256,
      "learningRate": 0.0001
    },
    "timing": {
      "startedAt": "2024-01-15T14:30:00Z",
      "elapsedSeconds": 1200,
      "estimatedRemainingSeconds": 1600
    },
    "createdAt": "2024-01-15T14:30:00Z"
  }
}
```

---

### Stop Training Job

**POST** `/training/jobs/{jobId}/stop`

**Response (200 OK):**
```json
{
  "data": {
    "jobId": "train_abc123",
    "status": "stopped",
    "stoppedAt": "2024-01-15T15:00:00Z"
  }
}
```

---

### Get Training Logs

**GET** `/training/jobs/{jobId}/logs`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | string | Filter by level (DEBUG, INFO, WARNING, ERROR) |
| `since` | datetime | Logs since timestamp |
| `limit` | int | Max lines (default 100) |

**Response (200 OK):**
```json
{
  "data": {
    "logs": [
      {
        "timestamp": "2024-01-15T14:45:23Z",
        "level": "INFO",
        "message": "Epoch 45/100 - train_loss: 0.00234, val_loss: 0.00256",
        "logger": "trainer"
      }
    ],
    "hasMore": true
  }
}
```

---

## Backtest Endpoints

### Run Backtest

**POST** `/backtest/run`

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "1h",
  "bars": 50000,
  "modelDir": "./saved_models",
  "realisticMode": true,
  "monteCarlo": true,
  "config": {
    "initialBalance": 10000,
    "leverage": 100,
    "spreadPips": 1.5,
    "commissionPerLot": 7.0,
    "slippagePips": 1.0,
    "riskPerTrade": 0.02,
    "nSimulations": 1000
  }
}
```

**Response (201 Created):**
```json
{
  "data": {
    "jobId": "backtest_xyz789",
    "status": "running",
    "createdAt": "2024-01-15T14:30:00Z"
  }
}
```

---

### List Backtest Results

**GET** `/backtest/results`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | string | Filter by symbol |
| `minSharpe` | float | Minimum Sharpe ratio |
| `limit` | int | Max results |
| `offset` | int | Pagination offset |

**Response (200 OK):**
```json
{
  "data": {
    "results": [
      {
        "resultId": "backtest_xyz789",
        "symbol": "EURUSD",
        "timeframe": "1h",
        "summary": {
          "totalReturn": 0.124,
          "sharpeRatio": 1.85,
          "maxDrawdown": -0.082,
          "winRate": 0.583,
          "totalTrades": 156
        },
        "completedAt": "2024-01-15T14:35:00Z"
      }
    ],
    "total": 25
  }
}
```

---

### Get Backtest Result

**GET** `/backtest/results/{resultId}`

**Response (200 OK):**
```json
{
  "data": {
    "resultId": "backtest_xyz789",
    "symbol": "EURUSD",
    "timeframe": "1h",
    "config": { ... },
    "metrics": {
      "returns": {
        "totalReturn": 0.124,
        "annualizedReturn": 0.285,
        "cagr": 0.285
      },
      "risk": {
        "volatility": 0.182,
        "downsideVolatility": 0.145,
        "maxDrawdown": -0.082,
        "maxDrawdownDuration": 72,
        "var95": -0.012,
        "cvar95": -0.018
      },
      "riskAdjusted": {
        "sharpeRatio": 1.85,
        "sortinoRatio": 2.12,
        "calmarRatio": 3.48,
        "omegaRatio": 1.65
      },
      "trade": {
        "totalTrades": 156,
        "winningTrades": 91,
        "losingTrades": 65,
        "winRate": 0.583,
        "profitFactor": 1.87,
        "avgTrade": 79.5,
        "avgWinner": 125.4,
        "avgLoser": -87.2,
        "largestWinner": 485.0,
        "largestLoser": -312.0
      },
      "distribution": {
        "skewness": 0.34,
        "kurtosis": 2.15,
        "tailRatio": 1.25
      }
    },
    "timeSeries": {
      "equityCurve": [10000, 10050, 10120, ...],
      "drawdownCurve": [0, -0.005, -0.003, ...],
      "timestamps": ["2024-01-01T00:00:00Z", ...]
    },
    "trades": [
      {
        "id": "trade_001",
        "entryTime": "2024-01-15T09:00:00Z",
        "exitTime": "2024-01-15T14:00:00Z",
        "direction": "long",
        "entryPrice": 1.0850,
        "exitPrice": 1.0892,
        "size": 0.5,
        "pnl": 125.4,
        "pnlPercent": 0.0125,
        "status": "tp_hit"
      }
    ],
    "monteCarlo": {
      "finalEquity": {
        "mean": 1.124,
        "std": 0.08,
        "median": 1.12,
        "percentile5": 1.02,
        "percentile95": 1.28
      },
      "maxDrawdown": {
        "mean": -0.082,
        "percentile95": -0.15
      },
      "probabilityOfProfit": 0.92,
      "probabilityOfRuin": 0.02
    },
    "completedAt": "2024-01-15T14:35:00Z"
  }
}
```

---

### Compare Backtests

**POST** `/backtest/compare`

**Request Body:**
```json
{
  "resultIds": ["backtest_xyz789", "backtest_abc456"]
}
```

**Response (200 OK):**
```json
{
  "data": {
    "results": [
      {
        "resultId": "backtest_xyz789",
        "symbol": "EURUSD",
        "metrics": { ... }
      },
      {
        "resultId": "backtest_abc456",
        "symbol": "EURUSD",
        "metrics": { ... }
      }
    ],
    "comparison": {
      "bestByReturn": "backtest_xyz789",
      "bestBySharpe": "backtest_xyz789",
      "bestByDrawdown": "backtest_abc456"
    }
  }
}
```

---

## Walk-Forward Endpoints

### Run Walk-Forward

**POST** `/walkforward/run`

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "1h",
  "bars": 100000,
  "trainWindowDays": 180,
  "testWindowDays": 30,
  "stepSizeDays": 30
}
```

**Response (201 Created):**
```json
{
  "data": {
    "jobId": "wf_def456",
    "status": "running",
    "estimatedFolds": 8
  }
}
```

---

### Get Walk-Forward Result

**GET** `/walkforward/results/{resultId}`

**Response (200 OK):**
```json
{
  "data": {
    "resultId": "wf_def456",
    "symbol": "EURUSD",
    "nFolds": 8,
    "aggregate": {
      "totalReturn": {
        "mean": 0.062,
        "std": 0.038,
        "min": -0.023,
        "max": 0.091
      },
      "sharpeRatio": {
        "mean": 1.15,
        "std": 0.45,
        "min": 0.52,
        "max": 1.82
      },
      "maxDrawdown": {
        "mean": -0.058,
        "std": 0.021,
        "min": -0.024,
        "max": -0.092
      },
      "winRate": {
        "mean": 0.542,
        "std": 0.053,
        "min": 0.45,
        "max": 0.62
      }
    },
    "consistency": {
      "profitableFolds": 6,
      "profitableRatio": 0.75
    },
    "folds": [
      {
        "fold": 0,
        "trainStart": "2023-01-01",
        "trainEnd": "2023-06-30",
        "testStart": "2023-07-01",
        "testEnd": "2023-07-31",
        "metrics": {
          "totalReturn": 0.082,
          "sharpeRatio": 1.45,
          "maxDrawdown": -0.045,
          "winRate": 0.58
        }
      }
    ],
    "completedAt": "2024-01-15T16:00:00Z"
  }
}
```

---

## Configuration Endpoints

### Get Current Config

**GET** `/config`

**Response (200 OK):**
```json
{
  "data": {
    "data": {
      "symbols": ["EURUSD", "GBPUSD"],
      "primaryTimeframe": "1h",
      "additionalTimeframes": ["15m", "4h", "1d"],
      "lookbackWindow": 120,
      "predictionHorizon": 12
    },
    "transformer": {
      "dModel": 128,
      "nHeads": 8,
      "nEncoderLayers": 4,
      "dropout": 0.1,
      "learningRate": 0.0001,
      "epochs": 100,
      "patience": 15
    },
    "ppo": {
      "learningRate": 0.0003,
      "gamma": 0.99,
      "gaeambda": 0.95,
      "clipEpsilon": 0.2,
      "totalTimesteps": 1000000
    },
    "risk": {
      "maxPositionSize": 0.02,
      "maxDailyLoss": 0.05,
      "defaultStopLossPips": 50,
      "defaultTakeProfitPips": 100
    },
    "backtest": {
      "initialBalance": 10000,
      "leverage": 100,
      "spreadPips": 1.5,
      "commissionPerLot": 7.0
    }
  }
}
```

---

### Update Config

**PUT** `/config`

**Request Body:**
```json
{
  "transformer": {
    "epochs": 200,
    "learningRate": 0.00005
  }
}
```

**Response (200 OK):**
```json
{
  "data": {
    "updated": true,
    "config": { ... }
  }
}
```

---

### List Config Templates

**GET** `/config/templates`

**Response (200 OK):**
```json
{
  "data": {
    "templates": [
      {
        "id": "quick-test",
        "name": "Quick Test",
        "description": "Fast iteration with minimal training",
        "createdAt": "2024-01-10T10:00:00Z"
      },
      {
        "id": "production",
        "name": "Production",
        "description": "Full training for deployment",
        "createdAt": "2024-01-10T10:00:00Z"
      }
    ]
  }
}
```

---

### Save Config Template

**POST** `/config/templates`

**Request Body:**
```json
{
  "name": "My Config",
  "description": "Custom configuration for EURUSD",
  "config": { ... }
}
```

**Response (201 Created):**
```json
{
  "data": {
    "id": "my-config-abc123",
    "name": "My Config",
    "createdAt": "2024-01-15T14:30:00Z"
  }
}
```

---

### Validate Config

**POST** `/config/validate`

**Request Body:**
```json
{
  "transformer": {
    "epochs": -1,
    "learningRate": 0.0001
  }
}
```

**Response (400 Bad Request):**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Configuration validation failed",
    "details": {
      "transformer.epochs": ["Must be greater than 0"]
    }
  }
}
```

---

## Models Endpoints

### List Models

**GET** `/models`

**Response (200 OK):**
```json
{
  "data": {
    "models": [
      {
        "directory": "./saved_models",
        "predictor": {
          "exists": true,
          "inputDim": 128,
          "createdAt": "2024-01-14T10:00:00Z"
        },
        "agent": {
          "exists": true,
          "stateDim": 256,
          "actionDim": 4,
          "createdAt": "2024-01-14T10:00:00Z"
        },
        "metadata": {
          "symbol": "EURUSD",
          "timeframe": "1h",
          "trainedAt": "2024-01-14T10:00:00Z"
        }
      }
    ]
  }
}
```

---

### Get Model Details

**GET** `/models/{directory}`

**Response (200 OK):**
```json
{
  "data": {
    "directory": "./saved_models",
    "files": [
      "predictor.pt",
      "agent.pt",
      "model_metadata.json",
      "config.json"
    ],
    "metadata": {
      "predictor": {
        "inputDim": 128,
        "featureNames": ["open", "high", "low", "close", ...]
      },
      "agent": {
        "stateDim": 256,
        "actionDim": 4
      },
      "trainedSymbol": "EURUSD",
      "trainedTimeframe": "1h",
      "createdAt": "2024-01-14T10:00:00Z"
    },
    "config": { ... }
  }
}
```

---

### Download Model

**GET** `/models/{directory}/download`

Returns a ZIP file containing all model files.

**Response (200 OK):**
```
Content-Type: application/zip
Content-Disposition: attachment; filename="model_20240114.zip"
```

---

## Logs Endpoints

### List Log Files

**GET** `/logs/files`

**Response (200 OK):**
```json
{
  "data": {
    "files": [
      {
        "name": "leap_20240115.log",
        "size": 1024000,
        "modifiedAt": "2024-01-15T14:30:00Z"
      }
    ]
  }
}
```

---

### Get Log File

**GET** `/logs/files/{filename}`

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | string | Filter by level |
| `search` | string | Text search |
| `limit` | int | Max lines |
| `offset` | int | Line offset |

**Response (200 OK):**
```json
{
  "data": {
    "filename": "leap_20240115.log",
    "lines": [
      {
        "number": 1,
        "timestamp": "2024-01-15T14:45:23Z",
        "level": "INFO",
        "logger": "main",
        "message": "Starting training..."
      }
    ],
    "totalLines": 5000
  }
}
```

---

### Stream Logs (SSE)

**GET** `/logs/stream`

Server-Sent Events endpoint for real-time log streaming.

```
event: log
data: {"timestamp":"2024-01-15T14:45:23Z","level":"INFO","message":"..."}

event: log
data: {"timestamp":"2024-01-15T14:45:24Z","level":"DEBUG","message":"..."}
```

---

## System Endpoints

### Health Check

**GET** `/health`

**Response (200 OK):**
```json
{
  "data": {
    "status": "healthy",
    "version": "1.0.0",
    "components": {
      "database": "ok",
      "mlflow": "ok",
      "gpu": "available"
    }
  }
}
```

---

### System Metrics

**GET** `/metrics/system`

**Response (200 OK):**
```json
{
  "data": {
    "cpu": {
      "percent": 34.5,
      "cores": 8
    },
    "memory": {
      "total": 16000000000,
      "used": 8500000000,
      "percent": 53.1
    },
    "gpu": {
      "available": true,
      "name": "NVIDIA GeForce RTX 3080",
      "memory": {
        "total": 10000000000,
        "used": 6200000000,
        "percent": 62.0
      }
    },
    "disk": {
      "total": 500000000000,
      "used": 250000000000,
      "percent": 50.0
    }
  }
}
```

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Client Messages

#### Subscribe

```json
{
  "type": "subscribe",
  "channel": "training",
  "jobId": "train_abc123"
}
```

#### Unsubscribe

```json
{
  "type": "unsubscribe",
  "channel": "training",
  "jobId": "train_abc123"
}
```

#### Ping

```json
{
  "type": "ping"
}
```

### Server Messages

#### Training Progress

```json
{
  "type": "training_progress",
  "data": {
    "jobId": "train_abc123",
    "phase": "transformer",
    "epoch": 45,
    "totalEpochs": 100,
    "trainLoss": 0.00234,
    "valLoss": 0.00256,
    "learningRate": 0.0001,
    "elapsedSeconds": 1200,
    "estimatedRemainingSeconds": 1600
  },
  "timestamp": "2024-01-15T14:45:23Z"
}
```

#### Backtest Progress

```json
{
  "type": "backtest_progress",
  "data": {
    "jobId": "backtest_xyz789",
    "currentBar": 12450,
    "totalBars": 50000,
    "currentEquity": 10450.50,
    "currentDrawdown": -0.023,
    "tradesExecuted": 45
  },
  "timestamp": "2024-01-15T14:45:23Z"
}
```

#### Log Entry

```json
{
  "type": "log_entry",
  "data": {
    "timestamp": "2024-01-15T14:45:23Z",
    "level": "INFO",
    "logger": "trainer",
    "message": "Epoch 45/100 - train_loss: 0.00234"
  },
  "timestamp": "2024-01-15T14:45:23Z"
}
```

#### Job Complete

```json
{
  "type": "job_complete",
  "data": {
    "jobId": "train_abc123",
    "status": "completed",
    "result": {
      "finalLoss": 0.00189,
      "bestEpoch": 92,
      "modelPath": "./saved_models"
    }
  },
  "timestamp": "2024-01-15T16:00:00Z"
}
```

#### Job Error

```json
{
  "type": "job_error",
  "data": {
    "jobId": "train_abc123",
    "error": {
      "code": "CUDA_OOM",
      "message": "CUDA out of memory. Reduce batch size.",
      "recoverable": true
    }
  },
  "timestamp": "2024-01-15T15:30:00Z"
}
```

#### Pong

```json
{
  "type": "pong",
  "timestamp": "2024-01-15T14:45:23Z"
}
```

---

## Rate Limits

| Endpoint Category | Limit |
|-------------------|-------|
| Training start | 5/minute |
| Backtest start | 10/minute |
| Config updates | 30/minute |
| Read operations | 100/minute |
| WebSocket messages | 50/second |

---

## Pagination

All list endpoints support pagination:

```
GET /training/jobs?limit=20&offset=40
```

Response includes pagination info:
```json
{
  "data": { ... },
  "pagination": {
    "limit": 20,
    "offset": 40,
    "total": 150,
    "hasMore": true
  }
}
```
