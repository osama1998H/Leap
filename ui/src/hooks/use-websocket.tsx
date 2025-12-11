/**
 * WebSocket hook for real-time updates with automatic reconnection
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  Channel,
  ClientMessage,
  ConnectionStatus,
  ServerMessage,
  TrainingProgressData,
  BacktestProgressData,
  SystemMetricsData,
  LogEntryData,
  UseWebSocketReturn,
} from '../types/websocket';

const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;
const PING_INTERVAL_MS = 30000;
const MAX_LOGS = 500;

// Silent error handler - errors are caught but not logged to avoid console pollution
// In production, consider integrating with an error tracking service
const logError = (_message: string, _error?: unknown): void => {
  // Intentionally empty - errors are caught to prevent unhandled exceptions
};

export function useWebSocket(): UseWebSocketReturn {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [lastMessage, setLastMessage] = useState<ServerMessage | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<Map<string, TrainingProgressData>>(new Map());
  const [backtestProgress, setBacktestProgress] = useState<Map<string, BacktestProgressData>>(new Map());
  const [systemMetrics, setSystemMetrics] = useState<SystemMetricsData | null>(null);
  const [logs, setLogs] = useState<LogEntryData[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const subscriptionsRef = useRef<Set<string>>(new Set());

  // Calculate reconnect delay with exponential backoff
  const getReconnectDelay = useCallback(() => {
    const delay = Math.min(
      RECONNECT_DELAY_MS * Math.pow(2, reconnectAttemptRef.current),
      MAX_RECONNECT_DELAY_MS
    );
    return delay;
  }, []);

  // Send message to WebSocket
  const sendMessage = useCallback((message: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Subscribe to a channel
  const subscribe = useCallback((channel: Channel, jobId?: string) => {
    const key = `${channel}:${jobId || 'all'}`;
    subscriptionsRef.current.add(key);
    sendMessage({ type: 'subscribe', channel, jobId });
  }, [sendMessage]);

  // Unsubscribe from a channel
  const unsubscribe = useCallback((channel: Channel, jobId?: string) => {
    const key = `${channel}:${jobId || 'all'}`;
    subscriptionsRef.current.delete(key);
    sendMessage({ type: 'unsubscribe', channel, jobId });
  }, [sendMessage]);

  // Handle incoming messages
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: ServerMessage = JSON.parse(event.data);
      setLastMessage(message);

      switch (message.type) {
        case 'training_progress': {
          const data = message.data as TrainingProgressData;
          setTrainingProgress(prev => {
            const newMap = new Map(prev);
            newMap.set(data.jobId, data);
            return newMap;
          });
          break;
        }

        case 'backtest_progress': {
          const data = message.data as BacktestProgressData;
          setBacktestProgress(prev => {
            const newMap = new Map(prev);
            newMap.set(data.jobId, data);
            return newMap;
          });
          break;
        }

        case 'log_entry': {
          const data = message.data as LogEntryData;
          setLogs(prev => {
            const newLogs = [...prev, data];
            // Keep only last MAX_LOGS entries
            if (newLogs.length > MAX_LOGS) {
              return newLogs.slice(-MAX_LOGS);
            }
            return newLogs;
          });
          break;
        }

        case 'system_metrics': {
          const data = message.data as SystemMetricsData;
          setSystemMetrics(data);
          break;
        }

        case 'job_complete': {
          const data = message.data as { jobId: string };
          // Update training progress to show completed
          setTrainingProgress(prev => {
            const newMap = new Map(prev);
            const existing = newMap.get(data.jobId);
            if (existing) {
              newMap.set(data.jobId, { ...existing, status: 'completed' });
            }
            return newMap;
          });
          break;
        }

        case 'job_error': {
          const data = message.data as { jobId: string };
          // Update training progress to show failed
          setTrainingProgress(prev => {
            const newMap = new Map(prev);
            const existing = newMap.get(data.jobId);
            if (existing) {
              newMap.set(data.jobId, { ...existing, status: 'failed' });
            }
            return newMap;
          });
          break;
        }

        case 'pong':
          // Ping-pong for connection health check
          break;

        case 'subscribed':
        case 'unsubscribed':
          // Subscription confirmations
          break;

        case 'error':
          logError('Server error:', message.data);
          break;
      }
    } catch (error) {
      logError('Failed to parse message:', error);
    }
  }, []);

  // Resubscribe to all channels after reconnection
  const resubscribe = useCallback(() => {
    subscriptionsRef.current.forEach(key => {
      const [channel, jobId] = key.split(':');
      sendMessage({
        type: 'subscribe',
        channel: channel as Channel,
        jobId: jobId === 'all' ? undefined : jobId,
      });
    });
  }, [sendMessage]);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setStatus('connecting');

    try {
      wsRef.current = new WebSocket(WS_URL);

      wsRef.current.onopen = () => {
        setStatus('connected');
        reconnectAttemptRef.current = 0;

        // Start ping interval
        pingIntervalRef.current = setInterval(() => {
          sendMessage({ type: 'ping' });
        }, PING_INTERVAL_MS);

        // Resubscribe to previous channels
        resubscribe();
      };

      wsRef.current.onmessage = handleMessage;

      wsRef.current.onclose = () => {
        setStatus('disconnected');

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Schedule reconnection
        const delay = getReconnectDelay();
        reconnectAttemptRef.current++;
        setStatus('reconnecting');

        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, delay);
      };

      wsRef.current.onerror = (error) => {
        logError('Connection error:', error);
      };
    } catch (error) {
      logError('Failed to create connection:', error);
      setStatus('disconnected');
    }
  }, [handleMessage, getReconnectDelay, resubscribe, sendMessage]);

  // Initial connection
  useEffect(() => {
    connect();

    return () => {
      // Cleanup on unmount
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
      // Only close if WebSocket is open or connecting
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return {
    status,
    subscribe,
    unsubscribe,
    lastMessage,
    trainingProgress,
    backtestProgress,
    systemMetrics,
    logs,
  };
}

// Singleton WebSocket context for app-wide usage
import { createContext, useContext, type ReactNode } from 'react';

const WebSocketContext = createContext<UseWebSocketReturn | null>(null);

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const ws = useWebSocket();
  return (
    <WebSocketContext.Provider value={ws}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocketContext(): UseWebSocketReturn {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider');
  }
  return context;
}
