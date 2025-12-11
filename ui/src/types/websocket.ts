/**
 * WebSocket message types and interfaces for real-time updates
 */

// Message types from server
export type MessageType =
  | 'training_progress'
  | 'backtest_progress'
  | 'log_entry'
  | 'system_metrics'
  | 'job_complete'
  | 'job_error'
  | 'pong'
  | 'subscribed'
  | 'unsubscribed'
  | 'error';

// Message types to server
export type ClientMessageType = 'subscribe' | 'unsubscribe' | 'ping';

// Subscription channels
export type Channel = 'training' | 'backtest' | 'logs' | 'system';

// Client message structure
export interface ClientMessage {
  type: ClientMessageType;
  channel?: Channel;
  jobId?: string;
}

// Server message structure
export interface ServerMessage<T = unknown> {
  type: MessageType;
  data: T;
  timestamp: string;
}

// Training progress data
export interface TrainingProgressData {
  jobId: string;
  status: string;
  progress: {
    currentEpoch?: number;
    totalEpochs?: number;
    percent?: number;
    currentTimestep?: number;
    totalTimesteps?: number;
  };
  phase?: 'transformer' | 'agent';
  epoch?: number;
  totalEpochs?: number;
  trainLoss?: number;
  valLoss?: number;
  learningRate?: number;
  elapsedSeconds?: number;
  estimatedRemainingSeconds?: number;
}

// Backtest progress data
export interface BacktestProgressData {
  jobId: string;
  status: string;
  currentBar?: number;
  totalBars?: number;
  currentEquity?: number;
  currentDrawdown?: number;
  tradesExecuted?: number;
}

// Log entry data
export interface LogEntryData {
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  logger: string;
  message: string;
}

// System metrics data
export interface SystemMetricsData {
  cpu: {
    percent: number;
    cores: number;
  };
  memory: {
    total: number;
    used: number;
    percent: number;
  };
  gpu?: {
    available: boolean;
    name?: string;
    memory?: {
      total: number;
      used: number;
      percent: number;
    };
  };
  disk: {
    total: number;
    used: number;
    percent: number;
  };
}

// Job complete data
export interface JobCompleteData {
  jobId: string;
  status: 'completed';
  result?: {
    finalLoss?: number;
    bestEpoch?: number;
    modelPath?: string;
  };
}

// Job error data
export interface JobErrorData {
  jobId: string;
  error: {
    code?: string;
    message: string;
    recoverable: boolean;
  };
}

// Subscription confirmation data
export interface SubscriptionData {
  channel: Channel;
  jobId?: string;
}

// Error data
export interface ErrorData {
  message: string;
}

// WebSocket connection status
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

// WebSocket hook return type
export interface UseWebSocketReturn {
  status: ConnectionStatus;
  subscribe: (channel: Channel, jobId?: string) => void;
  unsubscribe: (channel: Channel, jobId?: string) => void;
  lastMessage: ServerMessage | null;
  trainingProgress: Map<string, TrainingProgressData>;
  backtestProgress: Map<string, BacktestProgressData>;
  systemMetrics: SystemMetricsData | null;
  logs: LogEntryData[];
}
