/**
 * API client for Leap Trading System
 */

const API_BASE = '/api/v1'

interface ApiError {
  error: {
    code: string
    message: string
    details?: Record<string, string[]>
  }
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    let errorMessage = 'API request failed'
    try {
      const error: ApiError = await response.json()
      errorMessage = error.error?.message || errorMessage
    } catch {
      // Response was not JSON
    }
    throw new Error(errorMessage)
  }

  const json = await response.json()
  return json.data || json
}

// Training API
export const trainingApi = {
  start: (config: TrainingConfig) =>
    fetchApi<TrainingJob>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  list: (params?: { status?: string; limit?: number; offset?: number }) => {
    const filteredParams = params
      ? Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))
      : undefined
    const queryString = filteredParams && Object.keys(filteredParams).length > 0
      ? '?' + new URLSearchParams(filteredParams as Record<string, string>).toString()
      : ''
    return fetchApi<{ jobs: TrainingJob[]; total: number }>(`/training/jobs${queryString}`)
  },

  get: (jobId: string) =>
    fetchApi<TrainingJob>(`/training/jobs/${jobId}`),

  stop: (jobId: string) =>
    fetchApi<TrainingJob>(`/training/jobs/${jobId}/stop`, { method: 'POST' }),

  logs: (jobId: string) =>
    fetchApi<{ logs: LogEntry[]; hasMore: boolean }>(`/training/jobs/${jobId}/logs`),
}

// Backtest API
export const backtestApi = {
  run: (config: BacktestConfig) =>
    fetchApi<{ jobId: string; status: string }>('/backtest/run', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  jobs: (params?: { status?: string; limit?: number; offset?: number }) => {
    const filteredParams = params
      ? Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))
      : undefined
    const queryString = filteredParams && Object.keys(filteredParams).length > 0
      ? '?' + new URLSearchParams(filteredParams as Record<string, string>).toString()
      : ''
    return fetchApi<{ jobs: BacktestJob[]; total: number }>(`/backtest/jobs${queryString}`)
  },

  list: (params?: { symbol?: string; limit?: number; offset?: number }) => {
    const filteredParams = params
      ? Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))
      : undefined
    const queryString = filteredParams && Object.keys(filteredParams).length > 0
      ? '?' + new URLSearchParams(filteredParams as Record<string, string>).toString()
      : ''
    return fetchApi<{ results: BacktestResult[]; total: number }>(`/backtest/results${queryString}`)
  },

  get: (resultId: string) =>
    fetchApi<BacktestResultDetail>(`/backtest/results/${resultId}`),

  compare: (resultIds: string[]) =>
    fetchApi('/backtest/compare', {
      method: 'POST',
      body: JSON.stringify({ resultIds }),
    }),
}

// Config API
export const configApi = {
  get: () => fetchApi<SystemConfig>('/config'),

  update: (config: Partial<SystemConfig>) =>
    fetchApi('/config', {
      method: 'PUT',
      body: JSON.stringify(config),
    }),

  templates: () =>
    fetchApi<{ templates: ConfigTemplate[] }>('/config/templates'),

  validate: (config: Partial<SystemConfig>) =>
    fetchApi('/config/validate', {
      method: 'POST',
      body: JSON.stringify(config),
    }),
}

// Models API
export const modelsApi = {
  list: () => fetchApi<{ models: ModelInfo[] }>('/models'),

  get: (directory: string) =>
    fetchApi<ModelDetail>(`/models/${encodeURIComponent(directory)}`),
}

// Logs API
export const logsApi = {
  files: () => fetchApi<{ files: LogFile[] }>('/logs/files'),

  get: (filename: string, params?: { level?: string; search?: string; limit?: number; offset?: number }) => {
    const filteredParams = params
      ? Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null))
      : undefined
    const queryString = filteredParams && Object.keys(filteredParams).length > 0
      ? '?' + new URLSearchParams(filteredParams as Record<string, string>).toString()
      : ''
    return fetchApi<{ filename: string; lines: LogLine[]; totalLines: number }>(
      `/logs/files/${filename}${queryString}`
    )
  },
}

// System API
export const systemApi = {
  health: () => fetchApi<HealthStatus>('/health'),
  metrics: () => fetchApi<SystemMetrics>('/metrics/system'),
}

// Types
export interface TrainingConfig {
  symbols: string[]
  timeframe: string
  multiTimeframe?: boolean
  bars: number
  epochs: number
  timesteps: number
  modelDir?: string
}

export interface TrainingJob {
  jobId: string
  status: string
  symbols: string[]
  timeframe?: string
  phase?: string
  progress?: {
    currentEpoch?: number
    totalEpochs?: number
    percent: number
  }
  metrics?: {
    trainLoss?: number
    valLoss?: number
  }
  createdAt: string
}

export interface LogEntry {
  timestamp: string
  level: string
  message: string
  logger?: string
}

export interface BacktestConfig {
  symbol: string
  timeframe: string
  bars: number
  modelDir: string
  realisticMode?: boolean
  monteCarlo?: boolean
}

export interface BacktestJob {
  jobId: string
  status: string
  symbol: string
  timeframe: string
  progress?: {
    percent: number
    currentStep?: string
  }
  createdAt: string
}

export interface BacktestResult {
  resultId: string
  symbol: string
  timeframe: string
  summary: {
    totalReturn: number
    sharpeRatio: number
    maxDrawdown: number
    winRate: number
    totalTrades: number
  }
  completedAt: string
}

export interface BacktestResultDetail extends BacktestResult {
  metrics?: {
    returns: {
      totalReturn: number
      annualizedReturn: number
    }
    risk: {
      volatility: number
      maxDrawdown: number
      var_95?: number
      cvar_95?: number
    }
    riskAdjusted: {
      sharpeRatio: number
      sortinoRatio: number
      calmarRatio?: number
    }
    trade: {
      totalTrades: number
      winRate: number
      profitFactor: number
      winningTrades?: number
      losingTrades?: number
      avgWinner?: number
      avgLoser?: number
    }
  }
  timeSeries?: {
    equityCurve: number[]
    drawdownCurve: number[]
    timestamps: string[]
  }
  trades?: Trade[]
}

export interface Trade {
  id: string
  entryTime: string
  exitTime: string
  direction: string
  entryPrice: number
  exitPrice: number
  pnl: number
  pnlPercent?: number
  size?: number
  status: string
}

export interface SystemConfig {
  data: object
  transformer: object
  ppo: object
  risk: object
  backtest: object
}

export interface ConfigTemplate {
  id: string
  name: string
  description?: string
  createdAt: string
}

export interface ModelInfo {
  directory: string
  predictor: { exists: boolean }
  agent: { exists: boolean }
  metadata: {
    symbol?: string
    timeframe?: string
  }
}

export interface ModelDetail extends ModelInfo {
  files: string[]
}

export interface LogFile {
  name: string
  size: number
  modifiedAt: string
}

export interface LogLine {
  number: number
  timestamp?: string
  level?: string
  logger?: string
  message: string
}

export interface HealthStatus {
  status: string
  version: string
}

export interface SystemMetrics {
  cpu: { percent: number; cores: number }
  memory: { total: number; used: number; percent: number }
  gpu: { available: boolean; name?: string }
  disk: { total: number; used: number; percent: number }
}
