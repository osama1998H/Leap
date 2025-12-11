import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Square, Pause, Play, Brain, Clock, Activity, Wifi, WifiOff } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { trainingApi } from '@/lib/api'
import { useToast } from '@/hooks/use-toast'
import { useWebSocketContext } from '@/hooks/use-websocket'
import { useTrainingStore } from '@/stores/training'
import { useUIStore } from '@/stores/ui'
import { useState, useEffect, useRef } from 'react'

interface LossDataPoint {
  epoch: number
  trainLoss: number | null
  valLoss: number | null
}

export default function TrainingMonitorPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  // WebSocket for real-time updates
  const { status: wsStatus, subscribe, unsubscribe, trainingProgress } = useWebSocketContext()

  // Zustand stores
  const { updateJobFromProgress } = useTrainingStore()
  const { addJobNotification, autoScrollLogs } = useUIStore()

  // Track loss history for chart
  const [lossHistory, setLossHistory] = useState<LossDataPoint[]>([])

  // Auto-scroll ref for logs
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Initial data fetch (fallback for when WebSocket isn't connected)
  const { data: job, isLoading, isError } = useQuery({
    queryKey: ['training', 'job', id],
    queryFn: () => trainingApi.get(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      // Only poll if WebSocket is disconnected
      if (wsStatus === 'connected') {
        return false
      }
      const status = query.state.data?.status
      if (status === 'completed' || status === 'failed' || status === 'stopped') {
        return false
      }
      return 2000
    },
  })

  const { data: logs } = useQuery({
    queryKey: ['training', 'logs', id],
    queryFn: () => trainingApi.logs(id!),
    enabled: !!id,
    refetchInterval: () => {
      // Only poll if WebSocket is disconnected
      if (wsStatus === 'connected') {
        return false
      }
      const status = job?.status
      if (status === 'completed' || status === 'failed' || status === 'stopped') {
        return false
      }
      return 3000
    },
  })

  // Subscribe to WebSocket updates for this job
  useEffect(() => {
    if (id && wsStatus === 'connected') {
      subscribe('training', id)
      subscribe('logs', id)
      return () => {
        unsubscribe('training', id)
        unsubscribe('logs', id)
      }
    }
  }, [id, wsStatus, subscribe, unsubscribe])

  // Get real-time progress from WebSocket or fall back to REST data
  const wsProgress = id ? trainingProgress.get(id) : undefined
  const currentProgress = wsProgress || job

  // Update store and loss history from WebSocket data
  useEffect(() => {
    if (wsProgress) {
      updateJobFromProgress(wsProgress)

      // Check for job completion/failure and show notification
      if (wsProgress.status === 'completed' || wsProgress.status === 'failed' || wsProgress.status === 'stopped') {
        addJobNotification(wsProgress.jobId, wsProgress.status as 'completed' | 'failed' | 'stopped')
        queryClient.invalidateQueries({ queryKey: ['training'] })
      }
    }
  }, [wsProgress, updateJobFromProgress, addJobNotification, queryClient])

  // Update loss history when progress changes
  useEffect(() => {
    const progress = currentProgress
    if (progress) {
      const trainLoss = wsProgress?.trainLoss ?? job?.metrics?.trainLoss
      const valLoss = wsProgress?.valLoss ?? job?.metrics?.valLoss
      const epoch = wsProgress?.epoch ?? progress?.progress?.currentEpoch

      if (trainLoss !== undefined && epoch !== undefined) {
        setLossHistory(prev => {
          const existingIndex = prev.findIndex(p => p.epoch === epoch)
          const newPoint: LossDataPoint = {
            epoch,
            trainLoss: trainLoss ?? null,
            valLoss: valLoss ?? null,
          }

          if (existingIndex >= 0) {
            const updated = [...prev]
            updated[existingIndex] = newPoint
            return updated
          } else {
            return [...prev, newPoint].sort((a, b) => a.epoch - b.epoch)
          }
        })
      }
    }
  }, [currentProgress, wsProgress, job])

  // Auto-scroll logs
  useEffect(() => {
    if (autoScrollLogs && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScrollLogs])

  // Mutations for job control
  const stopJob = useMutation({
    mutationFn: trainingApi.stop,
    onSuccess: () => {
      toast({
        title: 'Training Stopped',
        description: 'The training job has been stopped.',
      })
      queryClient.invalidateQueries({ queryKey: ['training'] })
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to stop training',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const pauseJob = useMutation({
    mutationFn: trainingApi.pause,
    onSuccess: () => {
      toast({
        title: 'Training Paused',
        description: 'The training job has been paused. Resume when ready.',
      })
      queryClient.invalidateQueries({ queryKey: ['training'] })
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to pause training',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const resumeJob = useMutation({
    mutationFn: trainingApi.resume,
    onSuccess: () => {
      toast({
        title: 'Training Resumed',
        description: 'The training job is continuing.',
      })
      queryClient.invalidateQueries({ queryKey: ['training'] })
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to resume training',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'running':
        return <Badge variant="default">Running</Badge>
      case 'completed':
        return <Badge variant="success">Completed</Badge>
      case 'failed':
        return <Badge variant="destructive">Failed</Badge>
      case 'stopped':
        return <Badge variant="secondary">Stopped</Badge>
      case 'paused':
        return <Badge variant="outline" className="border-yellow-500 text-yellow-600">Paused</Badge>
      case 'pending':
        return <Badge variant="outline">Pending</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = seconds % 60

    if (hours > 0) {
      return `${hours}h ${minutes}m`
    }
    if (minutes > 0) {
      return `${minutes}m ${secs}s`
    }
    return `${secs}s`
  }

  const formatETA = (seconds: number | undefined) => {
    if (seconds === undefined || seconds <= 0) return '-'
    return formatDuration(Math.floor(seconds))
  }

  const calculateElapsedSeconds = (createdAt: string) => {
    const start = new Date(createdAt)
    const now = new Date()
    return Math.floor((now.getTime() - start.getTime()) / 1000)
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (isError || !job) {
    return (
      <div className="space-y-4">
        <Button variant="ghost" onClick={() => navigate('/')}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Button>
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            Training job not found or failed to load.
          </CardContent>
        </Card>
      </div>
    )
  }

  const status = wsProgress?.status || job.status
  const isRunning = status === 'running'
  const isPaused = status === 'paused'
  const isActive = isRunning || isPaused

  // Get progress values
  const progressPercent = wsProgress?.progress?.percent ?? job.progress?.percent ?? 0
  const currentEpoch = wsProgress?.epoch ?? job.progress?.currentEpoch ?? 0
  const totalEpochs = wsProgress?.totalEpochs ?? job.progress?.totalEpochs ?? 100
  const trainLoss = wsProgress?.trainLoss ?? job.metrics?.trainLoss
  const valLoss = wsProgress?.valLoss ?? job.metrics?.valLoss
  const elapsedSeconds = wsProgress?.elapsedSeconds ?? calculateElapsedSeconds(job.createdAt)
  const etaSeconds = wsProgress?.estimatedRemainingSeconds

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => navigate('/')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <Brain className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-bold">
                {job.symbols?.join(', ')} {job.timeframe}
              </h1>
              {getStatusBadge(status)}
              {/* WebSocket connection indicator */}
              <span className="ml-2" title={wsStatus === 'connected' ? 'Live updates active' : 'Using polling'}>
                {wsStatus === 'connected' ? (
                  <Wifi className="h-4 w-4 text-green-500" />
                ) : (
                  <WifiOff className="h-4 w-4 text-muted-foreground" />
                )}
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              Training Job {job.jobId.slice(0, 12)}...
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isRunning && (
            <Button
              variant="outline"
              onClick={() => pauseJob.mutate(job.jobId)}
              disabled={pauseJob.isPending}
            >
              <Pause className="mr-2 h-4 w-4" />
              Pause
            </Button>
          )}
          {isPaused && (
            <Button
              variant="outline"
              onClick={() => resumeJob.mutate(job.jobId)}
              disabled={resumeJob.isPending}
            >
              <Play className="mr-2 h-4 w-4" />
              Resume
            </Button>
          )}
          {isActive && (
            <Button
              variant="destructive"
              onClick={() => stopJob.mutate(job.jobId)}
              disabled={stopJob.isPending}
            >
              <Square className="mr-2 h-4 w-4" />
              Stop
            </Button>
          )}
        </div>
      </div>

      {/* Progress & Stats Cards */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Progress</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{progressPercent}%</div>
            <Progress value={progressPercent} className="mt-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Epoch</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {currentEpoch} / {totalEpochs}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Phase: {wsProgress?.phase ?? job.phase ?? 'Transformer'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Train Loss</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {trainLoss?.toFixed(6) ?? '-'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Val: {valLoss?.toFixed(6) ?? '-'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Elapsed</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatDuration(elapsedSeconds)}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Started: {new Date(job.createdAt).toLocaleTimeString()}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ETA</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatETA(etaSeconds)}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {isRunning ? 'Estimated remaining' : isPaused ? 'Paused' : 'Finished'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Loss Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Training Loss</CardTitle>
          <CardDescription>Train and validation loss over epochs (live updates)</CardDescription>
        </CardHeader>
        <CardContent>
          {lossHistory.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossHistory}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis
                  dataKey="epoch"
                  label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
                  className="text-xs"
                />
                <YAxis
                  label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
                  className="text-xs"
                  tickFormatter={(value) => value.toFixed(4)}
                />
                <Tooltip
                  formatter={(value: number) => value?.toFixed(6)}
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="trainLoss"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={false}
                  name="Train Loss"
                  connectNulls
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="valLoss"
                  stroke="hsl(var(--destructive))"
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                  connectNulls
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              {isActive ? 'Waiting for training data...' : 'No loss data available'}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Logs */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Training Logs</CardTitle>
              <CardDescription>Recent training output (live streaming)</CardDescription>
            </div>
            {wsStatus === 'connected' && isActive && (
              <Badge variant="outline" className="text-green-500 border-green-500">
                <span className="mr-1 h-2 w-2 rounded-full bg-green-500 animate-pulse inline-block" />
                Live
              </Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded-lg p-4 font-mono text-sm max-h-[300px] overflow-y-auto">
            {logs?.logs && logs.logs.length > 0 ? (
              <>
                {logs.logs.slice(-50).map((log, index) => (
                  <div key={index} className="py-0.5">
                    <span className="text-muted-foreground">{log.timestamp?.slice(11, 19)}</span>
                    {' '}
                    <span className={
                      log.level === 'ERROR' ? 'text-destructive' :
                      log.level === 'WARNING' ? 'text-yellow-500' :
                      log.level === 'DEBUG' ? 'text-muted-foreground' :
                      'text-foreground'
                    }>
                      [{log.level}]
                    </span>
                    {' '}
                    <span>{log.message}</span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </>
            ) : (
              <div className="text-muted-foreground">
                {isActive ? 'Waiting for logs...' : 'No logs available'}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
