import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowLeft, Square, Brain, Clock, Activity } from 'lucide-react'
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
import { useState, useEffect } from 'react'

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

  // Track loss history for chart
  const [lossHistory, setLossHistory] = useState<LossDataPoint[]>([])

  const { data: job, isLoading, isError } = useQuery({
    queryKey: ['training', 'job', id],
    queryFn: () => trainingApi.get(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      // Stop polling if job is completed or failed
      const status = query.state.data?.status
      if (status === 'completed' || status === 'failed' || status === 'stopped') {
        return false
      }
      return 2000 // Poll every 2 seconds while running
    },
  })

  const { data: logs } = useQuery({
    queryKey: ['training', 'logs', id],
    queryFn: () => trainingApi.logs(id!),
    enabled: !!id,
    refetchInterval: () => {
      const status = job?.status
      if (status === 'completed' || status === 'failed' || status === 'stopped') {
        return false
      }
      return 3000
    },
  })

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

  // Update loss history when job metrics change
  useEffect(() => {
    if (job?.metrics?.trainLoss !== undefined && job?.progress?.currentEpoch !== undefined) {
      setLossHistory(prev => {
        const epoch = job.progress?.currentEpoch ?? 0
        // Check if we already have this epoch
        const existingIndex = prev.findIndex(p => p.epoch === epoch)
        const newPoint: LossDataPoint = {
          epoch,
          trainLoss: job.metrics?.trainLoss ?? null,
          valLoss: job.metrics?.valLoss ?? null,
        }

        if (existingIndex >= 0) {
          // Update existing point
          const updated = [...prev]
          updated[existingIndex] = newPoint
          return updated
        } else {
          // Add new point
          return [...prev, newPoint].sort((a, b) => a.epoch - b.epoch)
        }
      })
    }
  }, [job?.metrics?.trainLoss, job?.metrics?.valLoss, job?.progress?.currentEpoch])

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
      case 'pending':
        return <Badge variant="outline">Pending</Badge>
      default:
        return <Badge variant="outline">{status}</Badge>
    }
  }

  const formatDuration = (createdAt: string) => {
    const start = new Date(createdAt)
    const now = new Date()
    const diffMs = now.getTime() - start.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)

    if (diffHours > 0) {
      return `${diffHours}h ${diffMins % 60}m`
    }
    return `${diffMins}m`
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

  const isRunning = job.status === 'running'

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
              {getStatusBadge(job.status)}
            </div>
            <p className="text-sm text-muted-foreground">
              Training Job {job.jobId.slice(0, 8)}...
            </p>
          </div>
        </div>
        {isRunning && (
          <Button
            variant="destructive"
            onClick={() => stopJob.mutate(job.jobId)}
            disabled={stopJob.isPending}
          >
            <Square className="mr-2 h-4 w-4" />
            Stop Training
          </Button>
        )}
      </div>

      {/* Progress & Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Progress</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{job.progress?.percent ?? 0}%</div>
            <Progress value={job.progress?.percent ?? 0} className="mt-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Epoch</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {job.progress?.currentEpoch ?? 0} / {job.progress?.totalEpochs ?? 100}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Phase: {job.phase ?? 'Transformer'}
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
              {job.metrics?.trainLoss?.toFixed(6) ?? '-'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Val: {job.metrics?.valLoss?.toFixed(6) ?? '-'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Duration</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatDuration(job.createdAt)}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Started: {new Date(job.createdAt).toLocaleTimeString()}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Loss Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Training Loss</CardTitle>
          <CardDescription>Train and validation loss over epochs</CardDescription>
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
                />
                <Line
                  type="monotone"
                  dataKey="valLoss"
                  stroke="hsl(var(--destructive))"
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[300px] flex items-center justify-center text-muted-foreground">
              {isRunning ? 'Waiting for training data...' : 'No loss data available'}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Training Logs */}
      <Card>
        <CardHeader>
          <CardTitle>Training Logs</CardTitle>
          <CardDescription>Recent training output</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted rounded-lg p-4 font-mono text-sm max-h-[300px] overflow-y-auto">
            {logs?.logs && logs.logs.length > 0 ? (
              logs.logs.slice(-50).map((log, index) => (
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
              ))
            ) : (
              <div className="text-muted-foreground">
                {isRunning ? 'Waiting for logs...' : 'No logs available'}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
