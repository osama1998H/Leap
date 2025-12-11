import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Plus, Brain, TestTube, Activity, Square, Eye } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { trainingApi, backtestApi, systemApi } from '@/lib/api'
import { formatPercent, formatDate } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'

export default function Dashboard() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const { data: trainingJobs } = useQuery({
    queryKey: ['training', 'jobs'],
    queryFn: () => trainingApi.list({ limit: 5 }),
    refetchInterval: 5000,
  })

  const stopJob = useMutation({
    mutationFn: trainingApi.stop,
    onSuccess: (data) => {
      toast({
        title: 'Training Stopped',
        description: `Job ${data.jobId} has been stopped.`,
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

  const { data: backtestJobs } = useQuery({
    queryKey: ['backtest', 'jobs'],
    queryFn: () => backtestApi.jobs({ limit: 10 }),
    refetchInterval: 5000,
  })

  const { data: backtestResults } = useQuery({
    queryKey: ['backtest', 'results'],
    queryFn: () => backtestApi.list({ limit: 5 }),
  })

  const { data: metrics } = useQuery({
    queryKey: ['system', 'metrics'],
    queryFn: systemApi.metrics,
    refetchInterval: 10000,
  })

  const activeTrainingJobs = trainingJobs?.jobs?.filter(j => j.status === 'running') || []
  const activeBacktestJobs = backtestJobs?.jobs?.filter(j => j.status === 'running') || []
  const totalActiveJobs = activeTrainingJobs.length + activeBacktestJobs.length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">Leap Trading System Overview</p>
        </div>
        <div className="flex gap-2">
          <Button asChild>
            <Link to="/training">
              <Plus className="mr-2 h-4 w-4" />
              New Training
            </Link>
          </Button>
          <Button variant="outline" asChild>
            <Link to="/backtest">
              <Plus className="mr-2 h-4 w-4" />
              New Backtest
            </Link>
          </Button>
        </div>
      </div>

      {/* System Status Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics?.cpu?.percent?.toFixed(1) ?? 0}%</div>
            <Progress value={metrics?.cpu?.percent ?? 0} className="mt-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics?.memory?.percent?.toFixed(1) ?? 0}%</div>
            <Progress value={metrics?.memory?.percent ?? 0} className="mt-2" />
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">GPU</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics?.gpu?.available ? 'Available' : 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {metrics?.gpu?.name || 'No GPU detected'}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalActiveJobs}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {totalActiveJobs > 0 ? `${activeTrainingJobs.length} training, ${activeBacktestJobs.length} backtest` : 'No active jobs'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Active Jobs */}
      {totalActiveJobs > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Active Jobs</CardTitle>
            <CardDescription>Currently running training and backtest jobs</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Training Jobs */}
              {activeTrainingJobs.map((job) => (
                <div key={job.jobId} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-4">
                    <Brain className="h-8 w-8 text-primary" />
                    <div>
                      <p className="font-medium">{job.symbols?.join(', ')} {job.timeframe}</p>
                      <p className="text-sm text-muted-foreground">
                        Phase: {job.phase ?? 'Training'} | Epoch: {job.progress?.currentEpoch ?? 0}/{job.progress?.totalEpochs ?? 100}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-32">
                      <Progress value={job.progress?.percent ?? 0} />
                      <p className="text-xs text-muted-foreground text-center mt-1">
                        {job.progress?.percent ?? 0}%
                      </p>
                    </div>
                    {job.metrics?.trainLoss && (
                      <div className="text-right">
                        <p className="text-sm font-medium">Loss: {job.metrics.trainLoss.toFixed(4)}</p>
                        {job.metrics.valLoss && (
                          <p className="text-xs text-muted-foreground">Val: {job.metrics.valLoss.toFixed(4)}</p>
                        )}
                      </div>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      asChild
                    >
                      <Link to={`/training/${job.jobId}`}>
                        <Eye className="h-4 w-4 mr-1" />
                        View
                      </Link>
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => stopJob.mutate(job.jobId)}
                      disabled={stopJob.isPending}
                    >
                      <Square className="h-4 w-4 mr-1" />
                      Stop
                    </Button>
                  </div>
                </div>
              ))}
              {/* Backtest Jobs */}
              {activeBacktestJobs.map((job) => (
                <div key={job.jobId} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-4">
                    <TestTube className="h-8 w-8 text-blue-500" />
                    <div>
                      <p className="font-medium">{job.symbol} {job.timeframe}</p>
                      <p className="text-sm text-muted-foreground">
                        Backtest {job.progress?.currentStep ? `- ${job.progress.currentStep}` : 'running...'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-32">
                      <Progress value={job.progress?.percent ?? 0} />
                      <p className="text-xs text-muted-foreground text-center mt-1">
                        {job.progress?.percent ?? 0}%
                      </p>
                    </div>
                    <Badge variant="secondary">Running</Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Results */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Results</CardTitle>
          <CardDescription>Latest backtest and training results</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Type</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Return</TableHead>
                <TableHead>Sharpe</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Date</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {backtestResults?.results?.map((result) => (
                <TableRow key={result.resultId}>
                  <TableCell>
                    <Badge variant="outline">
                      <TestTube className="h-3 w-3 mr-1" />
                      Backtest
                    </Badge>
                  </TableCell>
                  <TableCell className="font-medium">{result.symbol}</TableCell>
                  <TableCell className={result.summary.totalReturn >= 0 ? 'text-profit' : 'text-loss'}>
                    {formatPercent(result.summary.totalReturn)}
                  </TableCell>
                  <TableCell>{result.summary.sharpeRatio.toFixed(2)}</TableCell>
                  <TableCell>
                    <Badge variant="success">Completed</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {formatDate(result.completedAt)}
                  </TableCell>
                </TableRow>
              ))}
              {(!backtestResults?.results || backtestResults.results.length === 0) && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground py-8">
                    No recent results. Run a backtest to see results here.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
