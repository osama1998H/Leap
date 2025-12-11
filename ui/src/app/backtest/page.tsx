import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'
import { backtestApi, BacktestConfig, modelsApi } from '@/lib/api'
import { formatPercent, formatDate } from '@/lib/utils'

const SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
const TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

export default function BacktestPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const [config, setConfig] = useState<BacktestConfig>({
    symbol: 'EURUSD',
    timeframe: '1h',
    bars: 50000,
    modelDir: './saved_models',
    realisticMode: true,
    monteCarlo: true,
  })

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: modelsApi.list,
  })

  const { data: results } = useQuery({
    queryKey: ['backtest', 'results'],
    queryFn: () => backtestApi.list({ limit: 10 }),
  })

  const runBacktest = useMutation({
    mutationFn: backtestApi.run,
    onSuccess: (data) => {
      toast({
        title: 'Backtest Started',
        description: `Job ${data.jobId} has been started.`,
      })
      queryClient.invalidateQueries({ queryKey: ['backtest'] })
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to run backtest',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    runBacktest.mutate(config)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Backtest</h1>
        <p className="text-muted-foreground">Run backtests on historical data</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          {/* Configuration Form */}
          <form onSubmit={handleSubmit}>
            <Card>
              <CardHeader>
                <CardTitle>Backtest Configuration</CardTitle>
                <CardDescription>Configure backtest parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>Symbol</Label>
                    <Select
                      value={config.symbol}
                      onValueChange={(value) => setConfig({ ...config, symbol: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {SYMBOLS.map((symbol) => (
                          <SelectItem key={symbol} value={symbol}>
                            {symbol}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Timeframe</Label>
                    <Select
                      value={config.timeframe}
                      onValueChange={(value) => setConfig({ ...config, timeframe: value })}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {TIMEFRAMES.map((tf) => (
                          <SelectItem key={tf} value={tf}>
                            {tf}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Bars to Test</Label>
                  <Input
                    type="number"
                    value={config.bars}
                    onChange={(e) => setConfig({ ...config, bars: parseInt(e.target.value) || 50000 })}
                    min={1000}
                    max={500000}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Model Directory</Label>
                  <Select
                    value={config.modelDir}
                    onValueChange={(value) => setConfig({ ...config, modelDir: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="./saved_models">./saved_models (default)</SelectItem>
                      {models?.models?.map((model) => (
                        <SelectItem key={model.directory} value={model.directory}>
                          {model.directory}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-4 pt-4 border-t">
                  <h4 className="font-medium">Backtest Settings</h4>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Realistic Mode</Label>
                      <p className="text-xs text-muted-foreground">Apply trading limits and position caps</p>
                    </div>
                    <Switch
                      checked={config.realisticMode}
                      onCheckedChange={(checked) => setConfig({ ...config, realisticMode: checked })}
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Monte Carlo Analysis</Label>
                      <p className="text-xs text-muted-foreground">Run bootstrap simulations for risk analysis</p>
                    </div>
                    <Switch
                      checked={config.monteCarlo}
                      onCheckedChange={(checked) => setConfig({ ...config, monteCarlo: checked })}
                    />
                  </div>
                </div>

                <Button type="submit" size="lg" className="w-full" disabled={runBacktest.isPending}>
                  <Play className="mr-2 h-4 w-4" />
                  {runBacktest.isPending ? 'Running...' : 'Run Backtest'}
                </Button>
              </CardContent>
            </Card>
          </form>

          {/* Recent Results */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Results</CardTitle>
              <CardDescription>Previous backtest results</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Return</TableHead>
                    <TableHead>Sharpe</TableHead>
                    <TableHead>Drawdown</TableHead>
                    <TableHead>Win Rate</TableHead>
                    <TableHead>Date</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results?.results?.map((result) => (
                    <TableRow
                      key={result.resultId}
                      className="cursor-pointer"
                      onClick={() => navigate(`/backtest/${result.resultId}`)}
                    >
                      <TableCell className="font-medium">{result.symbol}</TableCell>
                      <TableCell className={result.summary.totalReturn >= 0 ? 'text-profit' : 'text-loss'}>
                        {formatPercent(result.summary.totalReturn)}
                      </TableCell>
                      <TableCell>{result.summary.sharpeRatio.toFixed(2)}</TableCell>
                      <TableCell className="text-loss">
                        {formatPercent(result.summary.maxDrawdown)}
                      </TableCell>
                      <TableCell>{(result.summary.winRate * 100).toFixed(1)}%</TableCell>
                      <TableCell className="text-muted-foreground">
                        {formatDate(result.completedAt)}
                      </TableCell>
                    </TableRow>
                  ))}
                  {(!results?.results || results.results.length === 0) && (
                    <TableRow>
                      <TableCell colSpan={6} className="text-center text-muted-foreground py-8">
                        No backtest results yet
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </div>

        {/* Model Info Sidebar */}
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Available Models</CardTitle>
              <CardDescription>Trained models for backtesting</CardDescription>
            </CardHeader>
            <CardContent>
              {models?.models && models.models.length > 0 ? (
                <div className="space-y-3">
                  {models.models.map((model) => (
                    <div key={model.directory} className="p-3 border rounded-lg">
                      <p className="font-medium text-sm">{model.directory}</p>
                      <div className="flex gap-2 mt-2">
                        <Badge variant={model.predictor.exists ? 'success' : 'secondary'}>
                          Predictor
                        </Badge>
                        <Badge variant={model.agent.exists ? 'success' : 'secondary'}>
                          Agent
                        </Badge>
                      </div>
                      {model.metadata.symbol && (
                        <p className="text-xs text-muted-foreground mt-2">
                          {model.metadata.symbol} {model.metadata.timeframe}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">
                  No trained models found. Run a training job first.
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
