import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Download, TrendingUp, TrendingDown, Activity } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { backtestApi } from '@/lib/api'
import { formatPercent, formatCurrency, formatDate, formatNumber } from '@/lib/utils'

export default function BacktestResultPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const { data: result, isLoading } = useQuery({
    queryKey: ['backtest', 'result', id],
    queryFn: () => backtestApi.get(id!),
    enabled: !!id,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Result not found</p>
        <Button variant="link" onClick={() => navigate('/backtest')}>
          Back to Backtest
        </Button>
      </div>
    )
  }

  const metrics = result.metrics

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => navigate('/backtest')}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              {result.symbol} {result.timeframe} Backtest
            </h1>
            <p className="text-muted-foreground">
              Completed {formatDate(result.completedAt)}
            </p>
          </div>
        </div>
        <Button variant="outline">
          <Download className="mr-2 h-4 w-4" />
          Export Results
        </Button>
      </div>

      {/* Summary Metrics */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Return</CardTitle>
            {(metrics?.returns?.totalReturn || 0) >= 0 ? (
              <TrendingUp className="h-4 w-4 text-profit" />
            ) : (
              <TrendingDown className="h-4 w-4 text-loss" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${(metrics?.returns?.totalReturn || 0) >= 0 ? 'text-profit' : 'text-loss'}`}>
              {formatPercent(metrics?.returns?.totalReturn || result.summary.totalReturn)}
            </div>
            <p className="text-xs text-muted-foreground">
              Annualized: {formatPercent(metrics?.returns?.annualizedReturn || 0)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(metrics?.riskAdjusted?.sharpeRatio || result.summary.sharpeRatio)}
            </div>
            <p className="text-xs text-muted-foreground">
              Sortino: {formatNumber(metrics?.riskAdjusted?.sortinoRatio || 0)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
            <TrendingDown className="h-4 w-4 text-loss" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-loss">
              {formatPercent(metrics?.risk?.maxDrawdown || result.summary.maxDrawdown)}
            </div>
            <p className="text-xs text-muted-foreground">
              Volatility: {formatPercent(metrics?.risk?.volatility || 0)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {((metrics?.trade?.winRate || result.summary.winRate) * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {metrics?.trade?.totalTrades || result.summary.totalTrades} trades
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Performance Metrics */}
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Annualized Return</p>
                  <p className="text-lg font-semibold">{formatPercent(metrics?.returns?.annualizedReturn || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Volatility</p>
                  <p className="text-lg font-semibold">{formatPercent(metrics?.risk?.volatility || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Sortino Ratio</p>
                  <p className="text-lg font-semibold">{formatNumber(metrics?.riskAdjusted?.sortinoRatio || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Calmar Ratio</p>
                  <p className="text-lg font-semibold">{formatNumber(metrics?.riskAdjusted?.calmarRatio || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">VaR (95%)</p>
                  <p className="text-lg font-semibold">{formatPercent(metrics?.risk?.var_95 || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">CVaR (95%)</p>
                  <p className="text-lg font-semibold">{formatPercent(metrics?.risk?.cvar_95 || 0)}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Trade Statistics */}
        <Card>
          <CardHeader>
            <CardTitle>Trade Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Total Trades</p>
                  <p className="text-lg font-semibold">{metrics?.trade?.totalTrades || result.summary.totalTrades}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Profit Factor</p>
                  <p className="text-lg font-semibold">{formatNumber(metrics?.trade?.profitFactor || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Winning Trades</p>
                  <p className="text-lg font-semibold text-profit">{metrics?.trade?.winningTrades || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Losing Trades</p>
                  <p className="text-lg font-semibold text-loss">{metrics?.trade?.losingTrades || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg Winner</p>
                  <p className="text-lg font-semibold text-profit">{formatCurrency(metrics?.trade?.avgWinner || 0)}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Avg Loser</p>
                  <p className="text-lg font-semibold text-loss">{formatCurrency(metrics?.trade?.avgLoser || 0)}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Trade History */}
      {result.trades && result.trades.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Trade History</CardTitle>
            <CardDescription>Individual trade breakdown</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Entry Time</TableHead>
                  <TableHead>Direction</TableHead>
                  <TableHead>Entry Price</TableHead>
                  <TableHead>Exit Price</TableHead>
                  <TableHead>P&L</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.trades.slice(0, 20).map((trade) => (
                  <TableRow key={trade.id}>
                    <TableCell>{formatDate(trade.entryTime)}</TableCell>
                    <TableCell>
                      <Badge variant={trade.direction === 'long' ? 'default' : 'secondary'}>
                        {trade.direction.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{trade.entryPrice.toFixed(5)}</TableCell>
                    <TableCell>{trade.exitPrice.toFixed(5)}</TableCell>
                    <TableCell className={trade.pnl >= 0 ? 'text-profit' : 'text-loss'}>
                      {formatCurrency(trade.pnl)}
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{trade.status}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {result.trades.length > 20 && (
              <p className="text-sm text-muted-foreground text-center mt-4">
                Showing 20 of {result.trades.length} trades
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
