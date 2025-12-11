import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Play, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/hooks/use-toast'
import { trainingApi, TrainingConfig } from '@/lib/api'

const SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF']
const TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

interface ExtendedTrainingConfig extends TrainingConfig {
  dModel: number
  nHeads: number
  gamma: number
  clipEpsilon: number
  device: string
}

export default function TrainingPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const [config, setConfig] = useState<ExtendedTrainingConfig>({
    symbols: ['EURUSD'],
    timeframe: '1h',
    multiTimeframe: false,
    bars: 50000,
    epochs: 100,
    timesteps: 1000000,
    modelDir: './saved_models',
    dModel: 128,
    nHeads: 8,
    gamma: 0.99,
    clipEpsilon: 0.2,
    device: 'auto',
  })

  const startTraining = useMutation({
    mutationFn: trainingApi.start,
    onSuccess: (data) => {
      toast({
        title: 'Training Started',
        description: `Job ${data.jobId} has been started.`,
      })
      queryClient.invalidateQueries({ queryKey: ['training'] })
      navigate(`/training/${data.jobId}`)
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to start training',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    startTraining.mutate(config)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">New Training Run</h1>
        <p className="text-muted-foreground">Configure and launch a new training job</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-6">
            <Tabs defaultValue="data" className="space-y-4">
              <TabsList>
                <TabsTrigger value="data">Data Settings</TabsTrigger>
                <TabsTrigger value="transformer">Transformer</TabsTrigger>
                <TabsTrigger value="ppo">PPO Agent</TabsTrigger>
                <TabsTrigger value="advanced">Advanced</TabsTrigger>
              </TabsList>

              <TabsContent value="data">
                <Card>
                  <CardHeader>
                    <CardTitle>Data Settings</CardTitle>
                    <CardDescription>Configure the trading data for training</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label>Symbols</Label>
                      <p className="text-xs text-muted-foreground mb-2">Click to select/deselect symbols for training</p>
                      <div className="flex flex-wrap gap-2">
                        {SYMBOLS.map((symbol) => {
                          const isSelected = config.symbols.includes(symbol)
                          return (
                            <Badge
                              key={symbol}
                              variant={isSelected ? 'default' : 'outline'}
                              className="cursor-pointer select-none"
                              onClick={() => {
                                if (isSelected) {
                                  // Don't allow deselecting if it's the only symbol
                                  if (config.symbols.length > 1) {
                                    setConfig({ ...config, symbols: config.symbols.filter(s => s !== symbol) })
                                  }
                                } else {
                                  setConfig({ ...config, symbols: [...config.symbols, symbol] })
                                }
                              }}
                            >
                              {symbol}
                              {isSelected && config.symbols.length > 1 && (
                                <X className="ml-1 h-3 w-3" />
                              )}
                            </Badge>
                          )
                        })}
                      </div>
                      {config.symbols.length > 1 && (
                        <p className="text-xs text-muted-foreground mt-2">
                          {config.symbols.length} symbols selected - multi-symbol training enabled
                        </p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="timeframe">Primary Timeframe</Label>
                      <Select
                        value={config.timeframe}
                        onValueChange={(value) => setConfig({ ...config, timeframe: value })}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select timeframe" />
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

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="multi-timeframe"
                        checked={config.multiTimeframe}
                        onCheckedChange={(checked) => setConfig({ ...config, multiTimeframe: checked })}
                      />
                      <Label htmlFor="multi-timeframe">Enable Multi-Timeframe Features</Label>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="bars">Bars to Load</Label>
                      <Input
                        id="bars"
                        type="number"
                        value={config.bars}
                        onChange={(e) => setConfig({ ...config, bars: parseInt(e.target.value) || 50000 })}
                        min={1000}
                        max={500000}
                      />
                      <p className="text-xs text-muted-foreground">Number of historical bars (1,000 - 500,000)</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="transformer">
                <Card>
                  <CardHeader>
                    <CardTitle>Transformer Settings</CardTitle>
                    <CardDescription>Configure the price prediction model</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="epochs">Training Epochs</Label>
                      <Input
                        id="epochs"
                        type="number"
                        value={config.epochs}
                        onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 100 })}
                        min={1}
                        max={1000}
                      />
                      <p className="text-xs text-muted-foreground">Number of training epochs (1 - 1,000)</p>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Model Dimension</Label>
                        <Select
                          value={String(config.dModel)}
                          onValueChange={(value) => setConfig({ ...config, dModel: parseInt(value) })}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="32">32</SelectItem>
                            <SelectItem value="64">64</SelectItem>
                            <SelectItem value="128">128</SelectItem>
                            <SelectItem value="256">256</SelectItem>
                            <SelectItem value="512">512</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label>Attention Heads</Label>
                        <Select
                          value={String(config.nHeads)}
                          onValueChange={(value) => setConfig({ ...config, nHeads: parseInt(value) })}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="1">1</SelectItem>
                            <SelectItem value="2">2</SelectItem>
                            <SelectItem value="4">4</SelectItem>
                            <SelectItem value="8">8</SelectItem>
                            <SelectItem value="16">16</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="ppo">
                <Card>
                  <CardHeader>
                    <CardTitle>PPO Agent Settings</CardTitle>
                    <CardDescription>Configure the reinforcement learning agent</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="timesteps">Total Timesteps</Label>
                      <Input
                        id="timesteps"
                        type="number"
                        value={config.timesteps}
                        onChange={(e) => setConfig({ ...config, timesteps: parseInt(e.target.value) || 1000000 })}
                        min={10000}
                        max={10000000}
                        step={10000}
                      />
                      <p className="text-xs text-muted-foreground">Total environment timesteps (10K - 10M)</p>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label>Gamma (Discount)</Label>
                        <Input
                          type="number"
                          value={config.gamma}
                          onChange={(e) => setConfig({ ...config, gamma: parseFloat(e.target.value) || 0.99 })}
                          step={0.01}
                          min={0.9}
                          max={0.999}
                        />
                      </div>

                      <div className="space-y-2">
                        <Label>Clip Epsilon</Label>
                        <Input
                          type="number"
                          value={config.clipEpsilon}
                          onChange={(e) => setConfig({ ...config, clipEpsilon: parseFloat(e.target.value) || 0.2 })}
                          step={0.05}
                          min={0.1}
                          max={0.3}
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="advanced">
                <Card>
                  <CardHeader>
                    <CardTitle>Advanced Settings</CardTitle>
                    <CardDescription>Additional configuration options</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="modelDir">Model Directory</Label>
                      <Input
                        id="modelDir"
                        value={config.modelDir}
                        onChange={(e) => setConfig({ ...config, modelDir: e.target.value })}
                      />
                      <p className="text-xs text-muted-foreground">Directory to save trained models</p>
                    </div>

                    <div className="space-y-2">
                      <Label>Device</Label>
                      <Select
                        value={config.device}
                        onValueChange={(value) => setConfig({ ...config, device: value })}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">Auto (recommended)</SelectItem>
                          <SelectItem value="cuda">CUDA (GPU)</SelectItem>
                          <SelectItem value="cpu">CPU</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Summary Sidebar */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Configuration Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Symbol(s)</span>
                  <span className="font-medium">{config.symbols.join(', ')}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Timeframe</span>
                  <span className="font-medium">{config.timeframe}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Bars</span>
                  <span className="font-medium">{config.bars.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Epochs</span>
                  <span className="font-medium">{config.epochs}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Timesteps</span>
                  <span className="font-medium">{(config.timesteps / 1000000).toFixed(1)}M</span>
                </div>
              </CardContent>
            </Card>

            <Button type="submit" size="lg" className="w-full" disabled={startTraining.isPending}>
              <Play className="mr-2 h-4 w-4" />
              {startTraining.isPending ? 'Starting...' : 'Start Training'}
            </Button>
          </div>
        </div>
      </form>
    </div>
  )
}
