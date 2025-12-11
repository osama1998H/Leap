import { useState, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Save, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/hooks/use-toast'
import { configApi } from '@/lib/api'

export default function ConfigPage() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const { data: config, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: configApi.get,
  })

  const { data: templates } = useQuery({
    queryKey: ['config', 'templates'],
    queryFn: configApi.templates,
  })

  const [formData, setFormData] = useState<any>({})

  useEffect(() => {
    if (config) {
      setFormData(config)
    }
  }, [config])

  const updateConfig = useMutation({
    mutationFn: configApi.update,
    onSuccess: () => {
      toast({
        title: 'Configuration saved',
        description: 'Your configuration has been updated.',
      })
      queryClient.invalidateQueries({ queryKey: ['config'] })
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to save configuration',
        description: error.message,
        variant: 'destructive',
      })
    },
  })

  const handleSave = () => {
    updateConfig.mutate(formData)
  }

  const handleReset = () => {
    if (config) {
      setFormData(config)
    }
  }

  const updateField = (section: string, field: string, value: any) => {
    setFormData((prev: any) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value,
      },
    }))
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Configuration</h1>
          <p className="text-muted-foreground">Manage system settings and templates</p>
        </div>
        <div className="flex gap-2">
          <Select>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Load Template" />
            </SelectTrigger>
            <SelectContent>
              {templates?.templates?.map((template) => (
                <SelectItem key={template.id} value={template.id}>
                  {template.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={handleReset}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset
          </Button>
          <Button onClick={handleSave} disabled={updateConfig.isPending}>
            <Save className="mr-2 h-4 w-4" />
            {updateConfig.isPending ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="data" className="space-y-4">
        <TabsList>
          <TabsTrigger value="data">Data</TabsTrigger>
          <TabsTrigger value="transformer">Transformer</TabsTrigger>
          <TabsTrigger value="ppo">PPO</TabsTrigger>
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="backtest">Backtest</TabsTrigger>
        </TabsList>

        <TabsContent value="data">
          <Card>
            <CardHeader>
              <CardTitle>Data Configuration</CardTitle>
              <CardDescription>Configure data loading and feature settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Primary Timeframe</Label>
                  <Select
                    value={formData.data?.primaryTimeframe || '1h'}
                    onValueChange={(value) => updateField('data', 'primaryTimeframe', value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1m">1 Minute</SelectItem>
                      <SelectItem value="5m">5 Minutes</SelectItem>
                      <SelectItem value="15m">15 Minutes</SelectItem>
                      <SelectItem value="30m">30 Minutes</SelectItem>
                      <SelectItem value="1h">1 Hour</SelectItem>
                      <SelectItem value="4h">4 Hours</SelectItem>
                      <SelectItem value="1d">1 Day</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Lookback Window</Label>
                  <Input
                    type="number"
                    value={formData.data?.lookbackWindow || 120}
                    onChange={(e) => updateField('data', 'lookbackWindow', parseInt(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Prediction Horizon</Label>
                  <Input
                    type="number"
                    value={formData.data?.predictionHorizon || 12}
                    onChange={(e) => updateField('data', 'predictionHorizon', parseInt(e.target.value))}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="transformer">
          <Card>
            <CardHeader>
              <CardTitle>Transformer Configuration</CardTitle>
              <CardDescription>Configure the price prediction model architecture</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Model Dimension (d_model)</Label>
                  <Select
                    value={String(formData.transformer?.dModel || 128)}
                    onValueChange={(value) => updateField('transformer', 'dModel', parseInt(value))}
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
                    value={String(formData.transformer?.nHeads || 8)}
                    onValueChange={(value) => updateField('transformer', 'nHeads', parseInt(value))}
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

                <div className="space-y-2">
                  <Label>Encoder Layers</Label>
                  <Input
                    type="number"
                    value={formData.transformer?.nEncoderLayers || 4}
                    onChange={(e) => updateField('transformer', 'nEncoderLayers', parseInt(e.target.value))}
                    min={1}
                    max={12}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Dropout</Label>
                  <Input
                    type="number"
                    value={formData.transformer?.dropout || 0.1}
                    onChange={(e) => updateField('transformer', 'dropout', parseFloat(e.target.value))}
                    step={0.05}
                    min={0}
                    max={0.5}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Learning Rate</Label>
                  <Input
                    type="text"
                    value={formData.transformer?.learningRate || 0.0001}
                    onChange={(e) => updateField('transformer', 'learningRate', parseFloat(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Epochs</Label>
                  <Input
                    type="number"
                    value={formData.transformer?.epochs || 100}
                    onChange={(e) => updateField('transformer', 'epochs', parseInt(e.target.value))}
                    min={1}
                    max={1000}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ppo">
          <Card>
            <CardHeader>
              <CardTitle>PPO Agent Configuration</CardTitle>
              <CardDescription>Configure the reinforcement learning agent</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Learning Rate</Label>
                  <Input
                    type="text"
                    value={formData.ppo?.learningRate || 0.0003}
                    onChange={(e) => updateField('ppo', 'learningRate', parseFloat(e.target.value))}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Gamma (Discount Factor)</Label>
                  <Input
                    type="number"
                    value={formData.ppo?.gamma || 0.99}
                    onChange={(e) => updateField('ppo', 'gamma', parseFloat(e.target.value))}
                    step={0.01}
                    min={0.9}
                    max={0.999}
                  />
                </div>

                <div className="space-y-2">
                  <Label>GAE Lambda</Label>
                  <Input
                    type="number"
                    value={formData.ppo?.gaeLambda || 0.95}
                    onChange={(e) => updateField('ppo', 'gaeLambda', parseFloat(e.target.value))}
                    step={0.01}
                    min={0.9}
                    max={0.999}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Clip Epsilon</Label>
                  <Input
                    type="number"
                    value={formData.ppo?.clipEpsilon || 0.2}
                    onChange={(e) => updateField('ppo', 'clipEpsilon', parseFloat(e.target.value))}
                    step={0.05}
                    min={0.1}
                    max={0.3}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Total Timesteps</Label>
                  <Input
                    type="number"
                    value={formData.ppo?.totalTimesteps || 1000000}
                    onChange={(e) => updateField('ppo', 'totalTimesteps', parseInt(e.target.value))}
                    step={10000}
                    min={10000}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk">
          <Card>
            <CardHeader>
              <CardTitle>Risk Configuration</CardTitle>
              <CardDescription>Configure risk management parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Max Position Size (%)</Label>
                  <Input
                    type="number"
                    value={(formData.risk?.maxPositionSize || 0.02) * 100}
                    onChange={(e) => updateField('risk', 'maxPositionSize', parseFloat(e.target.value) / 100)}
                    step={0.5}
                    min={0.5}
                    max={10}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Max Daily Loss (%)</Label>
                  <Input
                    type="number"
                    value={(formData.risk?.maxDailyLoss || 0.05) * 100}
                    onChange={(e) => updateField('risk', 'maxDailyLoss', parseFloat(e.target.value) / 100)}
                    step={1}
                    min={1}
                    max={20}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Default Stop Loss (pips)</Label>
                  <Input
                    type="number"
                    value={formData.risk?.defaultStopLossPips || 50}
                    onChange={(e) => updateField('risk', 'defaultStopLossPips', parseInt(e.target.value))}
                    min={10}
                    max={200}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Default Take Profit (pips)</Label>
                  <Input
                    type="number"
                    value={formData.risk?.defaultTakeProfitPips || 100}
                    onChange={(e) => updateField('risk', 'defaultTakeProfitPips', parseInt(e.target.value))}
                    min={10}
                    max={500}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="backtest">
          <Card>
            <CardHeader>
              <CardTitle>Backtest Configuration</CardTitle>
              <CardDescription>Configure backtest simulation parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Initial Balance ($)</Label>
                  <Input
                    type="number"
                    value={formData.backtest?.initialBalance || 10000}
                    onChange={(e) => updateField('backtest', 'initialBalance', parseInt(e.target.value))}
                    min={1000}
                    step={1000}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Leverage</Label>
                  <Input
                    type="number"
                    value={formData.backtest?.leverage || 100}
                    onChange={(e) => updateField('backtest', 'leverage', parseInt(e.target.value))}
                    min={1}
                    max={500}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Spread (pips)</Label>
                  <Input
                    type="number"
                    value={formData.backtest?.spreadPips || 1.5}
                    onChange={(e) => updateField('backtest', 'spreadPips', parseFloat(e.target.value))}
                    step={0.1}
                    min={0}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Commission per Lot ($)</Label>
                  <Input
                    type="number"
                    value={formData.backtest?.commissionPerLot || 7}
                    onChange={(e) => updateField('backtest', 'commissionPerLot', parseFloat(e.target.value))}
                    step={0.5}
                    min={0}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
