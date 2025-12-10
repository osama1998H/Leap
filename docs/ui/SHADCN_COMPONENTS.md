# shadcn/ui Component Usage Plan

This document specifies which shadcn/ui components to install and how they map to UI features.

---

## Installation

### Initial Setup

```bash
# Create Vite project
npm create vite@latest leap-ui -- --template react-ts
cd leap-ui

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Initialize shadcn/ui
npx shadcn-ui@latest init

# Select options:
# - Style: Default
# - Base color: Slate
# - CSS variables: Yes
# - tailwind.config.js location: tailwind.config.js
# - components.json location: ./
# - utils location: @/lib/utils
# - React Server Components: No
```

---

## Core Components Needed

### Phase 1 (MVP) - Essential Components

| Component | Install Command | Usage |
|-----------|-----------------|-------|
| **button** | `npx shadcn-ui add button` | Actions throughout the app |
| **input** | `npx shadcn-ui add input` | Form fields |
| **label** | `npx shadcn-ui add label` | Form labels |
| **card** | `npx shadcn-ui add card` | Metric cards, run cards |
| **form** | `npx shadcn-ui add form` | React Hook Form integration |
| **select** | `npx shadcn-ui add select` | Dropdowns (timeframe, symbol) |
| **checkbox** | `npx shadcn-ui add checkbox` | Boolean options |
| **switch** | `npx shadcn-ui add switch` | Toggle options (realistic mode) |
| **slider** | `npx shadcn-ui add slider` | Hyperparameter ranges |
| **tabs** | `npx shadcn-ui add tabs` | Page sections, config tabs |
| **table** | `npx shadcn-ui add table` | Trade history, results lists |
| **badge** | `npx shadcn-ui add badge` | Status indicators |
| **progress** | `npx shadcn-ui add progress` | Training progress bars |
| **skeleton** | `npx shadcn-ui add skeleton` | Loading states |
| **toast** | `npx shadcn-ui add toast` | Notifications |
| **tooltip** | `npx shadcn-ui add tooltip` | Help text on hover |
| **separator** | `npx shadcn-ui add separator` | Visual dividers |

**Batch install (Phase 1):**
```bash
npx shadcn-ui add button input label card form select checkbox switch slider tabs table badge progress skeleton toast tooltip separator
```

### Phase 2 - Monitoring Components

| Component | Install Command | Usage |
|-----------|-----------------|-------|
| **scroll-area** | `npx shadcn-ui add scroll-area` | Log viewer scrolling |
| **alert** | `npx shadcn-ui add alert` | Error/warning messages |
| **alert-dialog** | `npx shadcn-ui add alert-dialog` | Confirmation dialogs |

**Batch install (Phase 2):**
```bash
npx shadcn-ui add scroll-area alert alert-dialog
```

### Phase 3 - Comparison Components

| Component | Install Command | Usage |
|-----------|-----------------|-------|
| **dialog** | `npx shadcn-ui add dialog` | Modal dialogs |
| **sheet** | `npx shadcn-ui add sheet` | Side panels |
| **dropdown-menu** | `npx shadcn-ui add dropdown-menu` | Context menus |
| **collapsible** | `npx shadcn-ui add collapsible` | Expandable sections |
| **data-table** | (Custom, uses table) | Advanced tables with sorting |

**Batch install (Phase 3):**
```bash
npx shadcn-ui add dialog sheet dropdown-menu collapsible
```

### Phase 4 - Advanced Components

| Component | Install Command | Usage |
|-----------|-----------------|-------|
| **command** | `npx shadcn-ui add command` | Command palette (⌘K) |
| **popover** | `npx shadcn-ui add popover` | Rich tooltips, pickers |
| **calendar** | `npx shadcn-ui add calendar` | Date range selection |
| **avatar** | `npx shadcn-ui add avatar` | User profile (future) |
| **navigation-menu** | `npx shadcn-ui add navigation-menu` | Main navigation |
| **menubar** | `npx shadcn-ui add menubar` | App menu bar |
| **toggle** | `npx shadcn-ui add toggle` | Toggle buttons |
| **toggle-group** | `npx shadcn-ui add toggle-group` | Radio-style toggles |

**Batch install (Phase 4):**
```bash
npx shadcn-ui add command popover calendar avatar navigation-menu menubar toggle toggle-group
```

---

## Component-to-Feature Mapping

### Dashboard

```tsx
// Components used
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"

// Dashboard layout example
<div className="grid grid-cols-4 gap-4">
  <Card>
    <CardHeader>
      <CardTitle>Active Runs</CardTitle>
    </CardHeader>
    <CardContent>
      <Badge variant="default">2 Running</Badge>
    </CardContent>
  </Card>
  ...
</div>
```

### Training Configuration Form

```tsx
// Components used
import { Form, FormField, FormItem, FormLabel, FormControl } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"

// Form structure
<Form {...form}>
  <Tabs defaultValue="data">
    <TabsList>
      <TabsTrigger value="data">Data</TabsTrigger>
      <TabsTrigger value="transformer">Transformer</TabsTrigger>
      <TabsTrigger value="ppo">PPO</TabsTrigger>
    </TabsList>

    <TabsContent value="data">
      <FormField name="symbol" render={...} />
      <FormField name="timeframe" render={...} />
    </TabsContent>
    ...
  </Tabs>
</Form>
```

### Backtest Results

```tsx
// Components used
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"

// Metrics grid
<div className="grid grid-cols-4 gap-4">
  <MetricCard title="Total Return" value="+12.4%" trend="up" />
  <MetricCard title="Sharpe Ratio" value="1.85" trend="up" />
  ...
</div>

// Trade table
<Table>
  <TableHeader>
    <TableRow>
      <TableHead>Entry Time</TableHead>
      <TableHead>Direction</TableHead>
      <TableHead>P&L</TableHead>
      <TableHead>Status</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    {trades.map(trade => (
      <TableRow key={trade.id}>
        <TableCell>{formatDate(trade.entryTime)}</TableCell>
        <TableCell>
          <Badge variant={trade.direction === 'long' ? 'default' : 'secondary'}>
            {trade.direction.toUpperCase()}
          </Badge>
        </TableCell>
        <TableCell className={trade.pnl > 0 ? 'text-green-600' : 'text-red-600'}>
          {formatCurrency(trade.pnl)}
        </TableCell>
        <TableCell>{trade.status}</TableCell>
      </TableRow>
    ))}
  </TableBody>
</Table>
```

### Log Viewer

```tsx
// Components used
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Select } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"

// Log viewer structure
<div className="flex flex-col h-full">
  <div className="flex gap-2 p-2">
    <Input placeholder="Search logs..." />
    <Select>
      <SelectTrigger>
        <SelectValue placeholder="Log level" />
      </SelectTrigger>
      ...
    </Select>
  </div>

  <ScrollArea className="flex-1 font-mono text-sm">
    {logs.map(log => (
      <div key={log.id} className="flex gap-2 py-1 hover:bg-muted">
        <span className="text-muted-foreground">{log.timestamp}</span>
        <Badge variant={getLevelVariant(log.level)}>{log.level}</Badge>
        <span>{log.message}</span>
      </div>
    ))}
  </ScrollArea>
</div>
```

---

## Custom Components to Build

These components extend or compose shadcn/ui primitives:

### MetricsCard

```tsx
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { TrendingUp, TrendingDown, Minus } from "lucide-react"

interface MetricsCardProps {
  title: string
  value: string | number
  format?: 'percent' | 'currency' | 'number' | 'ratio'
  trend?: 'up' | 'down' | 'neutral'
  description?: string
  className?: string
}

export function MetricsCard({
  title,
  value,
  format = 'number',
  trend,
  description,
  className
}: MetricsCardProps) {
  const formattedValue = formatValue(value, format)
  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus

  return (
    <Card className={cn("", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2">
          <span className="text-2xl font-bold">{formattedValue}</span>
          {trend && (
            <TrendIcon className={cn(
              "h-4 w-4",
              trend === 'up' && "text-green-500",
              trend === 'down' && "text-red-500",
              trend === 'neutral' && "text-gray-500"
            )} />
          )}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
      </CardContent>
    </Card>
  )
}
```

### MetricsLineChart

Using Recharts with shadcn theme colors:

```tsx
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

interface MetricsLineChartProps {
  data: { x: number | Date; train: number; val?: number }[]
  title: string
  xLabel?: string
  yLabel?: string
  height?: number
}

export function MetricsLineChart({
  data,
  title,
  xLabel,
  yLabel,
  height = 300
}: MetricsLineChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data}>
            <XAxis
              dataKey="x"
              label={{ value: xLabel, position: 'bottom' }}
              className="text-muted-foreground"
            />
            <YAxis
              label={{ value: yLabel, angle: -90, position: 'left' }}
              className="text-muted-foreground"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: 'var(--radius)',
              }}
            />
            <Line
              type="monotone"
              dataKey="train"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={false}
            />
            {data[0]?.val !== undefined && (
              <Line
                type="monotone"
                dataKey="val"
                stroke="hsl(var(--destructive))"
                strokeWidth={2}
                dot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
```

### LogStream

Virtual scrolling for performance:

```tsx
import { useVirtualizer } from '@tanstack/react-virtual'
import { useRef } from 'react'
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

interface LogEntry {
  id: string
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  message: string
}

interface LogStreamProps {
  logs: LogEntry[]
  autoScroll?: boolean
}

export function LogStream({ logs, autoScroll = true }: LogStreamProps) {
  const parentRef = useRef<HTMLDivElement>(null)

  const rowVirtualizer = useVirtualizer({
    count: logs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 24,
    overscan: 5,
  })

  return (
    <div
      ref={parentRef}
      className="h-[400px] overflow-auto font-mono text-sm"
    >
      <div
        style={{
          height: `${rowVirtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const log = logs[virtualRow.index]
          return (
            <div
              key={log.id}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualRow.size}px`,
                transform: `translateY(${virtualRow.start}px)`,
              }}
              className="flex gap-2 items-center px-2 hover:bg-muted"
            >
              <span className="text-muted-foreground w-20 flex-shrink-0">
                {log.timestamp}
              </span>
              <Badge variant={getLogLevelVariant(log.level)} className="w-16">
                {log.level}
              </Badge>
              <span className="truncate">{log.message}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
```

### RunCard

```tsx
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { Play, Pause, Square, ExternalLink } from "lucide-react"

interface RunCardProps {
  type: 'training' | 'backtest'
  id: string
  symbol: string
  timeframe: string
  status: 'running' | 'completed' | 'failed' | 'paused'
  progress?: number
  metrics?: {
    loss?: number
    return?: number
    sharpe?: number
  }
  onViewDetails?: () => void
  onPause?: () => void
  onStop?: () => void
}

export function RunCard({
  type,
  id,
  symbol,
  timeframe,
  status,
  progress,
  metrics,
  onViewDetails,
  onPause,
  onStop
}: RunCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">{symbol} {timeframe}</CardTitle>
          <Badge variant={getStatusVariant(status)}>{status}</Badge>
        </div>
        <p className="text-sm text-muted-foreground">
          {type === 'training' ? 'Training' : 'Backtest'} #{id.slice(0, 8)}
        </p>
      </CardHeader>

      <CardContent>
        {status === 'running' && progress !== undefined && (
          <div className="space-y-2">
            <Progress value={progress} />
            <p className="text-sm text-muted-foreground">{progress}% complete</p>
          </div>
        )}

        {metrics && (
          <div className="grid grid-cols-3 gap-2 mt-2">
            {metrics.loss !== undefined && (
              <div className="text-center">
                <p className="text-xs text-muted-foreground">Loss</p>
                <p className="font-semibold">{metrics.loss.toFixed(4)}</p>
              </div>
            )}
            {metrics.return !== undefined && (
              <div className="text-center">
                <p className="text-xs text-muted-foreground">Return</p>
                <p className="font-semibold text-green-600">
                  {metrics.return > 0 ? '+' : ''}{(metrics.return * 100).toFixed(1)}%
                </p>
              </div>
            )}
            {metrics.sharpe !== undefined && (
              <div className="text-center">
                <p className="text-xs text-muted-foreground">Sharpe</p>
                <p className="font-semibold">{metrics.sharpe.toFixed(2)}</p>
              </div>
            )}
          </div>
        )}
      </CardContent>

      <CardFooter className="gap-2">
        <Button variant="outline" size="sm" onClick={onViewDetails}>
          <ExternalLink className="h-4 w-4 mr-1" />
          Details
        </Button>
        {status === 'running' && (
          <>
            <Button variant="outline" size="sm" onClick={onPause}>
              <Pause className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={onStop}>
              <Square className="h-4 w-4" />
            </Button>
          </>
        )}
      </CardFooter>
    </Card>
  )
}
```

---

## Theming

### Color Scheme

Extend the default shadcn theme with trading-specific colors:

```css
/* globals.css additions */
@layer base {
  :root {
    /* Trading-specific colors */
    --profit: 142.1 76.2% 36.3%;      /* Green for profit */
    --loss: 0 84.2% 60.2%;            /* Red for loss */
    --neutral: 220 13% 46%;           /* Gray for neutral */

    /* Chart colors */
    --chart-1: 220 70% 50%;           /* Primary line */
    --chart-2: 160 60% 45%;           /* Secondary line */
    --chart-3: 30 80% 55%;            /* Tertiary line */
    --chart-4: 280 65% 60%;           /* Quaternary line */
  }

  .dark {
    --profit: 142.1 70.6% 45.3%;
    --loss: 0 72.2% 50.6%;
    --neutral: 217.9 10.6% 64.9%;

    --chart-1: 220 70% 60%;
    --chart-2: 160 60% 55%;
    --chart-3: 30 80% 65%;
    --chart-4: 280 65% 70%;
  }
}
```

### Utility Classes

```css
/* globals.css additions */
@layer utilities {
  .text-profit {
    color: hsl(var(--profit));
  }
  .text-loss {
    color: hsl(var(--loss));
  }
  .bg-profit {
    background-color: hsl(var(--profit));
  }
  .bg-loss {
    background-color: hsl(var(--loss));
  }
}
```

### Dark Mode

Use shadcn's built-in dark mode with next-themes:

```bash
npm install next-themes
```

```tsx
// theme-provider.tsx
import { ThemeProvider as NextThemesProvider } from "next-themes"

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="system" enableSystem>
      {children}
    </NextThemesProvider>
  )
}

// theme-toggle.tsx
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
    >
      <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
      <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
      <span className="sr-only">Toggle theme</span>
    </Button>
  )
}
```

---

## Additional Libraries

### Charts (Recharts)

```bash
npm install recharts
```

### Form Validation (Zod + React Hook Form)

Already included with shadcn/ui form component:
```bash
npm install zod @hookform/resolvers
```

### Icons (Lucide)

Already included with shadcn/ui:
```bash
npm install lucide-react
```

### Virtual Scrolling (for logs)

```bash
npm install @tanstack/react-virtual
```

### Date Formatting

```bash
npm install date-fns
```
