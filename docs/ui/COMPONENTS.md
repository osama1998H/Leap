# Component Specification

## Pages

### Dashboard (`/`)

**Purpose**: Overview of system status, recent runs, and quick actions.

**Layout**:
- Header with navigation
- Quick action buttons (New Training, New Backtest)
- Active jobs panel (real-time updates)
- Recent results panel
- System status indicators

**Data Requirements**:
- Active training/backtest jobs (WebSocket)
- Recent backtest results (last 10)
- System health status
- Model availability status

**Components Used**:
- `RunCard` - Display active/recent runs
- `StatusBadge` - System health indicators
- `QuickActionButton` - Launch new operations
- `MetricsSummary` - High-level metrics

---

### Training Configuration (`/training`)

**Purpose**: Configure and launch training runs for Transformer predictor and PPO agent.

**Layout**:
- Configuration form (collapsible sections)
- Preset selector
- Summary panel
- Launch button

**Form Fields** (mapped from CLI options):

| Section | Field | Type | Default | CLI Flag | Validation |
|---------|-------|------|---------|----------|------------|
| **Data** | Symbol(s) | Multi-select | EURUSD | `--symbols` | Required, valid forex pairs |
| | Primary Timeframe | Select | 1h | `--timeframe` | 1m, 5m, 15m, 30m, 1h, 4h, 1d |
| | Multi-Timeframe | Toggle | false | `--multi-timeframe` | - |
| | Additional Timeframes | Multi-select | - | Config | Shown if multi-TF enabled |
| | Bars to Load | Number | 50000 | `--bars` | 1000-500000 |
| **Transformer** | Epochs | Number | 100 | `--epochs` | 1-1000 |
| | Learning Rate | Number | 1e-4 | Config | 1e-6 to 1e-2 |
| | Model Dimension | Number | 128 | Config | 32, 64, 128, 256, 512 |
| | Attention Heads | Number | 8 | Config | 1, 2, 4, 8, 16 |
| | Encoder Layers | Number | 4 | Config | 1-12 |
| | Batch Size | Number | 64 | Config | 16, 32, 64, 128, 256 |
| | Early Stop Patience | Number | 15 | Config | 1-50 |
| **PPO** | Timesteps | Number | 1000000 | `--timesteps` | 10000-10000000 |
| | Learning Rate | Number | 3e-4 | Config | 1e-6 to 1e-2 |
| | Gamma | Slider | 0.99 | Config | 0.9-0.999 |
| | Clip Epsilon | Slider | 0.2 | Config | 0.1-0.3 |
| | Entropy Coefficient | Number | 0.01 | Config | 0.001-0.1 |
| **Output** | Model Directory | Text | ./saved_models | `--model-dir` | Valid path |
| | MLflow Experiment | Text | leap-trading | Config | - |
| **Advanced** | Device | Select | auto | Config | auto, cpu, cuda |
| | Seed | Number | 42 | Config | 0-999999 |

**Components**:
- `TrainingConfigForm` - Main form container
- `SymbolSelector` - Multi-select for trading pairs
- `TimeframeSelector` - Timeframe dropdown
- `HyperparameterSlider` - Range inputs for hyperparams
- `ConfigPresetSelector` - Load/save presets
- `ConfigSummary` - Review before launch

---

### Training Monitor (`/training/[id]`)

**Purpose**: Real-time visualization of training progress.

**Layout**:
- Header with job info and controls (pause/stop)
- Phase indicator (Transformer / PPO)
- Loss charts (train/validation)
- Progress bars
- Log stream panel
- Resource usage (optional)

**Real-time Data** (WebSocket):
- Current epoch/timestep
- Training loss
- Validation loss
- Learning rate schedule
- Estimated time remaining

**Components**:
- `TrainingHeader` - Job info, controls
- `PhaseIndicator` - Current training phase
- `LossChart` - Dual-axis loss visualization
- `EpochProgress` - Progress bar with ETA
- `LogStream` - Live log output
- `ResourceMonitor` - GPU/CPU usage

**Update Frequency**: 1-2 seconds

---

### Backtest Configuration (`/backtest`)

**Purpose**: Configure and run backtests on historical data.

**Form Fields**:

| Section | Field | Type | Default | CLI Flag |
|---------|-------|------|---------|----------|
| **Data** | Symbol | Select | EURUSD | `--symbol` |
| | Timeframe | Select | 1h | `--timeframe` |
| | Bars | Number | 50000 | `--bars` |
| **Model** | Model Directory | Text/Browser | ./saved_models | `--model-dir` |
| **Settings** | Realistic Mode | Toggle | false | `--realistic` |
| | Monte Carlo | Toggle | false | `--monte-carlo` |
| | MC Simulations | Number | 1000 | Config (shown if MC on) |
| **Transaction Costs** | Spread (pips) | Number | 1.5 | Config |
| | Commission/Lot | Number | 7.0 | Config |
| | Slippage (pips) | Number | 1.0 | Config |
| **Risk** | Initial Balance | Number | 10000 | Config |
| | Leverage | Number | 100 | Config |
| | Risk per Trade | Percentage | 2% | Config |

**Components**:
- `BacktestConfigForm` - Main form
- `ModelSelector` - Browse available models
- `RealisticModeToggle` - With explanation tooltip
- `TransactionCostPanel` - Cost configuration
- `RiskSettingsPanel` - Risk parameters

---

### Backtest Results (`/backtest/[id]`)

**Purpose**: Display comprehensive backtest results and analysis.

**Layout**:
- Summary metrics cards (top row)
- Equity curve chart
- Drawdown chart
- Trade statistics panel
- Trade table (filterable)
- Monte Carlo results (if enabled)
- Regime analysis

**Metrics Displayed**:

| Category | Metrics |
|----------|---------|
| **Returns** | Total Return, Annualized Return, CAGR |
| **Risk** | Max Drawdown, Volatility, VaR 95%, CVaR 95% |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| **Trade Stats** | Win Rate, Profit Factor, Avg Winner/Loser, Total Trades |
| **Distribution** | Skewness, Kurtosis, Tail Ratio |

**Components**:
- `MetricsGrid` - Summary metric cards
- `EquityCurveChart` - Interactive equity visualization
- `DrawdownChart` - Drawdown over time
- `TradeStatisticsPanel` - Trade breakdown
- `TradeTable` - Sortable, filterable trade list
- `MonteCarloPanel` - Distribution charts
- `RegimeAnalysisPanel` - Performance by market regime
- `PerformanceReport` - Downloadable summary

---

### Walk-Forward Analysis (`/walkforward`)

**Purpose**: Run and visualize walk-forward optimization.

**Configuration**:
- Same data settings as backtest
- Train window (days)
- Test window (days)
- Step size (days)

**Results Display**:
- Per-fold results table
- Aggregated metrics
- Consistency metrics (profitable folds ratio)
- Visual timeline of train/test windows

**Components**:
- `WalkForwardConfigForm` - Configuration
- `FoldResultsTable` - Per-fold breakdown
- `AggregateMetricsPanel` - Statistical summary
- `WalkForwardTimeline` - Visual representation of folds
- `ConsistencyIndicator` - Strategy robustness score

---

### Experiment Browser (`/experiments`)

**Purpose**: Browse and compare MLflow-tracked experiments.

**Features**:
- List all experiment runs
- Filter by symbol, timeframe, date
- Compare selected runs
- View run details and artifacts

**Components**:
- `ExperimentTable` - Sortable run list
- `RunFilters` - Filter controls
- `RunComparisonView` - Side-by-side comparison
- `ArtifactBrowser` - Browse run artifacts

---

### Configuration Editor (`/config`)

**Purpose**: Edit and manage system configuration.

**Sections** (accordion/tabs):
1. Data Configuration
2. Transformer Configuration
3. PPO Configuration
4. Risk Configuration
5. Backtest Configuration
6. Logging Configuration
7. Auto-Trader Configuration
8. MLflow Configuration

**Features**:
- Load/save JSON files
- Preset templates
- Validation with error display
- Reset to defaults
- Export/import

**Components**:
- `ConfigEditor` - Main editor container
- `ConfigSection` - Collapsible section
- `ConfigField` - Smart field renderer
- `JsonEditor` - Raw JSON editing mode
- `ValidationErrors` - Error display
- `TemplateManager` - Save/load templates

---

### Log Viewer (`/logs`)

**Purpose**: View and search application logs.

**Features**:
- File selector
- Real-time streaming
- Level filtering (DEBUG, INFO, WARNING, ERROR)
- Text search
- Auto-scroll toggle
- Download logs

**Components**:
- `LogFileSelector` - Choose log file
- `LogStream` - Virtual scrolling log display
- `LogFilters` - Level and text filters
- `LogEntry` - Formatted log line

---

## Shared Components

### MetricsCard

**Purpose**: Display a single metric with optional trend indicator.

```typescript
interface MetricsCardProps {
  title: string;
  value: number | string;
  format?: 'percent' | 'currency' | 'number' | 'ratio';
  trend?: 'up' | 'down' | 'neutral';
  tooltip?: string;
  size?: 'sm' | 'md' | 'lg';
}
```

**Usage**:
```tsx
<MetricsCard
  title="Sharpe Ratio"
  value={1.85}
  format="ratio"
  trend="up"
  tooltip="Risk-adjusted return measure"
/>
```

---

### StatusBadge

**Purpose**: Display status indicators with color coding.

```typescript
interface StatusBadgeProps {
  status: 'running' | 'completed' | 'failed' | 'pending' | 'paused';
  label?: string;
  pulse?: boolean;  // Animate for active states
}
```

---

### RunCard

**Purpose**: Summary card for a training or backtest run.

```typescript
interface RunCardProps {
  type: 'training' | 'backtest';
  id: string;
  symbol: string;
  timeframe: string;
  status: RunStatus;
  startTime: Date;
  metrics?: {
    loss?: number;
    sharpe?: number;
    return?: number;
  };
  onClick?: () => void;
}
```

---

### LogViewer

**Purpose**: Virtual-scrolling log display with filtering.

```typescript
interface LogViewerProps {
  logs: LogEntry[];
  streaming?: boolean;
  onFilter?: (level: LogLevel) => void;
  onSearch?: (query: string) => void;
  maxLines?: number;
  autoScroll?: boolean;
}
```

---

### FileTree

**Purpose**: Browse file system (models, results, logs).

```typescript
interface FileTreeProps {
  root: string;
  onSelect: (path: string) => void;
  filter?: RegExp;
  showHidden?: boolean;
}
```

---

### EquityCurveChart

**Purpose**: Interactive equity curve visualization.

```typescript
interface EquityCurveChartProps {
  data: { timestamp: Date; equity: number }[];
  benchmark?: { timestamp: Date; value: number }[];
  showDrawdown?: boolean;
  height?: number;
  interactive?: boolean;  // Zoom, pan
}
```

**Features**:
- Time range selection
- Zoom and pan
- Hover tooltips
- Optional benchmark overlay
- Drawdown shading

---

### LossChart

**Purpose**: Training loss visualization.

```typescript
interface LossChartProps {
  trainLoss: { epoch: number; loss: number }[];
  valLoss?: { epoch: number; loss: number }[];
  height?: number;
  streaming?: boolean;  // Append new data
}
```

---

### ConfigForm

**Purpose**: Dynamic form generator from configuration schema.

```typescript
interface ConfigFormProps {
  schema: ConfigSchema;
  values: Record<string, any>;
  onChange: (values: Record<string, any>) => void;
  onSubmit?: () => void;
  sections?: string[];  // Which sections to show
}
```

---

### TradeTable

**Purpose**: Paginated, sortable trade history table.

```typescript
interface TradeTableProps {
  trades: Trade[];
  pageSize?: number;
  sortable?: boolean;
  filterable?: boolean;
  onTradeClick?: (trade: Trade) => void;
}
```

**Columns**:
- Entry Time
- Exit Time
- Direction (Long/Short)
- Entry Price
- Exit Price
- Size
- P&L ($)
- P&L (%)
- Status (closed, stopped, TP)
- Duration

---

## Component Styling Guidelines

### Color Palette

| Purpose | Light Mode | Dark Mode |
|---------|------------|-----------|
| Profit/Positive | `green-600` | `green-400` |
| Loss/Negative | `red-600` | `red-400` |
| Neutral | `gray-600` | `gray-400` |
| Warning | `amber-600` | `amber-400` |
| Primary Action | `blue-600` | `blue-400` |

### Spacing

Use Tailwind spacing scale consistently:
- Card padding: `p-4` or `p-6`
- Section gaps: `gap-4` or `gap-6`
- Form field gaps: `gap-2` or `gap-4`

### Typography

- Page titles: `text-2xl font-semibold`
- Section titles: `text-lg font-medium`
- Metric values: `text-2xl font-bold` (large), `text-lg font-semibold` (small)
- Labels: `text-sm text-muted-foreground`
