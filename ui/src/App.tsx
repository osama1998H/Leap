import { Routes, Route } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import Layout from '@/components/layout/Layout'
import Dashboard from '@/app/dashboard/page'
import TrainingPage from '@/app/training/page'
import TrainingMonitorPage from '@/app/training/[id]/page'
import BacktestPage from '@/app/backtest/page'
import BacktestResultPage from '@/app/backtest/[id]/page'
import LogsPage from '@/app/logs/page'
import ConfigPage from '@/app/config/page'

function App() {
  return (
    <>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/training/:id" element={<TrainingMonitorPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="/backtest/:id" element={<BacktestResultPage />} />
          <Route path="/logs" element={<LogsPage />} />
          <Route path="/config" element={<ConfigPage />} />
        </Routes>
      </Layout>
      <Toaster />
    </>
  )
}

export default App
