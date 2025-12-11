import { Link } from 'react-router-dom'
import { Activity } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { useQuery } from '@tanstack/react-query'
import { systemApi } from '@/lib/api'

export default function Header() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: systemApi.health,
    refetchInterval: 30000,
  })

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="mr-4 flex">
          <Link to="/" className="mr-6 flex items-center space-x-2">
            <Activity className="h-6 w-6 text-primary" />
            <span className="font-bold text-xl">LEAP</span>
          </Link>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link to="/" className="transition-colors hover:text-foreground/80 text-foreground/60">
              Dashboard
            </Link>
            <Link to="/training" className="transition-colors hover:text-foreground/80 text-foreground/60">
              Training
            </Link>
            <Link to="/backtest" className="transition-colors hover:text-foreground/80 text-foreground/60">
              Backtest
            </Link>
            <Link to="/config" className="transition-colors hover:text-foreground/80 text-foreground/60">
              Config
            </Link>
            <Link to="/logs" className="transition-colors hover:text-foreground/80 text-foreground/60">
              Logs
            </Link>
          </nav>
        </div>
        <div className="flex flex-1 items-center justify-end space-x-4">
          <Badge variant={health?.status === 'healthy' ? 'success' : 'destructive'}>
            {health?.status === 'healthy' ? 'Online' : 'Offline'}
          </Badge>
          <span className="text-sm text-muted-foreground">
            v{health?.version || '1.0.0'}
          </span>
        </div>
      </div>
    </header>
  )
}
