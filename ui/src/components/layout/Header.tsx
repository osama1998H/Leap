import { Link } from 'react-router-dom'
import { Activity } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { useQuery } from '@tanstack/react-query'
import { systemApi } from '@/lib/api'

export default function Header() {
  const { data: health, isLoading, isError } = useQuery({
    queryKey: ['health'],
    queryFn: systemApi.health,
    refetchInterval: 30000,
  })

  const getStatusBadge = () => {
    if (isLoading) {
      return <Badge variant="secondary">Connecting...</Badge>
    }
    if (isError) {
      return <Badge variant="destructive">Error</Badge>
    }
    if (health?.status === 'healthy') {
      return <Badge variant="success">Online</Badge>
    }
    return <Badge variant="destructive">Offline</Badge>
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <Link to="/" className="flex items-center space-x-2">
          <Activity className="h-6 w-6 text-primary" />
          <span className="font-bold text-xl">LEAP</span>
        </Link>
        <div className="flex flex-1 items-center justify-end space-x-4">
          {getStatusBadge()}
          <span className="text-sm text-muted-foreground">
            v{health?.version || '1.0.0'}
          </span>
        </div>
      </div>
    </header>
  )
}
