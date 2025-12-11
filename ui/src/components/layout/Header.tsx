import { Link } from 'react-router-dom'
import { Activity, Sun, Moon, Monitor } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useQuery } from '@tanstack/react-query'
import { systemApi } from '@/lib/api'
import { useTheme } from '@/hooks/use-theme'

export default function Header() {
  const { theme, setTheme, resolvedTheme } = useTheme()

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

  const cycleTheme = () => {
    if (theme === 'light') {
      setTheme('dark')
    } else if (theme === 'dark') {
      setTheme('system')
    } else {
      setTheme('light')
    }
  }

  const getThemeIcon = () => {
    if (theme === 'system') {
      return <Monitor className="h-4 w-4" />
    }
    return resolvedTheme === 'dark' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <Link to="/" className="flex items-center space-x-2">
          <Activity className="h-6 w-6 text-primary" />
          <span className="font-bold text-xl">LEAP</span>
        </Link>
        <div className="flex flex-1 items-center justify-end space-x-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={cycleTheme}
            title={`Theme: ${theme}`}
          >
            {getThemeIcon()}
          </Button>
          {getStatusBadge()}
          <span className="text-sm text-muted-foreground">
            v{health?.version || '1.0.0'}
          </span>
        </div>
      </div>
    </header>
  )
}
