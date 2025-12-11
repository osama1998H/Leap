import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Brain,
  TestTube,
  Settings,
  ScrollText,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Training', href: '/training', icon: Brain },
  { name: 'Backtest', href: '/backtest', icon: TestTube },
  { name: 'Config', href: '/config', icon: Settings },
  { name: 'Logs', href: '/logs', icon: ScrollText },
]

export default function Sidebar() {
  const location = useLocation()

  return (
    <aside className="hidden md:flex w-64 flex-col border-r bg-background">
      <nav className="flex flex-col gap-1 p-4">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-secondary text-secondary-foreground'
                  : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'
              )}
            >
              <item.icon className="h-4 w-4" />
              {item.name}
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
