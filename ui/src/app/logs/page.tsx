import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, FileText } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { logsApi } from '@/lib/api'
import { cn } from '@/lib/utils'

const LOG_LEVELS = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR']

export default function LogsPage() {
  const [selectedFile, setSelectedFile] = useState<string>('')
  const [level, setLevel] = useState<string>('ALL')
  const [search, setSearch] = useState('')

  const { data: files } = useQuery({
    queryKey: ['logs', 'files'],
    queryFn: logsApi.files,
  })

  const { data: logContent, isLoading } = useQuery({
    queryKey: ['logs', 'content', selectedFile, level, search],
    queryFn: () =>
      logsApi.get(selectedFile, {
        level: level !== 'ALL' ? level : undefined,
        search: search || undefined,
        limit: 500,
      }),
    enabled: !!selectedFile,
    refetchInterval: 5000,
  })

  const getLevelColor = (logLevel?: string) => {
    switch (logLevel?.toUpperCase()) {
      case 'ERROR':
        return 'destructive'
      case 'WARNING':
        return 'warning'
      case 'INFO':
        return 'default'
      case 'DEBUG':
        return 'secondary'
      default:
        return 'outline'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Log Viewer</h1>
          <p className="text-muted-foreground">View and search application logs</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-4">
        {/* File List Sidebar */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Log Files</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {files?.files?.map((file) => (
                <button
                  key={file.name}
                  onClick={() => setSelectedFile(file.name)}
                  className={cn(
                    'w-full flex items-center gap-2 p-2 rounded-lg text-left text-sm transition-colors',
                    selectedFile === file.name
                      ? 'bg-secondary text-secondary-foreground'
                      : 'hover:bg-secondary/50'
                  )}
                >
                  <FileText className="h-4 w-4 flex-shrink-0" />
                  <div className="overflow-hidden">
                    <p className="font-medium truncate">{file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </button>
              ))}
              {(!files?.files || files.files.length === 0) && (
                <p className="text-sm text-muted-foreground py-4 text-center">
                  No log files found
                </p>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Log Content */}
        <div className="lg:col-span-3 space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="p-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search logs..."
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      className="pl-8"
                    />
                  </div>
                </div>
                <Select value={level} onValueChange={setLevel}>
                  <SelectTrigger className="w-32">
                    <SelectValue placeholder="Level" />
                  </SelectTrigger>
                  <SelectContent>
                    {LOG_LEVELS.map((l) => (
                      <SelectItem key={l} value={l}>
                        {l}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Log Output */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between py-3">
              <CardTitle className="text-lg">
                {selectedFile || 'Select a log file'}
              </CardTitle>
              {logContent && (
                <span className="text-sm text-muted-foreground">
                  {logContent.lines.length} / {logContent.totalLines} lines
                </span>
              )}
            </CardHeader>
            <CardContent>
              {!selectedFile ? (
                <div className="text-center py-12 text-muted-foreground">
                  Select a log file from the sidebar
                </div>
              ) : isLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                </div>
              ) : (
                <div className="font-mono text-xs overflow-auto max-h-[600px] bg-muted/30 rounded-lg p-4">
                  {logContent?.lines?.map((line) => (
                    <div
                      key={line.number}
                      className="flex gap-2 py-0.5 hover:bg-muted/50 rounded"
                    >
                      <span className="text-muted-foreground w-12 text-right flex-shrink-0">
                        {line.number}
                      </span>
                      {line.timestamp && (
                        <span className="text-muted-foreground w-24 flex-shrink-0">
                          {line.timestamp.split('T')[1]?.replace('Z', '') || line.timestamp}
                        </span>
                      )}
                      {line.level && (
                        <Badge
                          variant={getLevelColor(line.level) as any}
                          className="h-5 px-1 text-xs flex-shrink-0"
                        >
                          {line.level.padEnd(7)}
                        </Badge>
                      )}
                      <span className="flex-1 break-all">{line.message}</span>
                    </div>
                  ))}
                  {logContent?.lines?.length === 0 && (
                    <p className="text-center text-muted-foreground py-8">
                      No log entries match the current filters
                    </p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
