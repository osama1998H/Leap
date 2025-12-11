"""Logs service."""

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..schemas.logs import LogFileInfo, LogLine

logger = logging.getLogger(__name__)


class LogsService:
    """Service for managing application logs."""

    def __init__(self):
        self.logs_dir = settings.LOGS_DIR

    def list_log_files(self) -> list[LogFileInfo]:
        """List available log files."""
        files = []

        if not self.logs_dir.exists():
            return files

        for log_file in sorted(self.logs_dir.glob("*.log"), reverse=True):
            stat = log_file.stat()
            files.append(LogFileInfo(
                name=log_file.name,
                size=stat.st_size,
                modifiedAt=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            ))

        return files

    def get_log_file(
        self,
        filename: str,
        level: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[LogLine], int]:
        """Get log file contents."""
        log_path = self.logs_dir / filename

        if not log_path.exists():
            return [], 0

        lines = []
        total_lines = 0

        with open(log_path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            total_lines = len(all_lines)

            for i, line in enumerate(all_lines):
                line = line.rstrip()
                if not line:
                    continue

                parsed = self._parse_log_line(i + 1, line)

                # Filter by level
                if level and parsed.level != level:
                    continue

                # Filter by search
                if search and search.lower() not in line.lower():
                    continue

                lines.append(parsed)

        # Apply pagination
        lines = lines[offset : offset + limit]

        return lines, total_lines

    def _parse_log_line(self, line_number: int, line: str) -> LogLine:
        """Parse a log line into structured format."""
        # Common log format: YYYY-MM-DD HH:MM:SS,mmm LEVEL logger - message
        # or: YYYY-MM-DD HH:MM:SS LEVEL message
        pattern = r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d{3})?)\s+(\w+)\s+(?:(\w+)\s+-\s+)?(.*)$"
        match = re.match(pattern, line)

        if match:
            timestamp = match.group(1).replace(" ", "T").replace(",", ".") + "Z"
            level = match.group(2)
            logger_name = match.group(3)
            message = match.group(4)

            return LogLine(
                number=line_number,
                timestamp=timestamp,
                level=level,
                logger=logger_name,
                message=message,
            )

        # Fallback for unparseable lines
        return LogLine(
            number=line_number,
            timestamp=None,
            level=None,
            logger=None,
            message=line,
        )


logs_service = LogsService()
