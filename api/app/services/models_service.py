"""Models service."""

import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from ..core.config import settings
from ..schemas.models import (
    AgentInfo,
    ModelDetailData,
    ModelDetailMetadata,
    ModelInfo,
    ModelMetadata,
    PredictorInfo,
)

logger = logging.getLogger(__name__)


class ModelsService:
    """Service for managing saved models."""

    def __init__(self):
        self.models_dir = settings.SAVED_MODELS_DIR

    def _validate_path(self, directory: str) -> Optional[Path]:
        """Validate and resolve path, preventing directory traversal attacks.

        Returns the resolved path if valid, None if path traversal detected.
        """
        model_path = Path(directory)
        if not model_path.is_absolute():
            model_path = settings.PROJECT_ROOT / directory

        # Resolve to absolute path (resolves .., symlinks, etc.)
        try:
            resolved_path = model_path.resolve()
        except (OSError, ValueError):
            return None

        # Verify the resolved path is within PROJECT_ROOT
        project_root = settings.PROJECT_ROOT.resolve()
        try:
            resolved_path.relative_to(project_root)
        except ValueError:
            # Path is outside PROJECT_ROOT - potential traversal attack
            logger.warning(f"Path traversal attempt detected: {directory}")
            return None

        return resolved_path

    def list_models(self) -> list[ModelInfo]:
        """List all saved models."""
        models = []

        # Check main saved_models directory
        if self.models_dir.exists():
            model_info = self._get_model_info(self.models_dir)
            if model_info:
                models.append(model_info)

        # Check subdirectories
        for subdir in self.models_dir.iterdir() if self.models_dir.exists() else []:
            if subdir.is_dir():
                model_info = self._get_model_info(subdir)
                if model_info:
                    models.append(model_info)

        return models

    def get_model_detail(self, directory: str) -> Optional[ModelDetailData]:
        """Get detailed model information."""
        model_path = self._validate_path(directory)
        if not model_path or not model_path.exists():
            return None

        # Get files
        files = []
        for f in model_path.iterdir():
            if f.is_file():
                files.append(f.name)

        # Get metadata
        metadata = self._load_metadata(model_path)
        config = self._load_config(model_path)

        return ModelDetailData(
            directory=str(model_path),
            files=sorted(files),
            metadata=ModelDetailMetadata(
                predictor=metadata.get("predictor"),
                agent=metadata.get("agent"),
                trainedSymbol=metadata.get("symbol"),
                trainedTimeframe=metadata.get("timeframe"),
                createdAt=metadata.get("created_at"),
            ),
            config=config,
        )

    def create_download(self, directory: str) -> Optional[BytesIO]:
        """Create a ZIP file of the model directory."""
        model_path = self._validate_path(directory)
        if not model_path or not model_path.exists():
            return None

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(model_path)
                    zf.write(file_path, arcname)

        buffer.seek(0)
        return buffer

    def _get_model_info(self, path: Path) -> Optional[ModelInfo]:
        """Get model info for a directory."""
        predictor_exists = (path / "predictor.pt").exists()
        agent_exists = (path / "agent.pt").exists()

        if not predictor_exists and not agent_exists:
            return None

        metadata = self._load_metadata(path)

        predictor_meta = metadata.get("predictor", {})
        agent_meta = metadata.get("agent", {})

        # Get file modification times
        predictor_mtime = None
        agent_mtime = None

        if predictor_exists:
            stat = (path / "predictor.pt").stat()
            predictor_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        if agent_exists:
            stat = (path / "agent.pt").stat()
            agent_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        return ModelInfo(
            directory=str(path),
            predictor=PredictorInfo(
                exists=predictor_exists,
                inputDim=predictor_meta.get("input_dim"),
                createdAt=predictor_mtime,
            ),
            agent=AgentInfo(
                exists=agent_exists,
                stateDim=agent_meta.get("state_dim"),
                actionDim=agent_meta.get("action_dim"),
                createdAt=agent_mtime,
            ),
            metadata=ModelMetadata(
                symbol=metadata.get("symbol"),
                timeframe=metadata.get("timeframe"),
                trainedAt=metadata.get("trained_at"),
            ),
        )

    def _load_metadata(self, path: Path) -> dict[str, Any]:
        """Load model metadata."""
        metadata_file = path / "model_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.exception(f"Error loading metadata from {metadata_file}")
        return {}

    def _load_config(self, path: Path) -> Optional[dict[str, Any]]:
        """Load model config."""
        config_file = path / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError, OSError) as e:
                logger.exception(f"Error loading config from {config_file}")
        return None


models_service = ModelsService()
