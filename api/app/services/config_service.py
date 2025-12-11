"""Configuration service."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.config import settings
from ..schemas.config import (
    BacktestConfigFull,
    ConfigTemplate,
    ConfigUpdateRequest,
    DataConfig,
    PPOConfigFull,
    RiskConfig,
    SystemConfigData,
    TransformerConfigFull,
)

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for managing configuration."""

    def __init__(self):
        self.config_file = settings.CONFIG_DIR / "system_config.json"
        self.templates_dir = settings.CONFIG_DIR / "templates"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self._current_config: Optional[SystemConfigData] = None

    def get_config(self) -> SystemConfigData:
        """Get current system configuration."""
        if self._current_config:
            return self._current_config

        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)
                self._current_config = SystemConfigData(**data)
                return self._current_config
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        # Return defaults
        self._current_config = SystemConfigData()
        return self._current_config

    def update_config(self, update: ConfigUpdateRequest) -> SystemConfigData:
        """Update system configuration."""
        current = self.get_config()

        if update.data:
            current.data = update.data
        if update.transformer:
            current.transformer = update.transformer
        if update.ppo:
            current.ppo = update.ppo
        if update.risk:
            current.risk = update.risk
        if update.backtest:
            current.backtest = update.backtest

        # Save to file
        self._save_config(current)
        self._current_config = current
        return current

    def _save_config(self, config: SystemConfigData) -> None:
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(config.model_dump(by_alias=True), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

    def list_templates(self) -> list[ConfigTemplate]:
        """List configuration templates."""
        templates = []

        # Built-in templates
        templates.extend([
            ConfigTemplate(
                id="quick-test",
                name="Quick Test",
                description="Fast iteration with minimal training",
                createdAt=datetime.utcnow().isoformat() + "Z",
            ),
            ConfigTemplate(
                id="standard",
                name="Standard",
                description="Balanced training configuration",
                createdAt=datetime.utcnow().isoformat() + "Z",
            ),
            ConfigTemplate(
                id="production",
                name="Production",
                description="Full training for deployment",
                createdAt=datetime.utcnow().isoformat() + "Z",
            ),
        ])

        # User templates
        for template_file in self.templates_dir.glob("*.json"):
            try:
                with open(template_file) as f:
                    data = json.load(f)
                templates.append(ConfigTemplate(
                    id=template_file.stem,
                    name=data.get("name", template_file.stem),
                    description=data.get("description"),
                    createdAt=data.get("createdAt", datetime.utcnow().isoformat() + "Z"),
                ))
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")

        return templates

    def create_template(
        self, name: str, description: Optional[str], config: dict[str, Any]
    ) -> ConfigTemplate:
        """Create a new configuration template."""
        template_id = f"{name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
        created_at = datetime.utcnow().isoformat() + "Z"

        template_data = {
            "name": name,
            "description": description,
            "config": config,
            "createdAt": created_at,
        }

        template_file = self.templates_dir / f"{template_id}.json"
        with open(template_file, "w") as f:
            json.dump(template_data, f, indent=2)

        return ConfigTemplate(
            id=template_id,
            name=name,
            description=description,
            createdAt=created_at,
        )

    def get_template(self, template_id: str) -> Optional[dict[str, Any]]:
        """Get a configuration template."""
        # Built-in templates
        if template_id == "quick-test":
            return {
                "transformer": {"epochs": 20},
                "ppo": {"totalTimesteps": 50000},
                "data": {"bars": 10000},
            }
        elif template_id == "standard":
            return {
                "transformer": {"epochs": 100},
                "ppo": {"totalTimesteps": 500000},
                "data": {"bars": 50000},
            }
        elif template_id == "production":
            return {
                "transformer": {"epochs": 200},
                "ppo": {"totalTimesteps": 1000000},
                "data": {"bars": 100000},
            }

        # User templates
        template_file = self.templates_dir / f"{template_id}.json"
        if template_file.exists():
            with open(template_file) as f:
                data = json.load(f)
            return data.get("config")

        return None

    def validate_config(self, config: dict[str, Any]) -> dict[str, list[str]]:
        """Validate configuration and return errors."""
        errors: dict[str, list[str]] = {}

        # Validate transformer config
        if "transformer" in config:
            t = config["transformer"]
            if "epochs" in t:
                if t["epochs"] < 1 or t["epochs"] > 1000:
                    errors["transformer.epochs"] = ["Must be between 1 and 1000"]
            if "dModel" in t:
                if t["dModel"] not in [32, 64, 128, 256, 512]:
                    errors["transformer.dModel"] = ["Must be one of: 32, 64, 128, 256, 512"]
            if "nHeads" in t:
                if t["nHeads"] not in [1, 2, 4, 8, 16]:
                    errors["transformer.nHeads"] = ["Must be one of: 1, 2, 4, 8, 16"]
            if "learningRate" in t:
                if t["learningRate"] <= 0 or t["learningRate"] > 0.01:
                    errors["transformer.learningRate"] = ["Must be between 0 and 0.01"]

        # Validate PPO config
        if "ppo" in config:
            p = config["ppo"]
            if "gamma" in p:
                if p["gamma"] < 0.9 or p["gamma"] > 0.999:
                    errors["ppo.gamma"] = ["Must be between 0.9 and 0.999"]
            if "clipEpsilon" in p:
                if p["clipEpsilon"] < 0.1 or p["clipEpsilon"] > 0.3:
                    errors["ppo.clipEpsilon"] = ["Must be between 0.1 and 0.3"]

        # Validate risk config
        if "risk" in config:
            r = config["risk"]
            if "maxPositionSize" in r:
                if r["maxPositionSize"] <= 0 or r["maxPositionSize"] > 0.1:
                    errors["risk.maxPositionSize"] = ["Must be between 0 and 0.1"]

        return errors


config_service = ConfigService()
