"""
Leap Trading System - MLflow Experiment Tracking

Centralized MLflow tracking for experiment management, model versioning,
and metrics logging across the trading system.

MLflow 3 Features:
- LoggedModel: First-class model entities with unique IDs
- Dataset tracking: Link metrics to specific datasets
- Model-metric linking: Associate metrics with model versions
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    import torch
    import mlflow
    import mlflow.pytorch
    import mlflow.data
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    torch = None  # type: ignore

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config.settings import (
    MLflowConfig,
    TransformerConfig,
    PPOConfig,
    SystemConfig,
)

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Centralized MLflow tracking for Leap trading system.

    Handles experiment tracking, metric logging, model versioning,
    and model registry operations.

    Example:
        >>> tracker = MLflowTracker(config.mlflow)
        >>> tracker.setup()
        >>> with tracker.start_run("predictor-training"):
        ...     tracker.log_params({"learning_rate": 1e-4})
        ...     tracker.log_metrics({"train_loss": 0.05}, step=1)
        ...     tracker.log_model(model, "predictor")
    """

    def __init__(self, config: MLflowConfig):
        """Initialize MLflow tracker.

        Args:
            config: MLflow configuration dataclass
        """
        self.config = config
        self.active_run = None
        self._is_setup = False
        self._system_metrics_available = False

        # MLflow 3: Track datasets and logged models
        self._datasets: Dict[str, Any] = {}  # name -> Dataset object
        self._logged_models: Dict[str, str] = {}  # name -> model_id

        if not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow is not installed. Install with: pip install 'mlflow>=3'"
            )

    @property
    def is_enabled(self) -> bool:
        """Check if MLflow tracking is enabled and available."""
        return self.config.enabled and MLFLOW_AVAILABLE

    def setup(self) -> bool:
        """Initialize MLflow tracking URI and experiment.

        Returns:
            True if setup successful, False otherwise
        """
        if not self.is_enabled:
            logger.info("MLflow tracking is disabled")
            return False

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            logger.info(f"MLflow tracking URI: {self.config.tracking_uri}")

            # Log the mlflow ui command for convenience
            # Extract the database path from the URI for the user
            if self.config.tracking_uri.startswith("sqlite:///"):
                db_path = self.config.tracking_uri[len("sqlite:///"):]
                logger.info(
                    f"To view experiments, run: "
                    f"mlflow ui --backend-store-uri {self.config.tracking_uri}"
                )

            # Set or create experiment
            experiment = mlflow.set_experiment(self.config.experiment_name)
            logger.info(
                f"MLflow experiment: {self.config.experiment_name} "
                f"(ID: {experiment.experiment_id})"
            )

            # Enable system metrics logging if configured and psutil is available
            if self.config.log_system_metrics:
                if PSUTIL_AVAILABLE:
                    try:
                        mlflow.enable_system_metrics_logging()
                        self._system_metrics_available = True
                        logger.info("System metrics logging enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable system metrics: {e}")
                else:
                    logger.warning(
                        "System metrics logging disabled: psutil not installed. "
                        "Install with: pip install psutil"
                    )

            self._is_setup = True
            return True

        except Exception as e:
            logger.exception(f"Failed to setup MLflow: {e}")
            return False

    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run as context manager.

        Args:
            run_name: Name for the run (auto-generated if None)
            nested: Whether to create a nested run
            tags: Optional tags for the run

        Yields:
            MLflow ActiveRun object
        """
        if not self.is_enabled or not self._is_setup:
            yield None
            return

        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{self.config.run_name_prefix}_{timestamp}"

        try:
            with mlflow.start_run(
                run_name=run_name,
                nested=nested,
                log_system_metrics=self._system_metrics_available
            ) as run:
                self.active_run = run

                # Add default tags
                if tags:
                    mlflow.set_tags(tags)

                logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
                yield run

        except Exception as e:
            logger.exception(f"Error in MLflow run: {e}")
            raise
        finally:
            self.active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self.is_enabled:
            return

        try:
            # Flatten nested dicts and convert to strings
            flat_params = self._flatten_dict(params)
            mlflow.log_params(flat_params)
            logger.debug(f"Logged {len(flat_params)} parameters")
        except Exception as e:
            logger.warning(f"Failed to log parameters: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to current run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for the metrics
        """
        if not self.is_enabled:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics at step {step}: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """Log a single metric to current run.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if not self.is_enabled:
            return

        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metric {key}: {e}")

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """Log an artifact (file) to current run.

        Args:
            local_path: Local path to the file
            artifact_path: Optional subdirectory in artifact storage
        """
        if not self.is_enabled or not self.config.log_artifacts:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """Log all files in a directory as artifacts.

        Args:
            local_dir: Local directory path
            artifact_path: Optional subdirectory in artifact storage
        """
        if not self.is_enabled or not self.config.log_artifacts:
            return

        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.debug(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            logger.warning(f"Failed to log artifacts from {local_dir}: {e}")

    # =========================================================================
    # MLflow 3: Dataset Tracking
    # =========================================================================

    def create_dataset(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        name: str,
        context: str = "training"
    ) -> Optional[Any]:
        """Create and log an MLflow dataset for tracking.

        MLflow 3 Feature: Datasets can be linked to metrics for full lineage.

        Args:
            data: DataFrame or numpy array to track
            name: Dataset name (e.g., "train", "val", "test")
            context: Context for the dataset ("training", "validation", "testing")

        Returns:
            MLflow Dataset object if successful, None otherwise
        """
        if not self.is_enabled:
            return None

        try:
            # Convert numpy array to DataFrame if needed
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    df = pd.DataFrame({"data": data.astype(np.float64)})
                elif data.ndim == 2:
                    df = pd.DataFrame(data.astype(np.float64))
                else:
                    # For 3D data (sequences), store metadata as floats to avoid
                    # MLflow warning about integer columns not supporting NaN
                    df = pd.DataFrame({
                        "shape": [str(data.shape)],
                        "samples": [float(data.shape[0])],
                        "seq_length": [float(data.shape[1]) if data.ndim > 1 else 1.0],
                        "features": [float(data.shape[2]) if data.ndim > 2 else float(data.shape[1]) if data.ndim > 1 else 1.0]
                    })
            else:
                # For DataFrames, convert integer columns to float to avoid warnings
                df = data.copy()
                for col in df.select_dtypes(include=['int', 'int64', 'int32']).columns:
                    df[col] = df[col].astype(np.float64)

            # Create MLflow dataset
            dataset = mlflow.data.from_pandas(df, name=name)

            # Log as input
            mlflow.log_input(dataset, context=context)

            # Store for later reference
            self._datasets[name] = dataset

            logger.info(f"Created dataset '{name}' with context '{context}'")
            return dataset

        except Exception as e:
            logger.warning(f"Failed to create dataset {name}: {e}")
            return None

    def get_dataset(self, name: str) -> Optional[Any]:
        """Get a previously created dataset by name.

        Args:
            name: Dataset name

        Returns:
            MLflow Dataset object if found, None otherwise
        """
        return self._datasets.get(name)

    def log_model_metrics(
        self,
        metrics: Dict[str, float],
        model_name: str,
        dataset_name: Optional[str] = None,
        step: Optional[int] = None
    ) -> None:
        """Log metrics linked to a specific model and optionally a dataset.

        MLflow 3 Feature: Links metrics to LoggedModel entities for tracking
        model performance across different datasets.

        Args:
            metrics: Dictionary of metric names and values
            model_name: Name of the logged model (e.g., "predictor", "agent")
            dataset_name: Optional dataset name to link metrics to
            step: Optional step number
        """
        if not self.is_enabled:
            return

        try:
            model_id = self._logged_models.get(model_name)
            dataset = self._datasets.get(dataset_name) if dataset_name else None

            if model_id:
                # Use MLflow 3's model-linked metrics
                mlflow.log_metrics(
                    metrics=metrics,
                    step=step,
                    model_id=model_id,
                    dataset=dataset
                )
                logger.debug(
                    f"Logged metrics linked to model '{model_name}'"
                    + (f" and dataset '{dataset_name}'" if dataset_name else "")
                )
            else:
                # Fallback to regular metrics
                mlflow.log_metrics(metrics, step=step)
                logger.debug(f"Logged metrics (no model link): {list(metrics.keys())}")

        except Exception as e:
            # Fallback for older MLflow versions
            logger.debug(f"Model-linked metrics not available, using standard: {e}")
            mlflow.log_metrics(metrics, step=step)

    def get_logged_model(self, model_name: str) -> Optional[Any]:
        """Get a LoggedModel by name.

        Args:
            model_name: Name of the model (e.g., "predictor", "agent")

        Returns:
            LoggedModel object if found, None otherwise
        """
        if not self.is_enabled:
            return None

        model_id = self._logged_models.get(model_name)
        if not model_id:
            return None

        try:
            return mlflow.get_logged_model(model_id)
        except Exception as e:
            logger.warning(f"Failed to get logged model {model_name}: {e}")
            return None

    def get_model_id(self, model_name: str) -> Optional[str]:
        """Get the model ID for a logged model.

        Args:
            model_name: Name of the model

        Returns:
            Model ID string if found, None otherwise
        """
        return self._logged_models.get(model_name)

    # =========================================================================
    # End MLflow 3 Dataset Tracking
    # =========================================================================

    def log_predictor_params(
        self,
        config: TransformerConfig,
        max_seq_length_override: Optional[int] = None
    ) -> None:
        """Log Transformer predictor hyperparameters.

        Args:
            config: TransformerConfig dataclass
            max_seq_length_override: Override for max_seq_length (use data.lookback_window)
                                     to log the actual value used by the model
        """
        # Use override if provided (this ensures we log the actual value used)
        actual_max_seq_length = (
            max_seq_length_override
            if max_seq_length_override is not None
            else config.max_seq_length
        )

        params = {
            "predictor.d_model": config.d_model,
            "predictor.n_heads": config.n_heads,
            "predictor.n_encoder_layers": config.n_encoder_layers,
            "predictor.d_ff": config.d_ff,
            "predictor.dropout": config.dropout,
            "predictor.max_seq_length": actual_max_seq_length,
            "predictor.learning_rate": config.learning_rate,
            "predictor.weight_decay": config.weight_decay,
            "predictor.batch_size": config.batch_size,
            "predictor.epochs": config.epochs,
            "predictor.patience": config.patience,
        }
        self.log_params(params)

    def log_agent_params(self, config: PPOConfig) -> None:
        """Log PPO agent hyperparameters.

        Args:
            config: PPOConfig dataclass
        """
        params = {
            "agent.learning_rate": config.learning_rate,
            "agent.gamma": config.gamma,
            "agent.gae_lambda": config.gae_lambda,
            "agent.clip_epsilon": config.clip_epsilon,
            "agent.entropy_coef": config.entropy_coef,
            "agent.value_coef": config.value_coef,
            "agent.max_grad_norm": config.max_grad_norm,
            "agent.n_steps": config.n_steps,
            "agent.n_epochs": config.n_epochs,
            "agent.batch_size": config.batch_size,
            "agent.total_timesteps": config.total_timesteps,
            "agent.actor_hidden_sizes": str(config.actor_hidden_sizes),
            "agent.critic_hidden_sizes": str(config.critic_hidden_sizes),
        }
        self.log_params(params)

    def log_pytorch_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        input_example: Optional[np.ndarray] = None,
        signature=None,
        extra_files: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log a PyTorch model to MLflow.

        MLflow 3: Creates a LoggedModel entity with unique ID for tracking.

        Args:
            model: PyTorch nn.Module to log
            artifact_path: Name for the model artifact
            registered_model_name: If provided, register model in Model Registry
            input_example: Sample input for signature inference
            signature: Model signature (inferred if input_example provided)
            extra_files: Additional files to include with model
            model_params: MLflow 3 - Model parameters to attach to LoggedModel

        Returns:
            Model URI if successful, None otherwise
        """
        if not self.is_enabled or not self.config.log_models:
            return None

        try:
            # Infer signature if input_example provided but no signature
            if signature is None and input_example is not None:
                model.eval()
                with torch.no_grad():
                    if isinstance(input_example, np.ndarray):
                        input_tensor = torch.from_numpy(input_example).float()
                    else:
                        input_tensor = input_example

                    # Get device from model parameters (nn.Module doesn't have .device)
                    try:
                        device = next(model.parameters()).device
                        input_tensor = input_tensor.to(device)
                    except StopIteration:
                        pass  # Model has no parameters, keep on CPU

                    output = model(input_tensor)

                    if isinstance(output, tuple):
                        output = output[0]
                    if isinstance(output, dict):
                        # Handle dict outputs (e.g., TemporalFusionTransformer)
                        # Prefer explicit keys, then fall back to first value
                        if 'prediction' in output:
                            output = output['prediction']
                        elif 'output' in output:
                            output = output['output']
                        else:
                            try:
                                output = next(iter(output.values()))
                            except StopIteration:
                                raise ValueError(
                                    "Model output dict is empty; cannot infer signature."
                                )

                    output_np = output.cpu().numpy()
                    signature = infer_signature(input_example, output_np)

            # Determine if we should register the model
            register_name = None
            if self.config.register_models and registered_model_name:
                register_name = registered_model_name

            # Log the model with MLflow 3 params support
            log_kwargs = {
                "pytorch_model": model,
                "name": artifact_path,
                "signature": signature,
                "input_example": input_example,
                "registered_model_name": register_name,
                "extra_files": extra_files,
            }

            # MLflow 3: Add params for LoggedModel
            if model_params:
                log_kwargs["params"] = model_params

            model_info = mlflow.pytorch.log_model(**log_kwargs)

            # MLflow 3: Store model_id for later metric linking
            if hasattr(model_info, 'model_id') and model_info.model_id:
                self._logged_models[artifact_path] = model_info.model_id
                logger.info(
                    f"Logged PyTorch model: {artifact_path} "
                    f"(model_id: {model_info.model_id})"
                )
            else:
                logger.info(f"Logged PyTorch model: {artifact_path}")

            if register_name:
                logger.info(f"Registered model as: {register_name}")

            return model_info.model_uri

        except Exception as e:
            logger.exception(f"Failed to log PyTorch model: {e}")
            return None

    def log_predictor_model(
        self,
        model,
        input_dim: int,
        seq_length: int = 120,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log Transformer predictor model with proper signature.

        MLflow 3: Creates LoggedModel with params for model tracking.

        Args:
            model: TransformerPredictor model
            input_dim: Number of input features
            seq_length: Sequence length
            model_config: Optional model configuration to attach as params

        Returns:
            Model URI if successful, None otherwise
        """
        # Create example input for signature inference
        input_example = np.zeros((1, seq_length, input_dim), dtype=np.float32)

        # Infer signature manually since model returns dict (which MLflow
        # serving validation can't handle). We pass signature but not
        # input_example to avoid the validation error.
        signature = None
        try:
            model.eval()
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_example).float()
                try:
                    device = next(model.parameters()).device
                    input_tensor = input_tensor.to(device)
                except StopIteration:
                    pass

                output = model(input_tensor)

                # Handle dict output from Transformer
                if isinstance(output, dict):
                    if 'prediction' in output:
                        output = output['prediction']
                    elif 'output' in output:
                        output = output['output']
                    else:
                        if not output:
                            raise ValueError(
                                "Model output dict is empty; cannot infer signature."
                            )
                        output = next(iter(output.values()))

                output_np = output.cpu().numpy()
                signature = infer_signature(input_example, output_np)
        except Exception as e:
            logger.warning(f"Could not infer predictor signature: {e}")

        # MLflow 3: Build model params
        model_params = {
            "input_dim": input_dim,
            "seq_length": seq_length,
            "model_type": "transformer_predictor",
        }
        if model_config:
            model_params.update(model_config)

        return self.log_pytorch_model(
            model=model,
            artifact_path="predictor",
            registered_model_name=self.config.registered_model_name_predictor,
            input_example=None,  # Skip input_example to avoid dict output validation error
            signature=signature,
            model_params=model_params
        )

    def log_agent_model(
        self,
        model,
        state_dim: int,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log PPO agent model with proper signature.

        MLflow 3: Creates LoggedModel with params for model tracking.

        Args:
            model: PPOAgent network
            state_dim: State dimension
            model_config: Optional model configuration to attach as params

        Returns:
            Model URI if successful, None otherwise
        """
        # Create example input for signature inference
        input_example = np.zeros((1, state_dim), dtype=np.float32)

        # Infer signature manually since model returns tuple (action_logits, value)
        # which MLflow serving validation can't handle. We pass signature but not
        # input_example to avoid the validation error.
        signature = None
        try:
            model.eval()
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_example).float()
                try:
                    device = next(model.parameters()).device
                    input_tensor = input_tensor.to(device)
                except StopIteration:
                    pass

                output = model(input_tensor)

                # Handle tuple output from PPO agent (action_logits, value)
                if isinstance(output, tuple):
                    output = output[0]  # Use action logits for signature

                output_np = output.cpu().numpy()
                signature = infer_signature(input_example, output_np)
        except Exception as e:
            logger.warning(f"Could not infer agent signature: {e}")

        # MLflow 3: Build model params
        model_params = {
            "state_dim": state_dim,
            "model_type": "ppo_agent",
        }
        if model_config:
            model_params.update(model_config)

        return self.log_pytorch_model(
            model=model,
            artifact_path="agent",
            registered_model_name=self.config.registered_model_name_agent,
            input_example=None,  # Skip input_example to avoid tuple output validation error
            signature=signature,
            model_params=model_params
        )

    def log_backtest_results(
        self,
        results: Dict[str, Any],
        prefix: str = "backtest"
    ) -> None:
        """Log backtesting results as metrics.

        Args:
            results: Dictionary of backtest results
            prefix: Prefix for metric names
        """
        if not self.is_enabled:
            return

        # Extract numeric metrics
        numeric_metrics = {}
        for key, value in results.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metric_name = f"{prefix}.{key}" if prefix else key
                numeric_metrics[metric_name] = float(value)

        self.log_metrics(numeric_metrics)

    def log_training_summary(
        self,
        model_type: str,
        final_metrics: Dict[str, float],
        duration_seconds: float
    ) -> None:
        """Log training summary at end of training.

        Args:
            model_type: "predictor" or "agent"
            final_metrics: Final training metrics
            duration_seconds: Training duration in seconds
        """
        if not self.is_enabled:
            return

        summary = {
            f"{model_type}.training_duration_seconds": duration_seconds,
            f"{model_type}.training_duration_minutes": duration_seconds / 60,
        }
        summary.update({
            f"{model_type}.final_{k}": v
            for k, v in final_metrics.items()
        })

        self.log_metrics(summary)

    def set_model_alias(
        self,
        model_name: str,
        version: Union[int, str],
        alias: str
    ) -> bool:
        """Set an alias for a model version in the registry.

        Args:
            model_name: Registered model name
            version: Model version number
            alias: Alias to set (e.g., "champion", "staging")

        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled:
            return False

        try:
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(model_name, alias, str(version))
            logger.info(f"Set alias '{alias}' for {model_name} version {version}")
            return True
        except Exception as e:
            logger.exception(f"Failed to set model alias: {e}")
            return False

    def load_model(
        self,
        model_uri: str,
        map_location: Optional[str] = None
    ):
        """Load a PyTorch model from MLflow.

        Args:
            model_uri: Model URI (e.g., "models:/leap-predictor/1" or
                      "models:/leap-predictor@champion")
            map_location: Device to load model to

        Returns:
            Loaded PyTorch model
        """
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow is not installed")

        try:
            if map_location:
                return mlflow.pytorch.load_model(model_uri, map_location=map_location)
            return mlflow.pytorch.load_model(model_uri)
        except Exception as e:
            logger.exception(f"Failed to load model from {model_uri}: {e}")
            raise

    def get_run_id(self) -> Optional[str]:
        """Get the current active run ID.

        Returns:
            Run ID if active run exists, None otherwise
        """
        if self.active_run:
            return self.active_run.info.run_id
        return None

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, str]:
        """Flatten nested dictionary for MLflow params.

        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator between keys

        Returns:
            Flattened dictionary with string values
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, str(v) if v is not None else "None"))

        return dict(items)


def create_tracker(config: SystemConfig) -> MLflowTracker:
    """Factory function to create MLflow tracker from system config.

    Args:
        config: System configuration

    Returns:
        Configured MLflowTracker instance
    """
    tracker = MLflowTracker(config.mlflow)
    if tracker.is_enabled:
        tracker.setup()
    return tracker
