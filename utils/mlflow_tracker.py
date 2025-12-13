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
            if self.config.tracking_uri.startswith("sqlite:///"):
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


    # =========================================================================
    # Walk-Forward Optimization Tracking
    # =========================================================================

    def log_walkforward_params(
        self,
        train_window: int,
        test_window: int,
        step_size: int,
        n_splits: Optional[int] = None
    ) -> None:
        """Log walk-forward optimization parameters.

        Args:
            train_window: Training window size in days
            test_window: Test window size in days
            step_size: Step size in days
            n_splits: Number of splits (if specified)
        """
        if not self.is_enabled:
            return

        params = {
            "walkforward.train_window_days": train_window,
            "walkforward.test_window_days": test_window,
            "walkforward.step_size_days": step_size,
        }
        if n_splits is not None:
            params["walkforward.n_splits"] = n_splits

        self.log_params(params)

    def log_walkforward_fold(
        self,
        fold_idx: int,
        metrics: Dict[str, float],
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None
    ) -> None:
        """Log metrics for a single walk-forward fold.

        Args:
            fold_idx: Fold index (0-based)
            metrics: Dictionary of metric names and values
            train_start: Training period start date
            train_end: Training period end date
            test_start: Test period start date
            test_end: Test period end date
        """
        if not self.is_enabled:
            return

        # Prefix metrics with fold index
        fold_metrics = {
            f"walkforward.fold_{fold_idx}.{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not np.isnan(v)
        }
        self.log_metrics(fold_metrics)

        # Log fold period as params (only works if not already logged)
        try:
            fold_params = {}
            if train_start:
                fold_params[f"walkforward.fold_{fold_idx}.train_start"] = str(train_start)
            if train_end:
                fold_params[f"walkforward.fold_{fold_idx}.train_end"] = str(train_end)
            if test_start:
                fold_params[f"walkforward.fold_{fold_idx}.test_start"] = str(test_start)
            if test_end:
                fold_params[f"walkforward.fold_{fold_idx}.test_end"] = str(test_end)
            if fold_params:
                self.log_params(fold_params)
        except Exception as e:
            logger.debug(f"Could not log fold params (may already exist): {e}")

    def log_walkforward_summary(self, results: Dict[str, Any]) -> None:
        """Log aggregated walk-forward optimization results.

        Args:
            results: Dictionary with aggregated results from WalkForwardOptimizer
        """
        if not self.is_enabled:
            return

        metrics = {}

        # Number of folds
        metrics["walkforward.n_folds"] = results.get("n_folds", 0)

        # Aggregate metrics
        agg = results.get("aggregate", {})
        for metric_name, stats in agg.items():
            if isinstance(stats, dict):
                for stat_name, value in stats.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics[f"walkforward.{metric_name}_{stat_name}"] = value

        # Consistency metrics
        consistency = results.get("consistency", {})
        metrics["walkforward.profitable_folds"] = consistency.get("profitable_folds", 0)
        metrics["walkforward.profitable_ratio"] = consistency.get("profitable_ratio", 0)

        self.log_metrics(metrics)

    # =========================================================================
    # Auto-Trader / Live Trading Tracking
    # =========================================================================

    def log_autotrader_params(self, config: Dict[str, Any]) -> None:
        """Log auto-trader configuration parameters.

        Args:
            config: Auto-trader configuration dictionary
        """
        if not self.is_enabled:
            return

        params = {
            "autotrader.paper_mode": config.get("paper_mode", True),
            "autotrader.symbols": ",".join(config.get("symbols", [])),
            "autotrader.risk_per_trade": config.get("risk_per_trade", 0.02),
            "autotrader.max_positions": config.get("max_positions", 5),
            "autotrader.default_sl_pips": config.get("default_sl_pips", 50),
            "autotrader.default_tp_pips": config.get("default_tp_pips", 100),
            "autotrader.min_confidence": config.get("min_confidence", 0.6),
            "autotrader.prediction_threshold": config.get("prediction_threshold", 0.001),
            "autotrader.enable_online_learning": config.get("enable_online_learning", False),
        }
        self.log_params(params)

    def log_autotrader_trade(
        self,
        trade_data: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """Log a single auto-trader trade execution.

        Args:
            trade_data: Dictionary containing trade information
            step: Optional step/trade number
        """
        if not self.is_enabled:
            return

        metrics = {}

        # Core trade metrics
        if "pnl" in trade_data:
            metrics["autotrader.trade_pnl"] = trade_data["pnl"]
        if "direction" in trade_data:
            metrics["autotrader.trade_direction"] = 1 if trade_data["direction"] == "long" else -1
        if "confidence" in trade_data:
            metrics["autotrader.trade_confidence"] = trade_data["confidence"]
        if "predicted_return" in trade_data:
            metrics["autotrader.trade_predicted_return"] = trade_data["predicted_return"]
        if "actual_return" in trade_data:
            metrics["autotrader.trade_actual_return"] = trade_data["actual_return"]
        if "slippage" in trade_data:
            metrics["autotrader.trade_slippage"] = trade_data["slippage"]

        if metrics:
            self.log_metrics(metrics, step=step)

    def log_autotrader_session(
        self,
        session_stats: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """Log auto-trader session statistics.

        Args:
            session_stats: Dictionary containing session statistics
            step: Optional step number (e.g., heartbeat count)
        """
        if not self.is_enabled:
            return

        metrics = {
            "autotrader.session_total_trades": session_stats.get("total_trades", 0),
            "autotrader.session_winning_trades": session_stats.get("winning_trades", 0),
            "autotrader.session_losing_trades": session_stats.get("losing_trades", 0),
            "autotrader.session_total_pnl": session_stats.get("total_pnl", 0),
            "autotrader.session_win_rate": session_stats.get("win_rate", 0),
            "autotrader.session_max_drawdown": session_stats.get("max_drawdown", 0),
            "autotrader.session_signals_generated": session_stats.get("signals_generated", 0),
            "autotrader.session_signals_executed": session_stats.get("signals_executed", 0),
            "autotrader.session_signals_rejected": session_stats.get("signals_rejected", 0),
            "autotrader.session_online_adaptations": session_stats.get("online_adaptations", 0),
        }

        # Filter out None values and NaN
        metrics = {
            k: v for k, v in metrics.items()
            if v is not None and (not isinstance(v, float) or not np.isnan(v))
        }

        self.log_metrics(metrics, step=step)

    def log_autotrader_heartbeat(
        self,
        balance: float,
        equity: float,
        positions: int,
        step: Optional[int] = None
    ) -> None:
        """Log auto-trader heartbeat metrics.

        Args:
            balance: Current account balance
            equity: Current account equity
            positions: Number of open positions
            step: Optional step/heartbeat number
        """
        if not self.is_enabled:
            return

        metrics = {
            "autotrader.balance": balance,
            "autotrader.equity": equity,
            "autotrader.positions": positions,
            "autotrader.floating_pnl": equity - balance,
        }
        self.log_metrics(metrics, step=step)

    # =========================================================================
    # Online Learning Tracking
    # =========================================================================

    def log_online_learning_params(self, config: Dict[str, Any]) -> None:
        """Log online learning configuration parameters.

        Args:
            config: AdaptationConfig as dictionary
        """
        if not self.is_enabled:
            return

        params = {
            "online_learning.error_threshold": config.get("error_threshold", 0.05),
            "online_learning.drawdown_threshold": config.get("drawdown_threshold", 0.1),
            "online_learning.performance_window": config.get("performance_window", 100),
            "online_learning.adaptation_frequency": config.get("adaptation_frequency", 100),
            "online_learning.min_samples_for_adaptation": config.get("min_samples_for_adaptation", 50),
            "online_learning.learning_rate_decay": config.get("learning_rate_decay", 0.95),
            "online_learning.max_adaptations_per_day": config.get("max_adaptations_per_day", 10),
            "online_learning.regime_detection_enabled": config.get("regime_detection_enabled", True),
        }
        self.log_params(params)

    def log_online_learning_step(
        self,
        metrics: Dict[str, Any],
        step: int
    ) -> None:
        """Log online learning step metrics.

        Args:
            metrics: Dictionary containing step metrics
            step: Current step number
        """
        if not self.is_enabled:
            return

        log_metrics = {}

        if "prediction_error" in metrics:
            log_metrics["online_learning.prediction_error"] = metrics["prediction_error"]
        if "trading_reward" in metrics:
            log_metrics["online_learning.trading_reward"] = metrics["trading_reward"]
        if "sharpe_ratio" in metrics:
            log_metrics["online_learning.sharpe_ratio"] = metrics["sharpe_ratio"]
        if "win_rate" in metrics:
            log_metrics["online_learning.win_rate"] = metrics["win_rate"]
        if "profit_factor" in metrics:
            log_metrics["online_learning.profit_factor"] = metrics["profit_factor"]

        if log_metrics:
            self.log_metrics(log_metrics, step=step)

    def log_online_adaptation(
        self,
        adaptation_result: Dict[str, Any],
        step: int
    ) -> None:
        """Log online learning adaptation event.

        Args:
            adaptation_result: Dictionary containing adaptation results
            step: Current step number
        """
        if not self.is_enabled:
            return

        metrics = {
            "online_learning.adaptation_triggered": 1,
            "online_learning.predictor_updated": int(adaptation_result.get("predictor_updated", False)),
            "online_learning.agent_updated": int(adaptation_result.get("agent_updated", False)),
        }

        if adaptation_result.get("predictor_loss") is not None:
            metrics["online_learning.predictor_adaptation_loss"] = adaptation_result["predictor_loss"]
        if adaptation_result.get("agent_loss") is not None:
            metrics["online_learning.agent_adaptation_loss"] = adaptation_result["agent_loss"]

        # Log regime as a metric (encode as integer)
        regime_map = {
            "unknown": 0, "trending_up": 1, "trending_down": 2,
            "ranging": 3, "high_volatility": 4, "low_volatility": 5
        }
        regime = adaptation_result.get("regime", "unknown")
        metrics["online_learning.regime"] = regime_map.get(regime, 0)

        self.log_metrics(metrics, step=step)

    def log_online_learning_report(self, report: Dict[str, Any]) -> None:
        """Log online learning performance report.

        Args:
            report: Performance report from OnlineLearningManager
        """
        if not self.is_enabled:
            return

        metrics = {
            "online_learning.total_steps": report.get("total_steps", 0),
            "online_learning.total_adaptations": report.get("total_adaptations", 0),
            "online_learning.avg_prediction_error": report.get("avg_prediction_error", 0),
            "online_learning.total_pnl": report.get("total_pnl", 0),
            "online_learning.avg_sharpe": report.get("avg_sharpe", 0),
            "online_learning.avg_win_rate": report.get("avg_win_rate", 0),
        }

        # Log regime distribution
        regime_dist = report.get("regime_distribution", {})
        for regime, pct in regime_dist.items():
            metrics[f"online_learning.regime_pct_{regime}"] = pct

        self.log_metrics(metrics)

    # =========================================================================
    # Feature Engineering Tracking
    # =========================================================================

    def log_feature_engineering_params(
        self,
        n_features: int,
        feature_names: Optional[List[str]] = None,
        multi_timeframe_enabled: bool = False,
        additional_timeframes: Optional[List[str]] = None
    ) -> None:
        """Log feature engineering configuration.

        Args:
            n_features: Total number of features
            feature_names: List of feature names
            multi_timeframe_enabled: Whether multi-timeframe features are used
            additional_timeframes: List of additional timeframes
        """
        if not self.is_enabled:
            return

        params = {
            "features.n_features": n_features,
            "features.multi_timeframe_enabled": multi_timeframe_enabled,
        }

        if additional_timeframes:
            params["features.additional_timeframes"] = ",".join(additional_timeframes)
            params["features.n_timeframes"] = len(additional_timeframes) + 1

        # Log feature categories count
        if feature_names:
            feature_categories = self._categorize_features(feature_names)
            for category, count in feature_categories.items():
                params[f"features.n_{category}"] = count

        self.log_params(params)

    def log_feature_statistics(
        self,
        feature_stats: Dict[str, Dict[str, float]],
        prefix: str = "features"
    ) -> None:
        """Log feature statistics (mean, std, min, max, etc.).

        Args:
            feature_stats: Dictionary mapping feature names to their statistics
            prefix: Prefix for metric names
        """
        if not self.is_enabled:
            return

        metrics = {}

        # Aggregate statistics across features
        means = []
        stds = []
        missing_rates = []

        for feat_name, stats in feature_stats.items():
            if "mean" in stats:
                means.append(stats["mean"])
            if "std" in stats:
                stds.append(stats["std"])
            if "missing_rate" in stats:
                missing_rates.append(stats["missing_rate"])

        if means:
            metrics[f"{prefix}.avg_feature_mean"] = np.mean(means)
        if stds:
            metrics[f"{prefix}.avg_feature_std"] = np.mean(stds)
        if missing_rates:
            metrics[f"{prefix}.avg_missing_rate"] = np.mean(missing_rates)
            metrics[f"{prefix}.max_missing_rate"] = np.max(missing_rates)

        self.log_metrics(metrics)

    def _categorize_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize features by type for reporting."""
        categories = {
            "momentum": 0,
            "volatility": 0,
            "trend": 0,
            "volume": 0,
            "price": 0,
            "time": 0,
            "candlestick": 0,
            "other": 0
        }

        momentum_keywords = ["rsi", "macd", "stoch", "momentum", "roc", "williams"]
        volatility_keywords = ["atr", "bb_", "keltner", "volatility", "tr"]
        trend_keywords = ["sma", "ema", "adx", "trend", "di_"]
        volume_keywords = ["volume", "obv", "vpt", "mfi", "ad"]
        price_keywords = ["returns", "ratio", "gap", "close", "open", "high", "low"]
        time_keywords = ["hour", "day", "month", "sin", "cos"]
        candlestick_keywords = ["doji", "hammer", "engulfing", "body", "shadow", "bullish", "bearish"]

        for feat in feature_names:
            feat_lower = feat.lower()
            if any(kw in feat_lower for kw in momentum_keywords):
                categories["momentum"] += 1
            elif any(kw in feat_lower for kw in volatility_keywords):
                categories["volatility"] += 1
            elif any(kw in feat_lower for kw in trend_keywords):
                categories["trend"] += 1
            elif any(kw in feat_lower for kw in volume_keywords):
                categories["volume"] += 1
            elif any(kw in feat_lower for kw in price_keywords):
                categories["price"] += 1
            elif any(kw in feat_lower for kw in time_keywords):
                categories["time"] += 1
            elif any(kw in feat_lower for kw in candlestick_keywords):
                categories["candlestick"] += 1
            else:
                categories["other"] += 1

        return categories

    # =========================================================================
    # Monte Carlo Analysis Tracking
    # =========================================================================

    def log_monte_carlo_params(
        self,
        n_simulations: int,
        confidence_level: float
    ) -> None:
        """Log Monte Carlo simulation parameters.

        Args:
            n_simulations: Number of simulations
            confidence_level: Confidence level for statistics
        """
        if not self.is_enabled:
            return

        self.log_params({
            "monte_carlo.n_simulations": n_simulations,
            "monte_carlo.confidence_level": confidence_level,
        })

    def log_monte_carlo_results(self, results: Dict[str, Any]) -> None:
        """Log Monte Carlo simulation results.

        Args:
            results: Dictionary with Monte Carlo simulation results
        """
        if not self.is_enabled:
            return

        metrics = {}

        # Final equity statistics
        final_eq = results.get("final_equity", {})
        metrics["monte_carlo.final_equity_mean"] = final_eq.get("mean", 0)
        metrics["monte_carlo.final_equity_std"] = final_eq.get("std", 0)
        metrics["monte_carlo.final_equity_median"] = final_eq.get("median", 0)
        metrics["monte_carlo.final_equity_p5"] = final_eq.get("percentile_5", 0)
        metrics["monte_carlo.final_equity_p95"] = final_eq.get("percentile_95", 0)

        # Max drawdown statistics
        max_dd = results.get("max_drawdown", {})
        metrics["monte_carlo.max_drawdown_mean"] = max_dd.get("mean", 0)
        metrics["monte_carlo.max_drawdown_std"] = max_dd.get("std", 0)
        metrics["monte_carlo.max_drawdown_p95"] = max_dd.get("percentile_95", 0)

        # Probability metrics
        metrics["monte_carlo.probability_of_profit"] = results.get("probability_of_profit", 0)
        metrics["monte_carlo.probability_of_ruin"] = results.get("probability_of_ruin", 0)

        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}

        self.log_metrics(metrics)

    # =========================================================================
    # Extended Backtest Metrics
    # =========================================================================

    def log_extended_backtest_results(
        self,
        results: Dict[str, Any],
        prefix: str = "backtest"
    ) -> None:
        """Log extended backtest results with all available metrics.

        Args:
            results: Dictionary of backtest results (from BacktestResult or dict)
            prefix: Prefix for metric names
        """
        if not self.is_enabled:
            return

        metrics = {}

        # Return metrics
        for key in ["total_return", "annualized_return", "cagr"]:
            if key in results:
                metrics[f"{prefix}.{key}"] = results[key]

        # Risk metrics
        for key in ["volatility", "downside_volatility", "max_drawdown",
                    "max_drawdown_duration", "var_95", "cvar_95"]:
            if key in results:
                metrics[f"{prefix}.{key}"] = results[key]

        # Risk-adjusted metrics
        for key in ["sharpe_ratio", "sortino_ratio", "calmar_ratio",
                    "omega_ratio", "information_ratio"]:
            if key in results:
                val = results[key]
                if val is not None and not (isinstance(val, float) and np.isinf(val)):
                    metrics[f"{prefix}.{key}"] = val

        # Distribution metrics
        for key in ["skewness", "kurtosis"]:
            if key in results:
                metrics[f"{prefix}.{key}"] = results[key]

        # Trade statistics
        for key in ["total_trades", "winning_trades", "losing_trades",
                    "win_rate", "profit_factor", "avg_trade_return",
                    "avg_winner", "avg_loser", "largest_winner", "largest_loser",
                    "avg_trade_duration", "payoff_ratio", "expectancy",
                    "consecutive_winners", "consecutive_losers"]:
            if key in results:
                val = results[key]
                if val is not None and not (isinstance(val, float) and np.isinf(val)):
                    metrics[f"{prefix}.{key}"] = val

        # Tail risk metrics
        for key in ["tail_ratio", "gain_to_pain_ratio"]:
            if key in results:
                val = results[key]
                if val is not None and not (isinstance(val, float) and np.isinf(val)):
                    metrics[f"{prefix}.{key}"] = val

        self.log_metrics(metrics)

    # =========================================================================
    # Risk Management Tracking
    # =========================================================================

    def log_risk_management_params(self, config: Dict[str, Any]) -> None:
        """Log risk management configuration.

        Args:
            config: Risk management configuration dictionary
        """
        if not self.is_enabled:
            return

        params = {
            "risk.max_position_size_pct": config.get("max_position_size_pct", 0.1),
            "risk.max_portfolio_risk_pct": config.get("max_portfolio_risk_pct", 0.2),
            "risk.max_correlated_positions": config.get("max_correlated_positions", 3),
            "risk.max_sector_exposure_pct": config.get("max_sector_exposure_pct", 0.3),
            "risk.daily_loss_limit_pct": config.get("daily_loss_limit_pct", 0.05),
            "risk.trailing_stop_enabled": config.get("trailing_stop_enabled", False),
        }
        self.log_params(params)

    def log_risk_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        step: Optional[int] = None
    ) -> None:
        """Log a risk management event.

        Args:
            event_type: Type of risk event (e.g., "position_rejected", "daily_limit_hit")
            details: Event details
            step: Optional step number
        """
        if not self.is_enabled:
            return

        # Map event types to numeric codes for tracking
        event_codes = {
            "position_rejected": 1,
            "daily_limit_hit": 2,
            "drawdown_limit_hit": 3,
            "correlation_limit_hit": 4,
            "position_reduced": 5,
            "stop_loss_triggered": 6,
            "take_profit_triggered": 7,
        }

        metrics = {
            f"risk.event_{event_type}": 1,
            "risk.event_code": event_codes.get(event_type, 0),
        }

        # Add numeric details
        for key, value in details.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                metrics[f"risk.{event_type}_{key}"] = value

        self.log_metrics(metrics, step=step)


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
