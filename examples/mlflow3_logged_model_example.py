"""
MLflow 3 LoggedModel Example for Leap Trading System

This example demonstrates MLflow 3's new LoggedModel feature which provides:
- First-class model entities with unique IDs
- Dataset tracking and linking
- Metrics linked to specific models and datasets
- Enhanced model lifecycle tracking

Requirements:
    pip install 'mlflow>=3'

Usage:
    python examples/mlflow3_logged_model_example.py

References:
    - https://mlflow.org/releases/3
    - https://mlflow.org/docs/latest/genai/mlflow-3/
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Simple Trading Model (for demonstration)
# =============================================================================

class SimplePricePredictor(nn.Module):
    """Simple LSTM price predictor for demonstration."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_forex_data(
    n_samples: int = 1000,
    n_features: int = 5,
    sequence_length: int = 20
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generate synthetic forex-like data for demonstration.

    Returns:
        X: Feature sequences (n_samples, sequence_length, n_features)
        y: Target values (n_samples,)
        df: Raw DataFrame for MLflow dataset tracking
    """
    np.random.seed(42)

    # Generate synthetic price data with trend and noise
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_samples),
        periods=n_samples + sequence_length,
        freq='h'
    )

    # Simulate price with random walk
    price = 1.1000 + np.cumsum(np.random.randn(len(dates)) * 0.0010)

    # Create features (OHLCV-like)
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(len(dates)) * 0.0005,
        'high': price + np.abs(np.random.randn(len(dates)) * 0.0010),
        'low': price - np.abs(np.random.randn(len(dates)) * 0.0010),
        'close': price,
        'volume': np.random.randint(1000, 10000, len(dates))
    })

    # Create sequences
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])

    X = []
    y = []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i + sequence_length])
        # Target: next close price change (normalized)
        y.append(scaled_features[i + sequence_length, 3])  # close column

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), df


# =============================================================================
# Training with MLflow 3 LoggedModel
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # Directional accuracy (important for trading)
    direction_accuracy = np.mean(
        (y_true > 0) == (y_pred > 0)
    ) if len(y_true) > 0 else 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "direction_accuracy": float(direction_accuracy)
    }


def train_with_mlflow3():
    """
    Demonstrate MLflow 3 LoggedModel features:
    1. Create datasets and track them
    2. Log model with name and params (creates LoggedModel)
    3. Link metrics to model and dataset
    4. Log checkpoints during training
    """

    # Setup MLflow
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("leap-mlflow3-demo")

    logger.info("Generating synthetic forex data...")
    X, y, raw_df = generate_synthetic_forex_data(
        n_samples=2000,
        n_features=5,
        sequence_length=20
    )

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Model hyperparameters
    model_params = {
        "input_dim": X_train.shape[2],
        "hidden_dim": 64,
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }

    # Start MLflow run
    with mlflow.start_run(run_name="mlflow3-price-predictor") as run:
        logger.info(f"Started run: {run.info.run_id}")

        # =====================================================================
        # MLflow 3 Feature 1: Dataset Tracking
        # Create MLflow datasets from our training data
        # =====================================================================
        train_df = pd.DataFrame({
            'features': [x.flatten().tolist() for x in X_train],
            'target': y_train
        })
        val_df = pd.DataFrame({
            'features': [x.flatten().tolist() for x in X_val],
            'target': y_val
        })
        test_df = pd.DataFrame({
            'features': [x.flatten().tolist() for x in X_test],
            'target': y_test
        })

        # Create MLflow Dataset objects
        train_dataset = mlflow.data.from_pandas(train_df, name="forex_train")
        val_dataset = mlflow.data.from_pandas(val_df, name="forex_val")
        test_dataset = mlflow.data.from_pandas(test_df, name="forex_test")

        logger.info("Created MLflow datasets for train/val/test splits")

        # Log dataset inputs
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")
        mlflow.log_input(test_dataset, context="testing")

        # =====================================================================
        # Model Training
        # =====================================================================
        model = SimplePricePredictor(
            input_dim=model_params["input_dim"],
            hidden_dim=model_params["hidden_dim"]
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=model_params["learning_rate"])
        criterion = nn.MSELoss()

        # Convert to tensors
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train).unsqueeze(1)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val).unsqueeze(1)

        best_val_loss = float('inf')
        best_model_state = None

        logger.info("Starting training...")
        for epoch in range(model_params["epochs"]):
            model.train()

            # Mini-batch training
            n_batches = len(X_train_t) // model_params["batch_size"]
            epoch_loss = 0.0

            for i in range(n_batches):
                start_idx = i * model_params["batch_size"]
                end_idx = start_idx + model_params["batch_size"]

                batch_X = X_train_t[start_idx:end_idx]
                batch_y = y_train_t[start_idx:end_idx]

                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / n_batches

            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                val_loss = criterion(val_output, y_val_t).item()

            # Log epoch metrics
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": val_loss
            }, step=epoch)

            logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # =====================================================================
        # MLflow 3 Feature 2: Log Model with Name and Params (LoggedModel)
        # This creates a first-class LoggedModel entity
        # =====================================================================
        logger.info("Logging model with MLflow 3 LoggedModel...")

        # Create input example for signature
        input_example = X_train[:1]  # Shape: (1, seq_len, features)

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="price_predictor",  # MLflow 3: use 'name' instead of 'artifact_path'
            params={
                "input_dim": model_params["input_dim"],
                "hidden_dim": model_params["hidden_dim"],
                "learning_rate": model_params["learning_rate"],
            },
            input_example=input_example
        )

        logger.info(f"Logged model with ID: {model_info.model_id}")

        # =====================================================================
        # MLflow 3 Feature 3: Get LoggedModel and Inspect Properties
        # =====================================================================
        logged_model = mlflow.get_logged_model(model_info.model_id)
        logger.info(f"LoggedModel ID: {logged_model.model_id}")
        logger.info(f"LoggedModel Params: {logged_model.params}")

        # =====================================================================
        # MLflow 3 Feature 4: Link Metrics to Model AND Dataset
        # This enables powerful model-dataset-metric relationships
        # =====================================================================

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t).numpy().flatten()

        val_metrics = compute_metrics(y_val, val_predictions)

        # Log metrics linked to both model and dataset
        mlflow.log_metrics(
            metrics={
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_direction_accuracy": val_metrics["direction_accuracy"]
            },
            model_id=logged_model.model_id,
            dataset=val_dataset
        )
        logger.info(f"Logged validation metrics linked to model and dataset")

        # Evaluate on test set
        X_test_t = torch.from_numpy(X_test)
        with torch.no_grad():
            test_predictions = model(X_test_t).numpy().flatten()

        test_metrics = compute_metrics(y_test, test_predictions)

        # Log test metrics linked to model and test dataset
        mlflow.log_metrics(
            metrics={
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_direction_accuracy": test_metrics["direction_accuracy"]
            },
            model_id=logged_model.model_id,
            dataset=test_dataset
        )
        logger.info(f"Logged test metrics linked to model and dataset")

        # =====================================================================
        # MLflow 3 Feature 5: Inspect LoggedModel with Metrics
        # =====================================================================
        logged_model = mlflow.get_logged_model(model_info.model_id)
        logger.info(f"\nFinal LoggedModel State:")
        logger.info(f"  Model ID: {logged_model.model_id}")
        logger.info(f"  Params: {logged_model.params}")
        logger.info(f"  Metrics: {logged_model.metrics}")

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 70)
        print("MLflow 3 LoggedModel Demo Complete!")
        print("=" * 70)
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Model ID: {logged_model.model_id}")
        print(f"\nModel URI: models:/{logged_model.model_id}")
        print(f"\nKey MLflow 3 Features Demonstrated:")
        print("  1. Dataset tracking with mlflow.data.from_pandas()")
        print("  2. LoggedModel creation with name and params")
        print("  3. Metrics linked to specific model and dataset")
        print("  4. Model lifecycle tracking via model_id")
        print("\nView in MLflow UI: mlflow ui --port 5000")
        print("=" * 70)

        return logged_model


def demonstrate_model_search():
    """
    Demonstrate MLflow 3's model search capabilities.
    """
    print("\n" + "=" * 70)
    print("Demonstrating MLflow 3 Model Search")
    print("=" * 70)

    try:
        # Search for logged models
        logged_models = mlflow.search_logged_models(
            filter_string="name = 'price_predictor'",
            max_results=5
        )

        print(f"\nFound {len(logged_models)} models matching 'price_predictor':")
        for lm in logged_models:
            print(f"\n  Model ID: {lm.model_id}")
            print(f"  Params: {lm.params}")
            if lm.metrics:
                print(f"  Metrics: {lm.metrics}")

    except Exception as e:
        logger.warning(f"Model search not available (may need MLflow 3): {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("MLflow 3 LoggedModel Example for Leap Trading System")
    print("=" * 70)
    print("\nThis example demonstrates MLflow 3's new LoggedModel feature:")
    print("  - First-class model entities with unique IDs")
    print("  - Dataset tracking and linking")
    print("  - Metrics linked to specific models and datasets")
    print("  - Enhanced model lifecycle tracking")
    print("=" * 70 + "\n")

    # Run the demo
    logged_model = train_with_mlflow3()

    # Demonstrate search capabilities
    demonstrate_model_search()
