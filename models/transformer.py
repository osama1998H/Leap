"""
Leap Trading System - Transformer-based Prediction Model
Implements Temporal Fusion Transformer for time series forecasting.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.w_o(context)

        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature selection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        # Primary pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Context integration
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        # Gating mechanism
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        # Skip connection
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Primary pathway
        hidden = F.elu(self.fc1(x))

        if self.context_dim is not None and context is not None:
            hidden = hidden + self.context_proj(context)

        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)

        # Gating
        gate = self.sigmoid(self.gate(F.elu(self.fc1(x))))
        gated = gate * hidden

        # Skip connection
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        return self.norm(skip + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable selection network for feature importance."""

    def __init__(
        self,
        input_dim: int,
        n_features: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Feature-wise GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim // n_features,
                hidden_dim,
                hidden_dim,
                dropout,
                context_dim
            )
            for _ in range(n_features)
        ])

        # Variable selection weights
        self.weight_grn = GatedResidualNetwork(
            input_dim,
            hidden_dim,
            n_features,
            dropout,
            context_dim
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split input into features
        batch_size, seq_len, _ = x.shape
        feature_dim = x.shape[-1] // self.n_features

        # Process each feature
        processed_features = []
        for i, grn in enumerate(self.feature_grns):
            feature_input = x[:, :, i * feature_dim:(i + 1) * feature_dim]
            processed = grn(feature_input, context)
            processed_features.append(processed)

        # Stack features: (batch, seq, n_features, hidden)
        processed_features = torch.stack(processed_features, dim=2)

        # Calculate variable selection weights
        weights = self.weight_grn(x.reshape(batch_size * seq_len, -1), context)
        weights = self.softmax(weights).view(batch_size, seq_len, self.n_features, 1)

        # Apply weights
        selected = (processed_features * weights).sum(dim=2)

        return selected, weights.squeeze(-1)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for time series prediction.

    Features:
    - Variable selection networks
    - Gated residual networks
    - Multi-head attention with interpretable weights
    - Quantile predictions for uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 120,
        output_dim: int = 1,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)

        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # Temporal self-attention for interpretability
        self.temporal_attention = MultiHeadAttention(d_model, n_heads, dropout)

        # Output layers
        self.pre_output = GatedResidualNetwork(d_model, d_ff, d_model, dropout)

        # Quantile outputs for uncertainty estimation
        self.quantile_outputs = nn.ModuleList([
            nn.Linear(d_model, output_dim)
            for _ in quantiles
        ])

        # Point prediction
        self.point_output = nn.Linear(d_model, output_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with predictions and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape

        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Encoder layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x)
            attention_weights.append(attn)

        # Temporal attention for interpretability
        temporal_context, temporal_attn = self.temporal_attention(x, x, x)
        x = self.norm(x + temporal_context)

        # Take the last timestep for prediction
        x = x[:, -1, :]  # (batch, d_model)

        # Pre-output processing
        x = self.pre_output(x)

        # Point prediction
        point_pred = self.point_output(x)

        # Quantile predictions
        quantile_preds = [
            output(x) for output in self.quantile_outputs
        ]
        quantile_preds = torch.stack(quantile_preds, dim=1)  # (batch, n_quantiles, output_dim)

        result = {
            'prediction': point_pred,
            'quantiles': quantile_preds,
        }

        if return_attention:
            result['encoder_attention'] = attention_weights
            result['temporal_attention'] = temporal_attn

        return result


class TransformerPredictor:
    """
    High-level interface for training and using the Transformer model.
    Supports online learning and model adaptation.
    """

    def __init__(
        self,
        input_dim: int,
        config: Optional[Dict] = None,
        device: str = 'auto'
    ):
        self.input_dim = input_dim
        self.config = config or {}

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model parameters
        self.d_model = self.config.get('d_model', 128)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_encoder_layers = self.config.get('n_encoder_layers', 4)
        self.d_ff = self.config.get('d_ff', 512)
        self.dropout = self.config.get('dropout', 0.1)
        self.max_seq_length = self.config.get('max_seq_length', 120)

        # Initialize model
        self.model = TemporalFusionTransformer(
            input_dim=input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length
        ).to(self.device)

        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 1e-5)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        verbose: bool = True
    ) -> Dict:
        """Train the model."""
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)

        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
            if len(y_val.shape) == 1:
                y_val = y_val.unsqueeze(1)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()

                output = self.model(batch_X)
                loss = self._compute_loss(output, batch_y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)

            # Validation
            if X_val is not None:
                val_loss = self._validate(X_val, y_val)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.6f} - "
                        f"Val Loss: {val_loss:.6f}"
                    )

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }

    def _compute_loss(self, output: Dict, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        # Point prediction loss (MSE)
        point_loss = F.mse_loss(output['prediction'], target)

        # Quantile loss
        quantile_loss = 0.0
        for i, q in enumerate(self.model.quantiles):
            pred = output['quantiles'][:, i, :]
            errors = target - pred
            quantile_loss += torch.mean(torch.max(q * errors, (q - 1) * errors))

        return point_loss + 0.5 * quantile_loss

    def _validate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Validate the model."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_val)
            loss = self._compute_loss(output, y_val)
        self.model.train()
        return loss.item()

    def predict(
        self,
        X: np.ndarray,
        return_uncertainty: bool = False
    ) -> Dict[str, np.ndarray]:
        """Make predictions."""
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            output = self.model(X, return_attention=True)

        result = {
            'prediction': output['prediction'].cpu().numpy(),
            'quantiles': output['quantiles'].cpu().numpy()
        }

        if return_uncertainty:
            # Uncertainty from quantile spread
            q_low = output['quantiles'][:, 0, :].cpu().numpy()
            q_high = output['quantiles'][:, -1, :].cpu().numpy()
            result['uncertainty'] = q_high - q_low

        return result

    def online_update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        learning_rate: Optional[float] = None
    ):
        """
        Online learning: Update model with new data.
        Uses a smaller learning rate to avoid catastrophic forgetting.
        """
        if learning_rate is None:
            learning_rate = self.config.get('online_learning_rate', 1e-5)

        # Set online learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        X = torch.FloatTensor(X_new).to(self.device)
        y = torch.FloatTensor(y_new).to(self.device)

        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(X)
        loss = self._compute_loss(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        # Reset learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        return loss.item()

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'input_dim': self.input_dim,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}")

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for interpretability."""
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            output = self.model(X, return_attention=True)

        return output['temporal_attention'].cpu().numpy()
