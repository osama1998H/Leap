"""
Leap Trading System - TransformerPredictor Tests
Comprehensive tests for the Transformer prediction model.
"""

import pytest
import numpy as np
import torch
import os
import tempfile

# Import components to test
from models.transformer import (
    TransformerPredictor, TemporalFusionTransformer,
    PositionalEncoding, MultiHeadAttention, FeedForward,
    TransformerEncoderLayer, GatedResidualNetwork
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_input():
    """Create sample input data."""
    batch_size = 16
    seq_length = 60
    input_dim = 25

    X = np.random.randn(batch_size, seq_length, input_dim).astype(np.float32)
    y = np.random.randn(batch_size).astype(np.float32)

    return X, y


@pytest.fixture
def training_data():
    """Create training and validation data."""
    n_train = 200
    n_val = 50
    seq_length = 60
    input_dim = 25

    X_train = np.random.randn(n_train, seq_length, input_dim).astype(np.float32)
    y_train = np.random.randn(n_train).astype(np.float32)
    X_val = np.random.randn(n_val, seq_length, input_dim).astype(np.float32)
    y_val = np.random.randn(n_val).astype(np.float32)

    return X_train, y_train, X_val, y_val


@pytest.fixture
def predictor():
    """Create TransformerPredictor for testing."""
    return TransformerPredictor(
        input_dim=25,
        config={
            'd_model': 32,
            'n_heads': 4,
            'n_encoder_layers': 2,
            'd_ff': 64,
            'dropout': 0.1,
            'max_seq_length': 60,
            'learning_rate': 1e-3
        },
        device='cpu'
    )


# ============================================================================
# PositionalEncoding Tests
# ============================================================================

class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        """Test output shape matches input shape."""
        d_model = 32
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)

        # Input: (seq_len, batch, d_model)
        x = torch.randn(50, 8, d_model)
        output = pe(x)

        assert output.shape == x.shape

    def test_positional_values_unique(self):
        """Test that positional encodings are unique per position."""
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.0)

        x = torch.zeros(10, 1, 32)
        output = pe(x)

        # Each position should have different encoding
        for i in range(9):
            assert not torch.allclose(output[i], output[i + 1])

    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        pe = PositionalEncoding(d_model=32, max_len=100, dropout=0.5)
        pe.train()

        x = torch.ones(10, 8, 32)

        # Run multiple times, should get different results
        outputs = [pe(x) for _ in range(5)]

        # At least some outputs should differ
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same


# ============================================================================
# MultiHeadAttention Tests
# ============================================================================

class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    def test_output_shape(self):
        """Test output shape is correct."""
        d_model = 32
        n_heads = 4
        mha = MultiHeadAttention(d_model, n_heads)

        batch_size = 8
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)

        output, attn_weights = mha(x, x, x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1."""
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        x = torch.randn(4, 10, 32)

        _, attn_weights = mha(x, x, x)

        # Attention weights should sum to 1 along last dimension
        weight_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_masking(self):
        """Test that masking works correctly."""
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        x = torch.randn(4, 10, 32)

        # Create causal mask
        mask = torch.tril(torch.ones(10, 10))

        output, _attn_weights = mha(x, x, x, mask=mask)

        assert output.shape == (4, 10, 32)


# ============================================================================
# FeedForward Tests
# ============================================================================

class TestFeedForward:
    """Tests for FeedForward module."""

    def test_output_shape(self):
        """Test output shape matches input shape."""
        d_model = 32
        d_ff = 128
        ff = FeedForward(d_model, d_ff)

        x = torch.randn(8, 10, d_model)
        output = ff(x)

        assert output.shape == x.shape


# ============================================================================
# TransformerEncoderLayer Tests
# ============================================================================

class TestTransformerEncoderLayer:
    """Tests for TransformerEncoderLayer module."""

    def test_output_shape(self):
        """Test output shape is correct."""
        layer = TransformerEncoderLayer(d_model=32, n_heads=4, d_ff=64)

        x = torch.randn(8, 10, 32)
        output, attn = layer(x)

        assert output.shape == x.shape
        assert attn.shape[0] == 8  # batch size

    def test_residual_connection(self):
        """Test residual connections don't explode gradients."""
        layer = TransformerEncoderLayer(d_model=32, n_heads=4, d_ff=64)

        x = torch.randn(8, 10, 32, requires_grad=True)
        output, _ = layer(x)

        # Backprop should work
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))


# ============================================================================
# GatedResidualNetwork Tests
# ============================================================================

class TestGatedResidualNetwork:
    """Tests for GatedResidualNetwork module."""

    def test_output_shape_same_dim(self):
        """Test output shape when input and output dims are same."""
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=32)

        x = torch.randn(8, 32)
        output = grn(x)

        assert output.shape == x.shape

    def test_output_shape_different_dim(self):
        """Test output shape when input and output dims differ."""
        grn = GatedResidualNetwork(input_dim=32, hidden_dim=64, output_dim=16)

        x = torch.randn(8, 32)
        output = grn(x)

        assert output.shape == (8, 16)

    def test_with_context(self):
        """Test GRN with context input."""
        grn = GatedResidualNetwork(
            input_dim=32,
            hidden_dim=64,
            output_dim=32,
            context_dim=16
        )

        x = torch.randn(8, 32)
        context = torch.randn(8, 16)
        output = grn(x, context)

        assert output.shape == (8, 32)


# ============================================================================
# TemporalFusionTransformer Tests
# ============================================================================

class TestTemporalFusionTransformer:
    """Tests for TemporalFusionTransformer module."""

    @pytest.fixture
    def tft_model(self):
        """Create TFT model for testing."""
        return TemporalFusionTransformer(
            input_dim=25,
            d_model=32,
            n_heads=4,
            n_encoder_layers=2,
            d_ff=64,
            max_seq_length=60
        )

    def test_forward_output_keys(self, tft_model):
        """Test forward returns expected keys."""
        x = torch.randn(8, 30, 25)
        output = tft_model(x)

        assert 'prediction' in output
        assert 'quantiles' in output

    def test_forward_prediction_shape(self, tft_model):
        """Test prediction output shape."""
        batch_size = 8
        x = torch.randn(batch_size, 30, 25)
        output = tft_model(x)

        assert output['prediction'].shape == (batch_size, 1)

    def test_forward_quantiles_shape(self, tft_model):
        """Test quantiles output shape."""
        batch_size = 8
        x = torch.randn(batch_size, 30, 25)
        output = tft_model(x)

        # Default: 3 quantiles (0.1, 0.5, 0.9)
        assert output['quantiles'].shape == (batch_size, 3, 1)

    def test_forward_with_attention(self, tft_model):
        """Test forward with attention weights returned."""
        x = torch.randn(8, 30, 25)
        output = tft_model(x, return_attention=True)

        assert 'encoder_attention' in output
        assert 'temporal_attention' in output

    def test_no_nan_outputs(self, tft_model):
        """Test that model doesn't produce NaN outputs."""
        x = torch.randn(8, 30, 25)
        output = tft_model(x)

        assert not torch.any(torch.isnan(output['prediction']))
        assert not torch.any(torch.isnan(output['quantiles']))


# ============================================================================
# TransformerPredictor Initialization Tests
# ============================================================================

class TestTransformerPredictorInitialization:
    """Tests for TransformerPredictor initialization."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        predictor = TransformerPredictor(input_dim=25)

        assert predictor.input_dim == 25
        assert predictor.model is not None

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = {
            'd_model': 64,
            'n_heads': 8,
            'n_encoder_layers': 3
        }
        predictor = TransformerPredictor(input_dim=25, config=config)

        assert predictor.d_model == 64
        assert predictor.n_heads == 8
        assert predictor.n_encoder_layers == 3

    def test_initialization_device_cpu(self):
        """Test initialization with CPU device."""
        predictor = TransformerPredictor(input_dim=25, device='cpu')

        assert predictor.device.type == 'cpu'

    def test_initialization_device_auto(self):
        """Test initialization with auto device."""
        predictor = TransformerPredictor(input_dim=25, device='auto')

        # Should be either cuda or cpu
        assert predictor.device.type in ['cuda', 'cpu']


# ============================================================================
# TransformerPredictor Training Tests
# ============================================================================

class TestTransformerPredictorTraining:
    """Tests for TransformerPredictor training."""

    def test_train_returns_results(self, predictor, training_data):
        """Test that train returns results dictionary."""
        X_train, y_train, X_val, y_val = training_data

        results = predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=2,
            batch_size=32,
            verbose=False
        )

        assert isinstance(results, dict)
        assert 'train_losses' in results
        assert 'val_losses' in results

    def test_train_records_losses_per_epoch(self, predictor, training_data):
        """Test that training records loss for each epoch."""
        X_train, y_train, X_val, y_val = training_data

        results = predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=5,
            batch_size=32,
            verbose=False
        )

        train_losses = results['train_losses']
        # Should have one loss value per epoch
        assert len(train_losses) == 5

    def test_train_with_callback(self, predictor, training_data):
        """Test training with mlflow callback."""
        X_train, y_train, X_val, y_val = training_data

        callback_calls = []

        def test_callback(metrics, epoch):
            callback_calls.append((metrics, epoch))

        predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=2,
            batch_size=32,
            verbose=False,
            mlflow_callback=test_callback
        )

        assert len(callback_calls) == 2


# ============================================================================
# TransformerPredictor Prediction Tests
# ============================================================================

class TestTransformerPredictorPrediction:
    """Tests for TransformerPredictor prediction."""

    def test_predict_returns_dict(self, predictor, sample_input):
        """Test predict returns dictionary."""
        X, _ = sample_input
        result = predictor.predict(X)

        assert isinstance(result, dict)

    def test_predict_contains_prediction(self, predictor, sample_input):
        """Test predict result contains prediction."""
        X, _ = sample_input
        result = predictor.predict(X)

        assert 'prediction' in result
        assert result['prediction'].shape[0] == X.shape[0]

    def test_predict_contains_quantiles(self, predictor, sample_input):
        """Test predict result contains quantiles."""
        X, _ = sample_input
        result = predictor.predict(X)

        assert 'quantiles' in result

    def test_predict_with_uncertainty(self, predictor, sample_input):
        """Test predict with uncertainty estimation."""
        X, _ = sample_input
        result = predictor.predict(X, return_uncertainty=True)

        assert 'uncertainty' in result
        assert result['uncertainty'].shape[0] == X.shape[0]

    def test_predict_no_nan(self, predictor, sample_input):
        """Test predictions contain no NaN values."""
        X, _ = sample_input
        result = predictor.predict(X)

        assert not np.any(np.isnan(result['prediction']))

    def test_predict_no_inf(self, predictor, sample_input):
        """Test predictions contain no infinite values."""
        X, _ = sample_input
        result = predictor.predict(X)

        assert not np.any(np.isinf(result['prediction']))


# ============================================================================
# TransformerPredictor Online Learning Tests
# ============================================================================

class TestTransformerPredictorOnlineLearning:
    """Tests for TransformerPredictor online learning."""

    def test_online_update_returns_loss(self, predictor, sample_input):
        """Test online_update returns loss value."""
        X, y = sample_input

        loss = predictor.online_update(X[:8], y[:8])

        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_online_update_modifies_weights(self, predictor, sample_input):
        """Test online_update modifies model weights."""
        X, y = sample_input

        # Get initial prediction
        initial_pred = predictor.predict(X[:4])

        # Perform online update
        predictor.online_update(X[:8], y[:8])

        # Get updated prediction
        updated_pred = predictor.predict(X[:4])

        # Predictions should be different after update
        assert not np.allclose(
            initial_pred['prediction'],
            updated_pred['prediction']
        )


# ============================================================================
# TransformerPredictor Save/Load Tests
# ============================================================================

class TestTransformerPredictorSaveLoad:
    """Tests for TransformerPredictor save/load."""

    def test_save_creates_file(self, predictor):
        """Test save creates model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            predictor.save(path)

            assert os.path.exists(path)

    def test_load_restores_model(self, predictor, sample_input):
        """Test load restores model correctly."""
        X, _ = sample_input

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')

            # Get prediction before save
            pred_before = predictor.predict(X)

            # Save
            predictor.save(path)

            # Create new predictor and load
            new_predictor = TransformerPredictor(
                input_dim=25,
                config=predictor.config,
                device='cpu'
            )
            new_predictor.load(path)

            # Get prediction after load
            pred_after = new_predictor.predict(X)

            # Predictions should be identical
            np.testing.assert_array_almost_equal(
                pred_before['prediction'],
                pred_after['prediction'],
                decimal=5
            )


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample_prediction(self, predictor):
        """Test prediction with single sample."""
        X = np.random.randn(1, 60, 25).astype(np.float32)
        result = predictor.predict(X)

        assert result['prediction'].shape == (1, 1)

    def test_large_batch_prediction(self, predictor):
        """Test prediction with large batch."""
        X = np.random.randn(256, 60, 25).astype(np.float32)
        result = predictor.predict(X)

        assert result['prediction'].shape[0] == 256

    def test_variable_sequence_length(self):
        """Test model handles different sequence lengths."""
        predictor = TransformerPredictor(
            input_dim=25,
            config={'max_seq_length': 120},
            device='cpu'
        )

        # Test with different sequence lengths
        for seq_len in [30, 60, 90, 120]:
            X = np.random.randn(8, seq_len, 25).astype(np.float32)
            result = predictor.predict(X)
            assert result['prediction'].shape == (8, 1)

    def test_training_with_small_batch(self, predictor):
        """Test training with batch size smaller than data."""
        X_train = np.random.randn(10, 60, 25).astype(np.float32)
        y_train = np.random.randn(10).astype(np.float32)
        X_val = np.random.randn(5, 60, 25).astype(np.float32)
        y_val = np.random.randn(5).astype(np.float32)

        results = predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=1,
            batch_size=32,  # Larger than data
            verbose=False
        )

        assert 'train_losses' in results


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
