"""
Leap Trading System - Model Trainer
Unified training interface for all models.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING
from datetime import datetime
import logging
import os
import json
import time

from utils.device import resolve_device

if TYPE_CHECKING:
    from utils.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified trainer for prediction and RL models.

    Features:
    - Coordinated training of Transformer predictor and PPO agent
    - Curriculum learning support
    - Model checkpointing
    - Training visualization
    """

    def __init__(
        self,
        predictor,
        agent,
        data_pipeline,
        config: Optional[Dict] = None,
        device: str = 'auto',
        mlflow_tracker: Optional['MLflowTracker'] = None
    ):
        self.predictor = predictor
        self.agent = agent
        self.data_pipeline = data_pipeline
        self.config = config or {}
        self.mlflow_tracker = mlflow_tracker

        # Device setup using centralized utility
        self.device = resolve_device(device)

        # Training configuration
        self.predictor_epochs = self.config.get('predictor_epochs', 100)
        self.agent_timesteps = self.config.get('agent_timesteps', 100000)
        self.batch_size = self.config.get('batch_size', 64)
        self.patience = self.config.get('patience', 15)

        # Checkpointing
        self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Training history
        self.training_history = {
            'predictor': [],
            'agent': [],
            'combined': []
        }

    def train_predictor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict:
        """Train the prediction model."""
        epochs = epochs or self.predictor_epochs
        start_time = time.time()

        logger.info(f"Training Transformer predictor for {epochs} epochs...")
        logger.info(f"Training data shape: {X_train.shape}")

        # Create MLflow callback for epoch-level logging
        mlflow_callback = None
        if self.mlflow_tracker and self.mlflow_tracker.is_enabled:
            def mlflow_callback(metrics: Dict, epoch: int):
                self.mlflow_tracker.log_metrics({
                    "predictor.train_loss": metrics.get("train_loss", 0),
                    "predictor.val_loss": metrics.get("val_loss", 0),
                    "predictor.learning_rate": metrics.get("learning_rate", 0),
                }, step=epoch)

        results = self.predictor.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            mlflow_callback=mlflow_callback
        )

        training_duration = time.time() - start_time

        self.training_history['predictor'].append({
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'final_train_loss': results['train_losses'][-1] if results['train_losses'] else None,
            'final_val_loss': results['val_losses'][-1] if results['val_losses'] else None,
            'best_val_loss': results.get('best_val_loss'),
            'training_duration_seconds': training_duration
        })

        # Log training summary to MLflow
        if self.mlflow_tracker and self.mlflow_tracker.is_enabled:
            final_metrics = {
                "train_loss": results['train_losses'][-1] if results['train_losses'] else 0,
                "val_loss": results['val_losses'][-1] if results['val_losses'] else 0,
                "best_val_loss": results.get('best_val_loss', 0),
            }
            self.mlflow_tracker.log_training_summary(
                "predictor", final_metrics, training_duration
            )

            # Log model to MLflow
            if hasattr(self.predictor, 'model') and hasattr(self.predictor, 'input_dim'):
                self.mlflow_tracker.log_predictor_model(
                    model=self.predictor.model,
                    input_dim=self.predictor.input_dim,
                    seq_length=X_train.shape[1] if len(X_train.shape) > 1 else 120
                )

        # Save checkpoint
        self._save_predictor_checkpoint(results)

        if callbacks:
            for callback in callbacks:
                callback('predictor_trained', results)

        return results

    def train_agent(
        self,
        env,
        total_timesteps: Optional[int] = None,
        eval_env=None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict:
        """Train the RL agent."""
        timesteps = total_timesteps or self.agent_timesteps
        start_time = time.time()

        logger.info(f"Training PPO agent for {timesteps} timesteps...")

        # Create MLflow callback for step-level logging
        mlflow_callback = None
        if self.mlflow_tracker and self.mlflow_tracker.is_enabled:
            def mlflow_callback(metrics: Dict, step: int):
                mlflow_metrics = {}
                if "policy_loss" in metrics:
                    mlflow_metrics["agent.policy_loss"] = metrics["policy_loss"]
                if "value_loss" in metrics:
                    mlflow_metrics["agent.value_loss"] = metrics["value_loss"]
                if "entropy" in metrics:
                    mlflow_metrics["agent.entropy"] = metrics["entropy"]
                if "clip_fraction" in metrics:
                    mlflow_metrics["agent.clip_fraction"] = metrics["clip_fraction"]
                if "episode_reward" in metrics:
                    mlflow_metrics["agent.episode_reward"] = metrics["episode_reward"]
                if "episode_length" in metrics:
                    mlflow_metrics["agent.episode_length"] = metrics["episode_length"]

                if mlflow_metrics:
                    self.mlflow_tracker.log_metrics(mlflow_metrics, step=step)

        results = self.agent.train_on_env(
            env=env,
            total_timesteps=timesteps,
            eval_env=eval_env,
            eval_frequency=10000,
            mlflow_callback=mlflow_callback
        )

        training_duration = time.time() - start_time

        final_reward = np.mean(results['episode_rewards'][-10:]) if results['episode_rewards'] else None

        self.training_history['agent'].append({
            'timestamp': datetime.now().isoformat(),
            'timesteps': timesteps,
            'final_reward': final_reward,
            'total_episodes': len(results['episode_rewards']),
            'training_duration_seconds': training_duration
        })

        # Log training summary to MLflow
        if self.mlflow_tracker and self.mlflow_tracker.is_enabled:
            final_metrics = {
                "final_reward": final_reward if final_reward is not None else 0,
                "total_episodes": len(results['episode_rewards']),
                "avg_episode_length": np.mean(results.get('episode_lengths', [0])) if results.get('episode_lengths') else 0,
            }
            self.mlflow_tracker.log_training_summary(
                "agent", final_metrics, training_duration
            )

            # Log model to MLflow
            if hasattr(self.agent, 'network') and hasattr(self.agent, 'state_dim'):
                self.mlflow_tracker.log_agent_model(
                    model=self.agent.network,
                    state_dim=self.agent.state_dim
                )

        # Save checkpoint
        self._save_agent_checkpoint(results)

        if callbacks:
            for callback in callbacks:
                callback('agent_trained', results)

        return results

    def train_combined(
        self,
        market_data,
        env,
        predictor_epochs: Optional[int] = None,
        agent_timesteps: Optional[int] = None,
        curriculum_stages: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Train both models in a coordinated manner.

        Optionally uses curriculum learning with increasing difficulty.
        """
        results = {
            'predictor': None,
            'agent': None,
            'stages': []
        }

        if curriculum_stages:
            # Curriculum learning
            for i, stage in enumerate(curriculum_stages):
                logger.info(f"Curriculum stage {i + 1}/{len(curriculum_stages)}: {stage.get('name', 'unnamed')}")

                stage_result = self._train_curriculum_stage(
                    market_data=market_data,
                    env=env,
                    stage_config=stage
                )
                results['stages'].append(stage_result)

        else:
            # Standard training
            # 1. Prepare data for predictor
            X, y, _ = self.data_pipeline.prepare_sequences(market_data)
            split = self.data_pipeline.create_train_val_test_split(X, y)

            # 2. Train predictor
            results['predictor'] = self.train_predictor(
                X_train=split['train'][0],
                y_train=split['train'][1],
                X_val=split['val'][0],
                y_val=split['val'][1],
                epochs=predictor_epochs
            )

            # 3. Train agent
            results['agent'] = self.train_agent(
                env=env,
                total_timesteps=agent_timesteps
            )

        self.training_history['combined'].append({
            'timestamp': datetime.now().isoformat(),
            'results_summary': {
                'predictor_trained': results['predictor'] is not None,
                'agent_trained': results['agent'] is not None,
                'curriculum_stages': len(results['stages'])
            }
        })

        return results

    def _train_curriculum_stage(
        self,
        market_data,
        env,
        stage_config: Dict
    ) -> Dict:
        """Train a single curriculum stage."""
        stage_name = stage_config.get('name', 'unnamed')
        difficulty = stage_config.get('difficulty', 1.0)

        # Adjust training parameters based on difficulty
        adjusted_epochs = int(self.predictor_epochs * difficulty)
        adjusted_timesteps = int(self.agent_timesteps * difficulty)

        # Apply any stage-specific modifications
        if 'data_filter' in stage_config:
            # Filter data based on stage requirements
            pass

        # Prepare data
        X, y, _ = self.data_pipeline.prepare_sequences(market_data)
        split = self.data_pipeline.create_train_val_test_split(X, y)

        # Train
        predictor_result = self.train_predictor(
            X_train=split['train'][0],
            y_train=split['train'][1],
            X_val=split['val'][0],
            y_val=split['val'][1],
            epochs=adjusted_epochs
        )

        agent_result = self.train_agent(
            env=env,
            total_timesteps=adjusted_timesteps
        )

        return {
            'stage': stage_name,
            'difficulty': difficulty,
            'predictor': predictor_result,
            'agent': agent_result
        }

    def evaluate(
        self,
        test_data,
        env,
        n_eval_episodes: int = 10
    ) -> Dict:
        """Evaluate both models."""
        results = {}

        # Evaluate predictor
        if test_data is not None:
            X_test, y_test, _ = self.data_pipeline.prepare_sequences(test_data)

            predictions = self.predictor.predict(X_test)
            pred_error = np.mean(np.abs(predictions['prediction'].flatten() - y_test))

            results['predictor'] = {
                'mae': float(pred_error),
                'samples': len(y_test)
            }

        # Evaluate agent
        if env is not None:
            avg_reward = self.agent.evaluate(env, n_episodes=n_eval_episodes)
            results['agent'] = {
                'avg_reward': float(avg_reward),
                'episodes': n_eval_episodes
            }

        return results

    def _save_predictor_checkpoint(self, results: Dict):
        """Save predictor checkpoint."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.checkpoint_dir, f'predictor_{timestamp}.pt')
        self.predictor.save(path)

        # Save training info
        info_path = os.path.join(self.checkpoint_dir, f'predictor_{timestamp}_info.json')
        with open(info_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'best_val_loss': results.get('best_val_loss'),
                'epochs': len(results.get('train_losses', []))
            }, f)

    def _save_agent_checkpoint(self, results: Dict):
        """Save agent checkpoint."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.checkpoint_dir, f'agent_{timestamp}.pt')
        self.agent.save(path)

        # Save training info
        info_path = os.path.join(self.checkpoint_dir, f'agent_{timestamp}_info.json')
        with open(info_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_episodes': len(results.get('episode_rewards', [])),
                'final_reward': float(np.mean(results.get('episode_rewards', [0])[-10:]))
            }, f)

    def save_all(self, directory: str):
        """Save all models and training history."""
        os.makedirs(directory, exist_ok=True)

        self.predictor.save(os.path.join(directory, 'predictor.pt'))
        self.agent.save(os.path.join(directory, 'agent.pt'))

        with open(os.path.join(directory, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)

        logger.info(f"All models saved to {directory}")

    def load_all(self, directory: str):
        """Load all models."""
        self.predictor.load(os.path.join(directory, 'predictor.pt'))
        self.agent.load(os.path.join(directory, 'agent.pt'))

        history_path = os.path.join(directory, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)

        logger.info(f"All models loaded from {directory}")

    def get_training_summary(self) -> Dict:
        """Get summary of all training runs."""
        return {
            'predictor_runs': len(self.training_history['predictor']),
            'agent_runs': len(self.training_history['agent']),
            'combined_runs': len(self.training_history['combined']),
            'latest_predictor': self.training_history['predictor'][-1] if self.training_history['predictor'] else None,
            'latest_agent': self.training_history['agent'][-1] if self.training_history['agent'] else None
        }
