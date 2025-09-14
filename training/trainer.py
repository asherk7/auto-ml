import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from config import config
from models.cv_models import CVModelFactory
from models.tabular_models import TabularModelFactory
from utils.database import db
from data.ingestion import data_ingestion
from models.tabular_models import save_model

logger = logging.getLogger(__name__)


class CVTrainer:
    """Trainer for computer vision models using PyTorch Lightning"""

    def __init__(self, experiment_name: str = None, wandb_project: str = None):
        self.experiment_name = experiment_name or f"cv_experiment_{int(time.time())}"
        self.wandb_project = wandb_project or config.experiment.wandb_project
        self.experiment_id = None
        self.best_model = None
        self.trainer = None

    def train(self, model: pl.LightningModule, train_loader, val_loader,
              test_loader=None, max_epochs: int = None) -> Dict[str, Any]:
        """Train a computer vision model"""
        logger.info(f"Starting CV training: {self.experiment_name}")

        # Create experiment record
        self.experiment_id = db.create_experiment(
            name=self.experiment_name,
            task_type="cv_classification",
            dataset_name="custom", 
            model_type=model.__class__.__name__,
            config_dict=self._get_config_dict()
        )

        # Setup callbacks and logger
        callbacks = self._setup_callbacks()
        wandb_logger = self._setup_wandb_logger()

        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs or config.training.max_epochs,
            accelerator=config.training.accelerator,
            devices=config.training.devices,
            precision=config.training.precision,
            callbacks=callbacks,
            logger=wandb_logger,
            enable_progress_bar=True,
            enable_model_summary=True
        )

        # Train model
        start_time = datetime.now()
        try:
            self.trainer.fit(model, train_loader, val_loader)

            test_results = {}
            if test_loader:
                test_results = self.trainer.test(model, test_loader)[0]

            best_metrics = self._extract_best_metrics()

            model_path = self._save_model(model)
            onnx_path = self._export_to_onnx(model, train_loader)

            # Update experiment
            end_time = datetime.now()
            all_metrics = {**best_metrics, **test_results}

            db.update_experiment(
                self.experiment_id,
                status="completed",
                metrics=all_metrics,
                end_time=end_time
            )

            # Save model metadata
            model_id = db.save_model(
                experiment_id=self.experiment_id,
                name=f"{self.experiment_name}_best",
                model_type=model.__class__.__name__,
                task_type="cv_classification",
                file_path=model_path,
                onnx_path=onnx_path,
                metrics=all_metrics,
                parameters=self._get_model_parameters(model),
                is_best=True
            )

            logger.info(f"Training completed successfully. Model ID: {model_id}")

            return {
                "experiment_id": self.experiment_id,
                "model_id": model_id,
                "metrics": all_metrics,
                "model_path": model_path,
                "onnx_path": onnx_path,
                "training_time": (end_time - start_time).total_seconds()
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            db.update_experiment(
                self.experiment_id,
                status="failed",
                end_time=datetime.now()
            )
            raise

    def _setup_callbacks(self) -> List[pl.Callback]:
        """Setup PyTorch Lightning callbacks"""
        callbacks = []

        early_stopping = EarlyStopping(
            monitor=config.training.monitor,
            patience=config.training.early_stopping_patience,
            mode=config.training.mode,
            verbose=True
        )
        callbacks.append(early_stopping)

        # Model checkpointing
        checkpoint_dir = Path(config.experiment.experiment_dir) / self.experiment_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:02d}-{val_accuracy:.3f}",
            monitor=config.training.monitor,
            mode=config.training.mode,
            save_top_k=config.training.save_top_k,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        return callbacks

    def _setup_wandb_logger(self) -> Optional[WandbLogger]:
        """Setup Weights & Biases logger"""
        try:
            wandb_logger = WandbLogger(
                project=self.wandb_project,
                name=self.experiment_name,
                entity=config.experiment.wandb_entity,
                log_model=config.experiment.log_model
            )
            return wandb_logger
        except Exception as e:
            logger.warning(f"Failed to setup W&B logger: {e}")
            return None

    def _extract_best_metrics(self) -> Dict[str, float]:
        """Extract best metrics from trainer"""
        if not self.trainer or not self.trainer.callback_metrics:
            return {}

        metrics = {}
        for key, value in self.trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.item())
            else:
                metrics[key] = float(value)

        return metrics

    def _save_model(self, model: pl.LightningModule) -> str:
        model_dir = Path(config.serving.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{self.experiment_name}_model.pth"
        torch.save(model.state_dict(), model_path)

        logger.info(f"Model saved to {model_path}")
        return str(model_path)

    def _export_to_onnx(self, model: pl.LightningModule, train_loader) -> Optional[str]:
        try:
            onnx_dir = Path(config.serving.onnx_dir)
            onnx_dir.mkdir(parents=True, exist_ok=True)

            onnx_path = onnx_dir / f"{self.experiment_name}_model.onnx"

            # Get sample input
            sample_batch = next(iter(train_loader))
            sample_input = sample_batch[0][:1] # First sample

            model.eval()
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )

            logger.info(f"Model exported to ONNX: {onnx_path}")
            return str(onnx_path)

        except Exception as e:
            logger.warning(f"Failed to export to ONNX: {e}")
            return None

    def _get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "training": {
                "max_epochs": config.training.max_epochs,
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "batch_size": config.data.batch_size
            },
            "model": {
                "backbone": config.model.cv_backbone,
                "pretrained": config.model.cv_pretrained
            }
        }

    def _get_model_parameters(self, model: pl.LightningModule) -> Dict[str, Any]:
        """Get model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_class": model.__class__.__name__,
            "hyperparameters": dict(model.hparams) if hasattr(model, 'hparams') else {}
        }


class TabularTrainer:
    """Trainer for tabular ML models using scikit-learn"""

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"tabular_experiment_{int(time.time())}"
        self.experiment_id = None

    def train(self, model_type: str, task_type: str, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None,
              feature_names: List[str] = None, target_names: List[str] = None,
              hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """Train a tabular ML model"""
        logger.info(f"Starting tabular training: {self.experiment_name}")

        # Create experiment record
        self.experiment_id = db.create_experiment(
            name=self.experiment_name,
            task_type=f"tabular_{task_type}",
            dataset_name="custom",
            model_type=model_type,
            config_dict=self._get_config_dict(model_type, task_type)
        )

        start_time = datetime.now()

        try:
            # Create model
            model = TabularModelFactory.create_model(task_type, model_type)

            # Hyperparameter tuning if requested
            if hyperparameter_tuning:
                model = self._tune_hyperparameters(model, X_train, y_train, task_type)

            # Train model
            if task_type == "clustering":
                train_metrics = model.fit(X_train, feature_names)
                val_metrics = {}
                test_metrics = {}
            else:
                train_metrics = model.fit(X_train, y_train, feature_names, target_names)

                # Validate model
                val_metrics = model.evaluate(X_val, y_val)

                # Test model if test data provided
                test_metrics = {}
                if X_test is not None and y_test is not None:
                    test_metrics = model.evaluate(X_test, y_test)

            # Cross-validation
            if task_type != "clustering":
                cv_metrics = model.cross_validate(X_train, y_train)
            else:
                cv_metrics = {}

            # Combine all metrics
            all_metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                **cv_metrics
            }

            # Save model
            model_path = self._save_model(model)

            # Log to W&B if available
            self._log_to_wandb(all_metrics, model)

            # Update experiment
            end_time = datetime.now()
            db.update_experiment(
                self.experiment_id,
                status="completed",
                metrics=all_metrics,
                end_time=end_time
            )

            # Save model metadata
            model_id = db.save_model(
                experiment_id=self.experiment_id,
                name=f"{self.experiment_name}_model",
                model_type=model_type,
                task_type=f"tabular_{task_type}",
                file_path=model_path,
                metrics=all_metrics,
                parameters=self._get_model_parameters(model),
                is_best=True
            )

            logger.info(f"Training completed successfully. Model ID: {model_id}")

            return {
                "experiment_id": self.experiment_id,
                "model_id": model_id,
                "model": model,
                "metrics": all_metrics,
                "model_path": model_path,
                "training_time": (end_time - start_time).total_seconds()
            }

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            db.update_experiment(
                self.experiment_id,
                status="failed",
                end_time=datetime.now()
            )
            raise

    def _tune_hyperparameters(self, model, X: np.ndarray, y: np.ndarray, task_type: str):
        """Perform hyperparameter tuning using GridSearchCV"""
        logger.info("Starting hyperparameter tuning...")

        # Define parameter grids for different models
        param_grids = {
            "random_forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10]
            },
            "logistic_regression": {
                "model__C": [0.1, 1.0, 10.0],
                "model__max_iter": [1000, 2000]
            },
            "svm": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["rbf", "linear"]
            },
            "knn": {
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"]
            }
        }

        param_grid = param_grids.get(model.model_type, {})

        if not param_grid:
            logger.warning(f"No parameter grid defined for {model.model_type}")
            return model

        # Prepare data
        X_scaled = model.scaler.fit_transform(X)

        # Setup scoring
        scoring = "accuracy" if task_type == "classification" else "r2"

        # Perform grid search
        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_scaled, y)

        # Update model with best parameters
        model.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return model

    def _save_model(self, model) -> str:
        model_dir = Path(config.serving.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{self.experiment_name}_model.joblib"
        save_model(model, str(model_path))

        return str(model_path)

    def _log_to_wandb(self, metrics: Dict[str, Any], model) -> None:
        """Log metrics to Weights & Biases"""
        try:
            wandb.init(
                project=config.experiment.wandb_project,
                name=self.experiment_name,
                entity=config.experiment.wandb_entity
            )

            wandb.log(metrics)

            # Log feature importance if available
            feature_importance = model.get_feature_importance()
            if feature_importance:
                wandb.log({"feature_importance": wandb.Table(
                    data=[[k, v] for k, v in feature_importance.items()],
                    columns=["feature", "importance"]
                )})

            wandb.finish()

        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def _get_config_dict(self, model_type: str, task_type: str) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "model_type": model_type,
            "task_type": task_type,
            "data_splits": {
                "train": config.data.train_split,
                "val": config.data.val_split,
                "test": config.data.test_split
            }
        }

    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Get model parameters"""
        params = {
            "model_type": model.model_type,
            "is_fitted": model.is_fitted,
            "sklearn_params": model.model.get_params() if hasattr(model.model, 'get_params') else {}
        }

        if hasattr(model.model, 'n_features_in_'):
            params["n_features"] = model.model.n_features_in_

        return params


class AutoMLTrainer:
    """Main AutoML trainer that coordinates different model types"""

    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"automl_experiment_{int(time.time())}"

    def auto_train(self, data_type: str, task_type: str, data_path: str = None,
                   target_column: str = None, models_to_try: List[str] = None,
                   max_trials: int = 5) -> Dict[str, Any]:
        """Automatically train and compare multiple models"""
        logger.info(f"Starting AutoML training: {self.experiment_name}")

        results = {}

        if data_type == "image":
            results = self._auto_train_cv(task_type, data_path, models_to_try, max_trials)
        elif data_type == "tabular":
            results = self._auto_train_tabular(task_type, data_path, target_column, models_to_try, max_trials)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        # Find best model
        best_model = self._find_best_model(results, task_type)

        return {
            "experiment_name": self.experiment_name,
            "best_model": best_model,
            "all_results": results,
            "summary": self._create_summary(results)
        }

    def _auto_train_cv(self, task_type: str, data_path: str, models_to_try: List[str], max_trials: int) -> Dict[str, Any]:
        """Auto-train computer vision models"""
        results = {}

        # Load data
        if data_path:
            train_loader, val_loader, test_loader = data_ingestion.load_custom_image_dataset(data_path, task_type)
        else:
            train_loader, val_loader, test_loader = data_ingestion.load_cifar10()

        # Get number of classes
        num_classes = len(train_loader.dataset.class_names) if hasattr(train_loader.dataset, 'class_names') else 10

        # Get available models
        available_models = CVModelFactory.get_available_models()[task_type]
        models_to_try = models_to_try or available_models[:max_trials]

        # Train each model
        for model_name in models_to_try:
            try:
                logger.info(f"Training CV model: {model_name}")

                model = CVModelFactory.create_model(task_type, num_classes, backbone=model_name)
                trainer = CVTrainer(f"{self.experiment_name}_{model_name}")

                result = trainer.train(model, train_loader, val_loader, test_loader, max_epochs=10)
                results[model_name] = result

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def _auto_train_tabular(self, task_type: str, data_path: str, target_column: str,
                           models_to_try: List[str], max_trials: int) -> Dict[str, Any]:
        """Auto-train tabular models"""
        results = {}

        # Load data
        if data_path:
            X_train, X_val, X_test, y_train, y_val, y_test, target_names = data_ingestion.load_custom_tabular_dataset(
                data_path, target_column, task_type
            )
        else:
            X_train, X_val, X_test, y_train, y_val, y_test, target_names = data_ingestion.load_iris_dataset()

        # Get available models
        available_models = TabularModelFactory.get_available_models()[task_type]
        models_to_try = models_to_try or available_models[:max_trials]

        # Train each model
        for model_name in models_to_try:
            try:
                logger.info(f"Training tabular model: {model_name}")

                trainer = TabularTrainer(f"{self.experiment_name}_{model_name}")

                result = trainer.train(
                    model_name, task_type, X_train, y_train, X_val, y_val,
                    X_test, y_test, target_names=target_names, hyperparameter_tuning=True
                )
                results[model_name] = result

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def _find_best_model(self, results: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Find the best performing model"""
        best_model = None
        best_score = -float('inf') if task_type in ["classification", "regression"] else float('inf')

        # Define metric to use for comparison
        metric_map = {
            "classification": "val_accuracy",
            "regression": "val_r2",
            "clustering": "silhouette_score"
        }

        metric = metric_map.get(task_type, "val_accuracy")

        for model_name, result in results.items():
            if "error" in result:
                continue

            model_metrics = result.get("metrics", {})
            score = model_metrics.get(metric, -float('inf'))

            if task_type == "clustering":
                # For clustering, higher silhouette score is better
                if score > best_score:
                    best_score = score
                    best_model = {"model_name": model_name, **result}
            else:
                # For classification/regression, higher is better
                if score > best_score:
                    best_score = score
                    best_model = {"model_name": model_name, **result}

        return best_model

    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of all results"""
        summary = {
            "total_models": len(results),
            "successful_models": len([r for r in results.values() if "error" not in r]),
            "failed_models": len([r for r in results.values() if "error" in r])
        }

        # Add comparison table
        comparison_data = []
        for model_name, result in results.items():
            if "error" not in result:
                metrics = result.get("metrics", {})
                comparison_data.append({
                    "model": model_name,
                    **metrics
                })

        if comparison_data:
            summary["model_comparison"] = pd.DataFrame(comparison_data)

        return summary


# Convenience functions
def train_cv_model(task_type: str, num_classes: int, train_loader, val_loader,
                   test_loader=None, model_type: str = "resnet18",
                   experiment_name: str = None) -> Dict[str, Any]:
    """Convenience function to train a CV model"""
    model = CVModelFactory.create_model(task_type, num_classes, backbone=model_type)
    trainer = CVTrainer(experiment_name)
    return trainer.train(model, train_loader, val_loader, test_loader)


def train_tabular_model(task_type: str, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None,
                       experiment_name: str = None) -> Dict[str, Any]:
    """Convenience function to train a tabular model"""
    trainer = TabularTrainer(experiment_name)
    return trainer.train(model_type, task_type, X_train, y_train, X_val, y_val, X_test, y_test)
