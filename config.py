"""
Configuration management for AutoML system.
Contains all configurable parameters for data processing, training, and serving.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # General
    data_dir: str = "data"
    cache_dir: str = "data/cache"

    # Image processing
    image_size: tuple = (224, 224)
    batch_size: int = 32
    num_workers: int = 4

    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class ModelConfig:
    """Configuration for model architectures"""
    # Computer Vision
    cv_backbone: str = "resnet18"  # Options: resnet18, resnet34, resnet50
    cv_pretrained: bool = True
    cv_num_classes: int = 10

    # Tabular ML
    tabular_models: List[str] = None

    def __post_init__(self):
        if self.tabular_models is None:
            self.tabular_models = ["linear", "random_forest", "decision_tree", "kmeans"]


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # General
    max_epochs: int = 50
    early_stopping_patience: int = 7
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # PyTorch Lightning
    accelerator: str = "cpu"  # Options: cpu, gpu, mps
    devices: int = 1
    precision: str = "32"  # Options: 16, 32, bf16

    # Checkpointing
    save_top_k: int = 3
    monitor: str = "val_accuracy"
    mode: str = "max"


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    # Weights & Biases
    wandb_project: str = "automl-experiments"
    wandb_entity: Optional[str] = None
    log_model: bool = True

    # Local tracking
    experiment_dir: str = "experiments"
    save_predictions: bool = True
    save_metrics: bool = True


@dataclass
class ServingConfig:
    """Configuration for model serving"""
    # FastAPI
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Model serving
    model_dir: str = "models/saved"
    onnx_dir: str = "models/onnx"
    max_batch_size: int = 32

    # API settings
    api_title: str = "AutoML API"
    api_description: str = "AutoML system for computer vision and tabular ML tasks"
    api_version: str = "1.0.0"


@dataclass
class DatabaseConfig:
    """Configuration for data storage"""
    db_path: str = "data/automl.db"
    echo: bool = False  # SQLAlchemy echo for debugging


class Config:
    """Main configuration class that combines all config sections"""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.experiment = ExperimentConfig()
        self.serving = ServingConfig()
        self.database = DatabaseConfig()

        # Environment-based overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Training overrides
        if os.getenv("MAX_EPOCHS"):
            self.training.max_epochs = int(os.getenv("MAX_EPOCHS"))

        if os.getenv("LEARNING_RATE"):
            self.training.learning_rate = float(os.getenv("LEARNING_RATE"))

        if os.getenv("BATCH_SIZE"):
            self.data.batch_size = int(os.getenv("BATCH_SIZE"))

        # Serving overrides
        if os.getenv("API_PORT"):
            self.serving.port = int(os.getenv("API_PORT"))

        if os.getenv("API_HOST"):
            self.serving.host = os.getenv("API_HOST")

        # Experiment tracking
        if os.getenv("WANDB_PROJECT"):
            self.experiment.wandb_project = os.getenv("WANDB_PROJECT")

        if os.getenv("WANDB_ENTITY"):
            self.experiment.wandb_entity = os.getenv("WANDB_ENTITY")

    def get_task_config(self, task_type: str) -> Dict:
        """Get task-specific configuration"""
        task_configs = {
            "classification": {
                "loss_function": "cross_entropy",
                "metrics": ["accuracy", "f1_score", "precision", "recall"],
                "optimizer": "adam"
            },
            "object_detection": {
                "loss_function": "yolo_loss",
                "metrics": ["map", "precision", "recall"],
                "optimizer": "sgd"
            },
            "regression": {
                "loss_function": "mse",
                "metrics": ["mse", "mae", "r2"],
                "optimizer": "adam"
            },
            "clustering": {
                "loss_function": "inertia",
                "metrics": ["silhouette_score", "calinski_harabasz"],
                "optimizer": "kmeans"
            }
        }
        return task_configs.get(task_type, {})


# Global config instance
config = Config()

# Task type mapping
TASK_TYPES = {
    "cv_classification": "Computer Vision Classification",
    "cv_object_detection": "Computer Vision Object Detection",
    "tabular_regression": "Tabular Regression",
    "tabular_classification": "Tabular Classification",
    "tabular_clustering": "Tabular Clustering"
}

# Supported file formats
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_TABULAR_FORMATS = [".csv", ".xlsx", ".json", ".parquet"]
