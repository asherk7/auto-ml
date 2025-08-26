"""
Utility functions for common operations across the AutoML system.
Includes logging, visualization, model evaluation, and file handling utilities.
"""

import os
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from PIL import Image

from config import config


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration"""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file:
        log_file_path = log_dir / log_file
        handlers.append(logging.FileHandler(log_file_path))
    else:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = log_dir / f"automl_{timestamp}.log"
        handlers.append(logging.FileHandler(default_log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(__name__)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file as dictionary"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"JSON file {filepath} not found")

    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object as pickle file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load pickle file"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Pickle file {filepath} not found")

    with open(filepath, 'rb') as f:
        return pickle.load(f)


def calculate_model_size(model_path: str) -> Dict[str, Union[int, str]]:
    """Calculate model file size"""
    if not Path(model_path).exists():
        return {"error": "File not found"}

    size_bytes = Path(model_path).stat().st_size

    # Convert to human readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            size_human = f"{size_bytes:.2f} {unit}"
            break
        size_bytes /= 1024.0
    else:
        size_human = f"{size_bytes:.2f} TB"

    return {
        "size_bytes": Path(model_path).stat().st_size,
        "size_human": size_human
    }


def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: List[str] = None,
                                save_path: str = None) -> str:
    """Create and save confusion matrix plot"""

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names or range(len(cm)),
        yticklabels=class_names or range(len(cm))
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"plots/confusion_matrix_{timestamp}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def create_feature_importance_plot(feature_importance: Dict[str, float],
                                  top_n: int = 20,
                                  save_path: str = None) -> str:
    """Create feature importance plot"""

    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Take top N features
    top_features = sorted_features[:top_n]
    features, importance = zip(*top_features)

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importance)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(f'Top {len(features)} Feature Importance')
    plt.gca().invert_yaxis()

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"plots/feature_importance_{timestamp}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def create_learning_curve_plot(train_losses: List[float],
                              val_losses: List[float],
                              train_metrics: List[float] = None,
                              val_metrics: List[float] = None,
                              metric_name: str = "Accuracy",
                              save_path: str = None) -> str:
    """Create learning curve plot"""

    epochs = range(1, len(train_losses) + 1)

    if train_metrics and val_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Metric plot
        ax2.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        ax2.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        ax2.set_title(f'Model {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.legend()
    else:
        # Only loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"plots/learning_curve_{timestamp}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def visualize_data_distribution(data: np.ndarray,
                               labels: np.ndarray = None,
                               method: str = "pca",
                               save_path: str = None) -> str:
    """Visualize high-dimensional data in 2D"""

    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    # Reduce dimensionality
    data_2d = reducer.fit_transform(data)

    # Create plot
    plt.figure(figsize=(10, 8))

    if labels is not None:
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 10:  # Only show legend for reasonable number of classes
            for label in unique_labels:
                mask = labels == label
                plt.scatter(data_2d[mask, 0], data_2d[mask, 1],
                           label=f'Class {label}', alpha=0.7)
            plt.legend()
    else:
        plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)

    plt.title(f'Data Visualization using {method.upper()}')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"plots/data_visualization_{method}_{timestamp}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def create_model_comparison_plot(results: Dict[str, Dict[str, float]],
                                metric: str = "accuracy",
                                save_path: str = None) -> str:
    """Create model comparison bar plot"""

    models = list(results.keys())
    scores = [results[model].get(metric, 0) for model in models]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, scores)
    plt.title(f'Model Comparison - {metric.title()}')
    plt.xlabel('Models')
    plt.ylabel(metric.title())
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')

    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"plots/model_comparison_{timestamp}.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: List[str] = None) -> Dict[str, Any]:
    """Generate detailed classification report"""

    # Basic classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names or range(len(cm))):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        per_class_metrics[str(class_name)] = {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        }

    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics
    }


def validate_data_format(data: Union[np.ndarray, pd.DataFrame],
                        data_type: str) -> Tuple[bool, str]:
    """Validate data format for training/inference"""

    if data_type == "tabular":
        if isinstance(data, pd.DataFrame):
            if data.isnull().any().any():
                return False, "Data contains missing values"
            if data.empty:
                return False, "Data is empty"
        elif isinstance(data, np.ndarray):
            if np.isnan(data).any():
                return False, "Data contains NaN values"
            if data.size == 0:
                return False, "Data is empty"
        else:
            return False, "Data must be pandas DataFrame or numpy array"

    elif data_type == "image":
        if isinstance(data, np.ndarray):
            if data.ndim not in [3, 4]:  # (H, W, C) or (N, H, W, C)
                return False, "Image data must be 3D or 4D array"
            if data.dtype not in [np.uint8, np.float32, np.float64]:
                return False, "Image data must be uint8 or float"
        else:
            return False, "Image data must be numpy array"

    return True, "Data format is valid"


def estimate_training_time(n_samples: int, n_features: int,
                          model_type: str, task_type: str) -> Dict[str, str]:
    """Estimate training time based on data size and model type"""

    # Very rough estimates (in seconds)
    base_times = {
        "linear": 0.001,
        "tree": 0.01,
        "ensemble": 0.1,
        "svm": 0.1,
        "neural_network": 1.0
    }

    model_categories = {
        "linear_regression": "linear",
        "logistic_regression": "linear",
        "ridge": "linear",
        "lasso": "linear",
        "decision_tree": "tree",
        "random_forest": "ensemble",
        "svm": "svm",
        "resnet": "neural_network",
        "efficientnet": "neural_network"
    }

    category = model_categories.get(model_type, "ensemble")
    base_time = base_times[category]

    # Scale by data size
    estimated_seconds = base_time * n_samples * np.log(n_features + 1)

    # Convert to human readable format
    if estimated_seconds < 60:
        time_str = f"{estimated_seconds:.1f} seconds"
    elif estimated_seconds < 3600:
        time_str = f"{estimated_seconds/60:.1f} minutes"
    else:
        time_str = f"{estimated_seconds/3600:.1f} hours"

    return {
        "estimated_seconds": estimated_seconds,
        "estimated_time": time_str,
        "note": "This is a rough estimate and actual time may vary significantly"
    }


def clean_experiment_name(name: str) -> str:
    """Clean experiment name for file system compatibility"""
    import re

    # Remove invalid characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', name)

    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)

    # Strip whitespace and underscores
    cleaned = cleaned.strip(' _')

    # Ensure not empty
    if not cleaned:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned = f"experiment_{timestamp}"

    return cleaned


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    import platform
    import psutil

    try:
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_usage_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "torch_version": torch.__version__ if torch else "Not installed",
        }

        # Check GPU availability
        if torch and torch.cuda.is_available():
            system_info["gpu_available"] = True
            system_info["gpu_count"] = torch.cuda.device_count()
            system_info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            system_info["gpu_available"] = False

        return system_info

    except Exception as e:
        return {"error": f"Failed to get system info: {str(e)}"}


def create_experiment_report(experiment_results: Dict[str, Any],
                           save_path: str = None) -> str:
    """Create a comprehensive experiment report"""

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"reports/experiment_report_{timestamp}.html"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AutoML Experiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px;
                      background-color: #e7f3ff; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AutoML Experiment Report</h1>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Experiment: {experiment_results.get('experiment_name', 'Unknown')}</p>
        </div>

        <div class="section">
            <h2>Best Model</h2>
            <div class="metric">
                <strong>Model:</strong> {experiment_results.get('best_model', {}).get('model_name', 'N/A')}
            </div>
        </div>

        <div class="section">
            <h2>System Information</h2>
            {_format_system_info_html(get_system_info())}
        </div>

        <div class="section">
            <h2>Summary</h2>
            {_format_summary_html(experiment_results.get('summary', {}))}
        </div>

    </body>
    </html>
    """

    with open(save_path, 'w') as f:
        f.write(html_content)

    return save_path


def _format_system_info_html(system_info: Dict[str, Any]) -> str:
    """Format system info as HTML"""
    html = "<table>"
    for key, value in system_info.items():
        html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
    html += "</table>"
    return html


def _format_summary_html(summary: Dict[str, Any]) -> str:
    """Format summary as HTML"""
    html = "<table>"
    for key, value in summary.items():
        if key != "model_comparison":  # Skip DataFrame
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
    html += "</table>"
    return html


# Initialize logging
logger = setup_logging()
