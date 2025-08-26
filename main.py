"""
Main entry point for the AutoML system.
Provides command-line interface for training models, running experiments, and serving APIs.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import config, TASK_TYPES
from data.ingestion import data_ingestion
from training.trainer import AutoMLTrainer, CVTrainer, TabularTrainer
from serving.api import app
from utils.helpers import setup_logging, get_system_info
from utils.database import db

logger = setup_logging()


def train_cv_command(args):
    """Handle computer vision training command"""
    logger.info("Starting computer vision training...")

    try:
        # Load data
        if args.dataset_path:
            train_loader, val_loader, test_loader = data_ingestion.load_custom_image_dataset(
                args.dataset_path, args.task_type
            )
        else:
            # Use CIFAR-10 as default
            train_loader, val_loader, test_loader = data_ingestion.load_cifar10()

        # Get number of classes
        if hasattr(train_loader.dataset, 'classes'):
            num_classes = len(train_loader.dataset.classes)
        elif hasattr(train_loader.dataset, 'class_names'):
            num_classes = len(train_loader.dataset.class_names)
        else:
            num_classes = 10  # Default for CIFAR-10

        # Create and train model
        from models.cv_models import CVModelFactory

        model = CVModelFactory.create_model(
            task_type=args.task_type,
            num_classes=num_classes,
            backbone=args.model_type
        )

        trainer = CVTrainer(experiment_name=args.experiment_name)
        results = trainer.train(
            model, train_loader, val_loader, test_loader,
            max_epochs=args.max_epochs
        )

        logger.info("Training completed successfully!")
        logger.info(f"Experiment ID: {results['experiment_id']}")
        logger.info(f"Model ID: {results['model_id']}")
        logger.info(f"Best metrics: {results['metrics']}")

        return results

    except Exception as e:
        logger.error(f"CV training failed: {e}")
        return None


def train_tabular_command(args):
    """Handle tabular ML training command"""
    logger.info("Starting tabular ML training...")

    try:
        # Load data
        if args.dataset_path:
            X_train, X_val, X_test, y_train, y_val, y_test, target_names = data_ingestion.load_custom_tabular_dataset(
                args.dataset_path, args.target_column, args.task_type
            )
        else:
            # Use Iris as default
            X_train, X_val, X_test, y_train, y_val, y_test, target_names = data_ingestion.load_iris_dataset()

        # Train model
        trainer = TabularTrainer(experiment_name=args.experiment_name)
        results = trainer.train(
            model_type=args.model_type,
            task_type=args.task_type,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            target_names=target_names,
            hyperparameter_tuning=args.tune_hyperparameters
        )

        logger.info("Training completed successfully!")
        logger.info(f"Experiment ID: {results['experiment_id']}")
        logger.info(f"Model ID: {results['model_id']}")
        logger.info(f"Best metrics: {results['metrics']}")

        return results

    except Exception as e:
        logger.error(f"Tabular training failed: {e}")
        return None


def auto_train_command(args):
    """Handle automated training command"""
    logger.info("Starting automated training...")

    try:
        trainer = AutoMLTrainer(experiment_name=args.experiment_name)

        results = trainer.auto_train(
            data_type=args.data_type,
            task_type=args.task_type,
            data_path=args.dataset_path,
            target_column=args.target_column,
            models_to_try=args.models.split(',') if args.models else None,
            max_trials=args.max_trials
        )

        logger.info("Automated training completed!")
        logger.info(f"Best model: {results['best_model']['model_name']}")
        logger.info(f"Summary: {results['summary']}")

        return results

    except Exception as e:
        logger.error(f"Automated training failed: {e}")
        return None


def serve_command(args):
    """Handle API serving command"""
    logger.info("Starting API server...")

    try:
        import uvicorn

        uvicorn.run(
            "serving.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )

    except Exception as e:
        logger.error(f"Failed to start API server: {e}")


def list_experiments_command(args):
    """Handle list experiments command"""
    try:
        experiments_df = db.get_experiments(task_type=args.task_type)

        if experiments_df.empty:
            print("No experiments found.")
            return

        # Display experiments
        print("\nExperiments:")
        print("-" * 80)

        for _, exp in experiments_df.head(args.limit).iterrows():
            print(f"ID: {exp['id']}")
            print(f"Name: {exp['name']}")
            print(f"Task Type: {exp['task_type']}")
            print(f"Status: {exp['status']}")
            print(f"Created: {exp['created_at']}")
            if exp['metrics']:
                metrics = eval(exp['metrics'])
                print(f"Metrics: {metrics}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")


def list_models_command(args):
    """Handle list models command"""
    try:
        models_df = db.get_models(task_type=args.task_type, is_best=args.best_only)

        if models_df.empty:
            print("No models found.")
            return

        # Display models
        print("\nModels:")
        print("-" * 80)

        for _, model in models_df.head(args.limit).iterrows():
            print(f"ID: {model['id']}")
            print(f"Name: {model['name']}")
            print(f"Model Type: {model['model_type']}")
            print(f"Task Type: {model['task_type']}")
            print(f"Is Best: {model['is_best']}")
            print(f"Created: {model['created_at']}")
            if model['metrics']:
                metrics = eval(model['metrics'])
                print(f"Metrics: {metrics}")
            print("-" * 80)

    except Exception as e:
        logger.error(f"Failed to list models: {e}")


def system_info_command(args):
    """Handle system info command"""
    try:
        info = get_system_info()

        print("\nSystem Information:")
        print("-" * 50)

        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        # Database statistics
        print("\nDatabase Statistics:")
        print("-" * 50)

        summary = db.get_experiment_summary()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="AutoML System - Train and serve ML models automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a computer vision model
  python main.py train-cv --task-type classification --model-type resnet18

  # Train a tabular model with hyperparameter tuning
  python main.py train-tabular --task-type classification --model-type random_forest --tune-hyperparameters

  # Run automated training
  python main.py auto-train --data-type tabular --task-type classification --dataset-path data.csv --target-column target

  # Start API server
  python main.py serve --port 8000

  # List experiments
  python main.py list-experiments --limit 10
        """
    )

    # Global arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Computer Vision Training
    cv_parser = subparsers.add_parser('train-cv', help='Train computer vision models')
    cv_parser.add_argument('--task-type', choices=['classification', 'object_detection'],
                          default='classification', help='CV task type')
    cv_parser.add_argument('--model-type', type=str, default='resnet18',
                          help='Model architecture (resnet18, resnet34, resnet50, etc.)')
    cv_parser.add_argument('--dataset-path', type=str, help='Path to custom dataset')
    cv_parser.add_argument('--experiment-name', type=str, help='Experiment name')
    cv_parser.add_argument('--max-epochs', type=int, default=50, help='Maximum training epochs')

    # Tabular ML Training
    tabular_parser = subparsers.add_parser('train-tabular', help='Train tabular ML models')
    tabular_parser.add_argument('--task-type', choices=['classification', 'regression', 'clustering'],
                               default='classification', help='Tabular task type')
    tabular_parser.add_argument('--model-type', type=str, default='random_forest',
                               help='Model type (random_forest, logistic_regression, etc.)')
    tabular_parser.add_argument('--dataset-path', type=str, help='Path to dataset file')
    tabular_parser.add_argument('--target-column', type=str, help='Target column name')
    tabular_parser.add_argument('--experiment-name', type=str, help='Experiment name')
    tabular_parser.add_argument('--tune-hyperparameters', action='store_true',
                               help='Enable hyperparameter tuning')

    # Automated Training
    auto_parser = subparsers.add_parser('auto-train', help='Run automated training')
    auto_parser.add_argument('--data-type', choices=['image', 'tabular'], required=True,
                            help='Type of data')
    auto_parser.add_argument('--task-type', type=str, required=True,
                            help='Task type (classification, regression, etc.)')
    auto_parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    auto_parser.add_argument('--target-column', type=str, help='Target column for tabular data')
    auto_parser.add_argument('--experiment-name', type=str, help='Experiment name')
    auto_parser.add_argument('--models', type=str, help='Comma-separated list of models to try')
    auto_parser.add_argument('--max-trials', type=int, default=5, help='Maximum number of models to try')

    # API Serving
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    # List Experiments
    list_exp_parser = subparsers.add_parser('list-experiments', help='List experiments')
    list_exp_parser.add_argument('--task-type', type=str, help='Filter by task type')
    list_exp_parser.add_argument('--limit', type=int, default=10, help='Number of experiments to show')

    # List Models
    list_models_parser = subparsers.add_parser('list-models', help='List models')
    list_models_parser.add_argument('--task-type', type=str, help='Filter by task type')
    list_models_parser.add_argument('--best-only', action='store_true', help='Show only best models')
    list_models_parser.add_argument('--limit', type=int, default=10, help='Number of models to show')

    # System Info
    subparsers.add_parser('system-info', help='Show system information')

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    if args.log_level or args.log_file:
        setup_logging(args.log_level, args.log_file)

    # Handle commands
    if args.command == 'train-cv':
        train_cv_command(args)
    elif args.command == 'train-tabular':
        train_tabular_command(args)
    elif args.command == 'auto-train':
        auto_train_command(args)
    elif args.command == 'serve':
        serve_command(args)
    elif args.command == 'list-experiments':
        list_experiments_command(args)
    elif args.command == 'list-models':
        list_models_command(args)
    elif args.command == 'system-info':
        system_info_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
