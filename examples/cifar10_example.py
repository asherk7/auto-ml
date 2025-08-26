"""
Example script for CIFAR-10 image classification using the AutoML system.
This demonstrates how to train a computer vision model on the CIFAR-10 dataset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.ingestion import data_ingestion
from training.trainer import CVTrainer
from models.cv_models import CVModelFactory
from utils.helpers import setup_logging

def main():
    """Run CIFAR-10 classification example"""

    # Setup logging
    logger = setup_logging()
    logger.info("Starting CIFAR-10 classification example")

    try:
        # Load CIFAR-10 dataset
        logger.info("Loading CIFAR-10 dataset...")
        train_loader, val_loader, test_loader = data_ingestion.load_cifar10(download=True)

        # CIFAR-10 has 10 classes
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']

        logger.info(f"Dataset loaded with {num_classes} classes: {class_names}")

        # Create model
        logger.info("Creating ResNet-18 model...")
        model = CVModelFactory.create_model(
            task_type="classification",
            num_classes=num_classes,
            backbone="resnet18",
            pretrained=True
        )

        # Create trainer
        experiment_name = "cifar10_resnet18_example"
        trainer = CVTrainer(experiment_name=experiment_name)

        # Train model
        logger.info("Starting training...")
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            max_epochs=10  # Reduced for quick example
        )

        # Print results
        logger.info("Training completed!")
        logger.info(f"Experiment ID: {results['experiment_id']}")
        logger.info(f"Model ID: {results['model_id']}")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")

        # Print metrics
        metrics = results['metrics']
        logger.info("Final metrics:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric_name}: {value:.4f}")
            else:
                logger.info(f"  {metric_name}: {value}")

        # Model file paths
        logger.info(f"Model saved to: {results['model_path']}")
        if results['onnx_path']:
            logger.info(f"ONNX model saved to: {results['onnx_path']}")

        logger.info("Example completed successfully!")

        return results

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
