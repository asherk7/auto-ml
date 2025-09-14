import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from data.ingestion import data_ingestion
from training.trainer import TabularTrainer
from models.tabular_models import TabularModelFactory
from utils.helpers import setup_logging, create_classification_report

def main():
    """Run Iris classification example"""

    logger = setup_logging()
    logger.info("Starting Iris classification example")

    try:
        logger.info("Loading Iris dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test, target_names = data_ingestion.load_iris_dataset()

        logger.info(f"Dataset loaded:")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Classes: {target_names}")

        # Try different models
        models_to_try = ["random_forest", "logistic_regression", "svm", "knn"]
        results = {}

        for model_type in models_to_try:
            logger.info(f"\nTraining {model_type} model...")

            # Create trainer
            experiment_name = f"iris_{model_type}_example"
            trainer = TabularTrainer(experiment_name=experiment_name)

            # Train model
            result = trainer.train(
                model_type=model_type,
                task_type="classification",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                target_names=target_names,
                hyperparameter_tuning=True
            )

            results[model_type] = result

            metrics = result['metrics']
            logger.info(f"Results for {model_type}:")
            logger.info(f"  Train accuracy: {metrics.get('train_accuracy', 'N/A'):.4f}")
            logger.info(f"  Validation accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")
            logger.info(f"  Test accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
            logger.info(f"  CV accuracy: {metrics.get('cv_accuracy_mean', 'N/A'):.4f} ± {metrics.get('cv_accuracy_std', 0):.4f}")

        # Find best model
        logger.info("\nModel Comparison:")
        logger.info("-" * 60)

        best_model = None
        best_score = 0

        for model_name, result in results.items():
            val_acc = result['metrics'].get('val_accuracy', 0)
            test_acc = result['metrics'].get('test_accuracy', 0)
            cv_acc = result['metrics'].get('cv_accuracy_mean', 0)

            logger.info(f"{model_name:20} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_acc:.4f}")

            if val_acc > best_score:
                best_score = val_acc
                best_model = model_name

        logger.info("-" * 60)
        logger.info(f"Best model: {best_model} (validation accuracy: {best_score:.4f})")

        # Detailed evaluation of best model
        if best_model:
            logger.info(f"\nDetailed evaluation of {best_model}:")

            best_result = results[best_model]
            model = best_result['model']

            y_pred = model.predict(X_test)

            report = create_classification_report(y_test, y_pred, target_names)

            logger.info("Classification Report:")
            for class_name, metrics in report['classification_report'].items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    logger.info(f"  {class_name}: precision={metrics['precision']:.3f}, "
                              f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")

            # Feature importance
            feature_importance = model.get_feature_importance()
            if feature_importance:
                logger.info("\nFeature Importance:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features:
                    logger.info(f"  {feature}: {importance:.4f}")

        logger.info("\nExample completed successfully!")

        return results

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
