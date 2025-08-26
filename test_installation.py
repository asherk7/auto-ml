#!/usr/bin/env python3
"""
Test installation script for the AutoML system.
This script verifies that all dependencies are properly installed
and the system is ready to use.
"""

import sys
import importlib
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("Testing Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
    return True

def test_required_packages():
    """Test that required packages can be imported"""
    required_packages = [
        'torch',
        'torchvision',
        'pytorch_lightning',
        'sklearn',
        'numpy',
        'pandas',
        'fastapi',
        'uvicorn',
        'PIL',
        'cv2',
        'wandb',
        'onnx',
        'onnxruntime',
        'matplotlib',
        'seaborn',
        'albumentations',
        'tqdm',
        'yaml'
    ]

    print("\nTesting required packages...")
    failed_imports = []

    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n❌ Missing packages: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False

    print("✅ All required packages installed")
    return True

def test_torch_functionality():
    """Test PyTorch functionality"""
    print("\nTesting PyTorch functionality...")
    try:
        import torch

        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)

        print(f"✅ PyTorch tensor operations - OK")

        # Check for CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s) detected")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available - CPU only mode")

        return True
    except Exception as e:
        print(f"❌ PyTorch functionality test failed: {e}")
        return False

def test_project_structure():
    """Test that project structure is correct"""
    print("\nTesting project structure...")

    required_dirs = [
        'data', 'models', 'training', 'serving', 'utils', 'examples'
    ]

    required_files = [
        'config.py', 'main.py', 'requirements.txt',
        'data/ingestion.py',
        'models/cv_models.py', 'models/tabular_models.py',
        'training/trainer.py',
        'serving/api.py',
        'utils/database.py', 'utils/helpers.py'
    ]

    missing_items = []

    # Check directories
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_items.append(f"Directory: {directory}")
        else:
            print(f"✅ {directory}/ - OK")

    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_items.append(f"File: {file_path}")
        else:
            print(f"✅ {file_path} - OK")

    if missing_items:
        print(f"\n❌ Missing items:")
        for item in missing_items:
            print(f"   - {item}")
        return False

    print("✅ Project structure complete")
    return True

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from config import config

        # Test basic config access
        assert hasattr(config, 'data')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'serving')

        print(f"✅ Configuration loaded successfully")
        print(f"   Batch size: {config.data.batch_size}")
        print(f"   Max epochs: {config.training.max_epochs}")
        print(f"   API port: {config.serving.port}")

        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\nTesting database...")
    try:
        from utils.database import db

        # Test database initialization
        summary = db.get_experiment_summary()
        print(f"✅ Database connection - OK")
        print(f"   Total experiments: {summary.get('total_experiments', 0)}")
        print(f"   Total models: {summary.get('total_models', 0)}")

        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_data_ingestion():
    """Test data ingestion functionality"""
    print("\nTesting data ingestion...")
    try:
        from data.ingestion import data_ingestion

        # Test Iris dataset loading (small and fast)
        X_train, X_val, X_test, y_train, y_val, y_test, target_names = data_ingestion.load_iris_dataset()

        print(f"✅ Data ingestion - OK")
        print(f"   Iris dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
        print(f"   Features: {X_train.shape[1]}, Classes: {len(target_names)}")

        return True
    except Exception as e:
        print(f"❌ Data ingestion test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    try:
        # Test tabular model
        from models.tabular_models import TabularModelFactory
        tabular_model = TabularModelFactory.create_model("classification", "random_forest")
        print(f"✅ Tabular model creation - OK")

        # Test CV model (simpler, no GPU required)
        from models.cv_models import CVModelFactory
        cv_model = CVModelFactory.create_model("classification", num_classes=10, backbone="resnet18")
        print(f"✅ CV model creation - OK")

        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def run_quick_training_test():
    """Run a very quick training test"""
    print("\nRunning quick training test (Iris dataset)...")
    try:
        from data.ingestion import data_ingestion
        from training.trainer import TabularTrainer

        # Load small dataset
        X_train, X_val, X_test, y_train, y_val, y_test, target_names = data_ingestion.load_iris_dataset()

        # Quick training (no hyperparameter tuning)
        trainer = TabularTrainer("installation_test")
        result = trainer.train(
            model_type="logistic_regression",
            task_type="classification",
            X_train=X_train[:50],  # Use only subset for speed
            y_train=y_train[:50],
            X_val=X_val[:20],
            y_val=y_val[:20],
            target_names=target_names,
            hyperparameter_tuning=False
        )

        accuracy = result['metrics'].get('val_accuracy', 0)
        print(f"✅ Quick training test - OK")
        print(f"   Model: Logistic Regression")
        print(f"   Validation accuracy: {accuracy:.3f}")

        return True
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False

def main():
    """Run all installation tests"""
    print("🔍 AutoML System Installation Test")
    print("=" * 50)

    tests = [
        ("Python Version", test_python_version),
        ("Required Packages", test_required_packages),
        ("PyTorch Functionality", test_torch_functionality),
        ("Project Structure", test_project_structure),
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Data Ingestion", test_data_ingestion),
        ("Model Creation", test_model_creation),
        ("Quick Training", run_quick_training_test)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")

    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your AutoML system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Run an example: python examples/iris_example.py")
        print("   2. Start the API: python main.py serve")
        print("   3. Train a model: python main.py train-tabular --model-type random_forest")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("\n🔧 Common solutions:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Check Python version: python --version")
        print("   3. Verify project structure is complete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
