# AutoML System

A comprehensive, modular AutoML system for computer vision and tabular machine learning tasks. This system provides automated model training, experiment tracking, and model serving capabilities with a focus on simplicity and extensibility.

## Features

### 🎯 Task Support
- **Computer Vision**: Image classification, object detection
- **Tabular ML**: Classification, regression, clustering

### 🤖 Model Support
- **CV Models**: ResNet, EfficientNet, MobileNet (PyTorch Lightning)
- **Tabular Models**: Random Forest, Logistic Regression, SVM, KNN, Decision Trees (scikit-learn)

### 🔧 Core Capabilities
- Automated data preprocessing and augmentation
- Hyperparameter tuning with GridSearchCV
- Experiment tracking with Weights & Biases
- Model comparison and ensembling
- ONNX export for optimized inference
- REST API serving with FastAPI
- SQLite database for metadata storage
- Docker containerization

### 📊 Built-in Datasets
- CIFAR-10 (computer vision)
- Iris (tabular classification)
- Support for custom datasets

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU training)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd auto-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Initialize the system**
```bash
python main.py system-info
```

### Quick Examples

#### 1. Train a Computer Vision Model (CIFAR-10)
```bash
# Train ResNet-18 on CIFAR-10
python main.py train-cv --task-type classification --model-type resnet18 --max-epochs 10

# Or run the example script
python examples/cifar10_example.py
```

#### 2. Train a Tabular Model (Iris)
```bash
# Train Random Forest on Iris dataset
python main.py train-tabular --task-type classification --model-type random_forest --tune-hyperparameters

# Or run the example script
python examples/iris_example.py
```

#### 3. Automated Training (Try Multiple Models)
```bash
# Auto-train on tabular data
python main.py auto-train --data-type tabular --task-type classification --models "random_forest,logistic_regression,svm"
```

#### 4. Start API Server
```bash
# Start the FastAPI server
python main.py serve --port 8000

# Test the API
curl http://localhost:8000/
```

## Project Structure

```
auto-ml/
├── config.py                 # System configuration
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Multi-service deployment
├── README.md                # This file
│
├── data/                    # Data handling
│   └── ingestion.py         # Data loading and preprocessing
│
├── models/                  # Model definitions
│   ├── cv_models.py         # Computer vision models
│   └── tabular_models.py    # Tabular ML models
│
├── training/                # Training logic
│   └── trainer.py           # Training orchestration
│
├── serving/                 # Model serving
│   └── api.py               # FastAPI endpoints
│
├── utils/                   # Utilities
│   ├── database.py          # SQLite operations
│   └── helpers.py           # Common utilities
│
└── examples/                # Example scripts
    ├── cifar10_example.py   # CV example
    └── iris_example.py      # Tabular example
```

## Usage Guide

### Command Line Interface

The system provides a comprehensive CLI for all operations:

```bash
python main.py <command> [options]
```

#### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `train-cv` | Train computer vision models | `python main.py train-cv --model-type resnet18` |
| `train-tabular` | Train tabular ML models | `python main.py train-tabular --model-type random_forest` |
| `auto-train` | Automated training with multiple models | `python main.py auto-train --data-type tabular --task-type classification` |
| `serve` | Start API server | `python main.py serve --port 8000` |
| `list-experiments` | Show experiments | `python main.py list-experiments --limit 10` |
| `list-models` | Show trained models | `python main.py list-models --best-only` |
| `system-info` | Display system information | `python main.py system-info` |

### Training Computer Vision Models

#### Supported Models
- `resnet18`, `resnet34`, `resnet50`
- `efficientnet_b0`
- `mobilenet_v3_small`

#### Example: Custom Image Dataset
```bash
# Organize your data as: dataset/class1/images, dataset/class2/images, etc.
python main.py train-cv \
    --task-type classification \
    --model-type resnet18 \
    --dataset-path /path/to/your/dataset \
    --experiment-name my_image_classifier \
    --max-epochs 50
```

### Training Tabular Models

#### Supported Models
- **Classification**: `random_forest`, `logistic_regression`, `svm`, `knn`, `decision_tree`
- **Regression**: `random_forest`, `linear_regression`, `ridge`, `lasso`, `decision_tree`, `svm`, `knn`
- **Clustering**: `kmeans`, `dbscan`, `agglomerative`

#### Example: Custom CSV Dataset
```bash
python main.py train-tabular \
    --task-type classification \
    --model-type random_forest \
    --dataset-path /path/to/your/data.csv \
    --target-column target_column_name \
    --experiment-name my_tabular_classifier \
    --tune-hyperparameters
```

### API Usage

Start the API server:
```bash
python main.py serve --port 8000
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/models` | GET | List available models |
| `/models/{id}` | GET | Get model details |
| `/predict/tabular` | POST | Predict on tabular data |
| `/predict/image` | POST | Predict on image data |
| `/predict/batch/tabular` | POST | Batch predictions on CSV |
| `/experiments` | GET | List experiments |
| `/stats` | GET | System statistics |

#### Example API Usage

```python
import requests
import numpy as np

# Predict on tabular data
response = requests.post("http://localhost:8000/predict/tabular", 
                        json={
                            "model_id": 1,
                            "data": [[5.1, 3.5, 1.4, 0.2]],
                            "return_probabilities": True
                        })
print(response.json())

# Upload image for prediction
with open("image.jpg", "rb") as f:
    response = requests.post("http://localhost:8000/predict/image",
                            files={"file": f},
                            data={"model_id": 2})
print(response.json())
```

## Configuration

The system is configured through `config.py`. Key settings include:

```python
# Training settings
config.training.max_epochs = 50
config.training.learning_rate = 1e-3
config.training.batch_size = 32

# Model settings
config.model.cv_backbone = "resnet18"
config.model.cv_pretrained = True

# Experiment tracking
config.experiment.wandb_project = "automl-experiments"
```

Environment variables can override settings:
```bash
export MAX_EPOCHS=100
export LEARNING_RATE=0.001
export WANDB_PROJECT=my-project
```

## Docker Deployment

### Build and Run
```bash
# Build the container
docker build -t automl-system .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data automl-system
```

### Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Experiment Tracking

The system integrates with Weights & Biases for experiment tracking:

1. **Setup W&B account** (optional)
   ```bash
   wandb login
   ```

2. **Configure project**
   ```bash
   export WANDB_PROJECT=my-automl-project
   export WANDB_ENTITY=my-username
   ```

3. **Track experiments**
   - All training runs are automatically logged
   - Metrics, hyperparameters, and model artifacts are saved
   - Compare experiments in the W&B dashboard

## Database

The system uses SQLite for metadata storage:

- **Location**: `data/automl.db`
- **Tables**: experiments, models, datasets, predictions
- **Access**: Through the database utility functions

```python
from utils.database import db

# Get experiment summary
summary = db.get_experiment_summary()

# List best models
best_models = db.get_models(is_best=True)
```

## Adding Custom Models

### Computer Vision Models
```python
# In models/cv_models.py
def create_custom_cv_model(num_classes):
    # Define your custom PyTorch Lightning model
    pass
```

### Tabular Models
```python
# In models/tabular_models.py
class CustomTabularModel:
    def __init__(self, **kwargs):
        # Initialize your custom scikit-learn compatible model
        pass
```

## Monitoring and Debugging

### Logging
```bash
# Enable debug logging
python main.py --log-level DEBUG train-cv --model-type resnet18

# Save logs to file
python main.py --log-file training.log train-tabular --model-type random_forest
```

### System Information
```bash
# Check system capabilities
python main.py system-info

# Monitor API health
curl http://localhost:8000/stats
```

## Performance Tips

1. **GPU Training**: Set `accelerator="gpu"` in config for faster CV training
2. **Batch Size**: Increase batch size for better GPU utilization
3. **Workers**: Adjust `num_workers` based on CPU cores
4. **Caching**: Enable data caching for faster loading
5. **ONNX**: Export models to ONNX for faster inference

## Common Issues and Solutions

### Memory Issues
```bash
# Reduce batch size
export BATCH_SIZE=16

# Use smaller models
python main.py train-cv --model-type mobilenet_v3_small
```

### Slow Training
```bash
# Use pre-trained models
# Reduce dataset size for testing
# Enable mixed precision training
```

### API Errors
```bash
# Check model file exists
python main.py list-models

# Clear model cache
curl -X DELETE http://localhost:8000/cache
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Example Workflows

### 1. Quick Prototype
```bash
# Test on built-in datasets
python examples/iris_example.py
python examples/cifar10_example.py
```

### 2. Custom Dataset Training
```bash
# Prepare your data
# Train models
python main.py auto-train --data-type tabular --task-type classification --dataset-path mydata.csv --target-column target

# Start API for serving
python main.py serve
```

### 3. Model Comparison
```bash
# Train multiple models
python main.py auto-train --data-type tabular --task-type classification --models "random_forest,svm,logistic_regression"

# Compare results
python main.py list-models --task-type tabular_classification
```

### 4. Production Deployment
```bash
# Build container
docker build -t my-automl .

# Deploy with docker-compose
docker-compose up -d

# Monitor
docker-compose logs -f
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the documentation above
2. Review example scripts in `/examples`
3. Check system info: `python main.py system-info`
4. Enable debug logging: `--log-level DEBUG`

---

**Note**: This is an educational AutoML system focused on simplicity and modularity. For production use cases, consider additional features like advanced security, scalability, and robustness measures.