# AutoML System

A lightweight, modular AutoML framework for computer vision and tabular machine learning. Supports training, experiment tracking, and model serving with minimal setup.

---

## Features

* **Tasks**:

  * Computer Vision → image classification, object detection
  * Tabular → classification, regression, clustering

* **Models**:

  * CV → ResNet, EfficientNet, MobileNet (PyTorch Lightning)
  * Tabular → Random Forest, Logistic Regression, SVM, KNN, Decision Trees (scikit-learn)

* **Core**:

  * Auto preprocessing + augmentation
  * Hyperparameter tuning (GridSearchCV)
  * Experiment tracking (Weights & Biases)
  * ONNX export for inference
  * FastAPI serving + SQLite metadata DB
  * Docker ready

* **Datasets**: CIFAR-10, Iris, + custom datasets

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/asherk7/auto-ml.git && cd auto-ml
pip install -r requirements.txt

# Check system info
python main.py system-info
```

### Examples

```bash
# Train CV model (ResNet-18 on CIFAR-10)
python main.py train-cv --model-type resnet18 --max-epochs 10

# Train tabular model (Random Forest on Iris)
python main.py train-tabular --model-type random_forest --tune-hyperparameters

# Try multiple models automatically
python main.py auto-train --data-type tabular --task-type classification --models "random_forest,svm,logistic_regression"

# Start API server
python main.py serve --port 8000
```

---

## Project Structure

```
auto-ml/
├── main.py            # CLI entry
├── models/            # CV + tabular models
├── training/          # Training logic
├── serving/           # FastAPI endpoints
├── utils/             # DB + helpers
└── examples/          # Example scripts
```

---

## Key Commands

| Command            | Use                  |
| ------------------ | -------------------- |
| `train-cv`         | Train vision model   |
| `train-tabular`    | Train tabular model  |
| `auto-train`       | Run multiple models  |
| `serve`            | Start FastAPI server |
| `list-experiments` | Show experiments     |
| `list-models`      | Show trained models  |

---

## Advanced

* W\&B for experiment tracking (`wandb login`)
* Export to ONNX for fast inference
* Docker & docker-compose for deployment

---

## Roadmap

* Optuna for smarter hyperparameter tuning
* Fine-tuning large pretrained models (QLoRA, PEFT)
* Explainability → GradCAM, SHAP/LIME
* Redis + Celery for async serving
* Interactive model builder
* Real-time deployment for latency and inference testing
