import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

class TabularClassifier:
    """Wrapper class for tabular classification models"""

    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None

        logger.info(f"Initialized TabularClassifier with {model_type}")

    def _create_model(self, model_type: str, **kwargs):
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                random_state=42
            ),
            "logistic_regression": LogisticRegression(
                C=kwargs.get("C", 1.0),
                max_iter=kwargs.get("max_iter", 1000),
                random_state=42
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                random_state=42
            ),
            "svm": SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf"),
                probability=True,
                random_state=42
            ),
            "knn": KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 5),
                weights=kwargs.get("weights", "uniform")
            )
        }

        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")

        return models[model_type]

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None,
            target_names: List[str] = None) -> Dict[str, float]:
        """Fit the model and return training metrics"""
        logger.info(f"Training {self.model_type} classifier...")

        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        self.target_names = target_names or [str(i) for i in range(len(np.unique(y)))]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled) if hasattr(self.model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted")
        }

        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if not hasattr(self.model, "predict_proba"):
            return None

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted")
        }

        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        X_scaled = self.scaler.fit_transform(X)

        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")

        return {
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
            "cv_scores": scores.tolist()
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores"""
        if not self.is_fitted:
            return None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        elif hasattr(self.model, "coef_"):
            # For linear models, use absolute coefficients
            importance = np.abs(self.model.coef_[0])
            return dict(zip(self.feature_names, importance))

        return None

class TabularRegressor:
    """Wrapper class for tabular regression models"""

    def __init__(self, model_type: str = "random_forest", **kwargs):
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None

        logger.info(f"Initialized TabularRegressor with {model_type}")

    def _create_model(self, model_type: str, **kwargs):
        """Create the underlying sklearn model"""
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", None),
                random_state=42
            ),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(
                alpha=kwargs.get("alpha", 1.0),
                random_state=42
            ),
            "lasso": Lasso(
                alpha=kwargs.get("alpha", 1.0),
                random_state=42
            ),
            "decision_tree": DecisionTreeRegressor(
                max_depth=kwargs.get("max_depth", None),
                min_samples_split=kwargs.get("min_samples_split", 2),
                random_state=42
            ),
            "svm": SVR(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf")
            ),
            "knn": KNeighborsRegressor(
                n_neighbors=kwargs.get("n_neighbors", 5),
                weights=kwargs.get("weights", "uniform")
            )
        }

        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")

        return models[model_type]

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict[str, float]:
        logger.info(f"Training {self.model_type} regressor...")

        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.is_fitted = True

        y_pred = self.model.predict(X_scaled)

        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred))
        }

        logger.info(f"Training complete. R²: {metrics['r2']:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)

        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred))
        }

        return metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        X_scaled = self.scaler.fit_transform(X)

        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="r2")

        return {
            "cv_r2_mean": scores.mean(),
            "cv_r2_std": scores.std(),
            "cv_scores": scores.tolist()
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores"""
        if not self.is_fitted:
            return None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_)
            return dict(zip(self.feature_names, importance))

        return None


class TabularClusterer:
    def __init__(self, model_type: str = "kmeans", **kwargs):
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.labels_ = None

        logger.info(f"Initialized TabularClusterer with {model_type}")

    def _create_model(self, model_type: str, **kwargs):
        models = {
            "kmeans": KMeans(
                n_clusters=kwargs.get("n_clusters", 3),
                random_state=42,
                n_init=10
            ),
            "dbscan": DBSCAN(
                eps=kwargs.get("eps", 0.5),
                min_samples=kwargs.get("min_samples", 5)
            ),
            "agglomerative": AgglomerativeClustering(
                n_clusters=kwargs.get("n_clusters", 3),
                linkage=kwargs.get("linkage", "ward")
            )
        }

        if model_type not in models:
            raise ValueError(f"Unsupported model type: {model_type}")

        return models[model_type]

    def fit(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, float]:
        logger.info(f"Training {self.model_type} clusterer...")

        # Store metadata
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled)
        self.labels_ = self.model.labels_
        self.is_fitted = True

        # Calculate clustering metrics
        metrics = {}

        if len(set(self.labels_)) > 1:  # Need at least 2 clusters for metrics
            metrics["silhouette_score"] = silhouette_score(X_scaled, self.labels_)
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_scaled, self.labels_)

        metrics["n_clusters"] = len(set(self.labels_))
        metrics["n_noise"] = list(self.labels_).count(-1) if -1 in self.labels_ else 0

        if hasattr(self.model, "inertia_"):
            metrics["inertia"] = self.model.inertia_

        logger.info(f"Clustering complete. Found {metrics['n_clusters']} clusters")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict"):
            return self.model.predict(X_scaled)
        else:
            # For models without predict method (like DBSCAN), return labels from fit
            logger.warning(f"{self.model_type} doesn't support prediction on new data")
            return self.labels_

    def fit_predict(self, X: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        metrics = self.fit(X, feature_names)
        return self.labels_, metrics

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers (if available)"""
        if hasattr(self.model, "cluster_centers_"):
            # Transform back to original scale
            centers_scaled = self.model.cluster_centers_
            centers = self.scaler.inverse_transform(centers_scaled)
            return centers
        return None


class TabularModelFactory:
    """Base class for creating tabular ML models"""

    @staticmethod
    def create_model(task_type: str, model_type: str = None, **kwargs):
        """Create a tabular model based on task type"""

        if task_type == "classification":
            model_type = model_type or "random_forest"
            return TabularClassifier(model_type=model_type, **kwargs)

        elif task_type == "regression":
            model_type = model_type or "random_forest"
            return TabularRegressor(model_type=model_type, **kwargs)

        elif task_type == "clustering":
            model_type = model_type or "kmeans"
            return TabularClusterer(model_type=model_type, **kwargs)

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        return {
            "classification": [
                "random_forest", "logistic_regression", "decision_tree",
                "svm", "knn"
            ],
            "regression": [
                "random_forest", "linear_regression", "ridge", "lasso",
                "decision_tree", "svm", "knn"
            ],
            "clustering": [
                "kmeans", "dbscan", "agglomerative"
            ]
        }

    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        model_info = {
            # Classification models
            "random_forest": {
                "type": "ensemble",
                "description": "Random Forest classifier, robust and interpretable",
                "pros": ["Handles overfitting well", "Feature importance", "Robust to outliers"],
                "cons": ["Can be slow on large datasets", "Memory intensive"]
            },
            "logistic_regression": {
                "type": "linear",
                "description": "Logistic regression, fast and interpretable",
                "pros": ["Fast", "Probabilistic output", "No hyperparameter tuning needed"],
                "cons": ["Assumes linear relationship", "Sensitive to outliers"]
            },
            "decision_tree": {
                "type": "tree",
                "description": "Decision tree classifier, highly interpretable",
                "pros": ["Easy to interpret", "Handles mixed data types", "No feature scaling needed"],
                "cons": ["Prone to overfitting", "Unstable"]
            },
            "svm": {
                "type": "kernel",
                "description": "Support Vector Machine, effective for high-dimensional data",
                "pros": ["Effective in high dimensions", "Memory efficient", "Versatile kernels"],
                "cons": ["Slow on large datasets", "Sensitive to feature scaling"]
            },
            "knn": {
                "type": "instance",
                "description": "K-Nearest Neighbors, simple and effective",
                "pros": ["Simple", "No assumptions about data", "Good for small datasets"],
                "cons": ["Computationally expensive", "Sensitive to irrelevant features"]
            },

            # Regression models
            "linear_regression": {
                "type": "linear",
                "description": "Linear regression, simple and fast",
                "pros": ["Fast", "Interpretable", "No hyperparameters"],
                "cons": ["Assumes linear relationship", "Sensitive to outliers"]
            },
            "ridge": {
                "type": "linear",
                "description": "Ridge regression with L2 regularization",
                "pros": ["Handles multicollinearity", "Prevents overfitting"],
                "cons": ["Biased estimates", "Feature selection not automatic"]
            },
            "lasso": {
                "type": "linear",
                "description": "Lasso regression with L1 regularization",
                "pros": ["Feature selection", "Prevents overfitting"],
                "cons": ["Can select only one from correlated features"]
            },

            # Clustering models
            "kmeans": {
                "type": "centroid",
                "description": "K-means clustering, fast and simple",
                "pros": ["Fast", "Simple", "Scales well"],
                "cons": ["Need to specify k", "Assumes spherical clusters"]
            },
            "dbscan": {
                "type": "density",
                "description": "DBSCAN clustering, finds arbitrary shaped clusters",
                "pros": ["Finds arbitrary shapes", "Identifies outliers", "No need to specify k"],
                "cons": ["Sensitive to hyperparameters", "Struggles with varying densities"]
            },
            "agglomerative": {
                "type": "hierarchical",
                "description": "Agglomerative clustering, hierarchical approach",
                "pros": ["No need to specify k initially", "Deterministic", "Creates hierarchy"],
                "cons": ["Computationally expensive", "Sensitive to noise"]
            }
        }

        return model_info.get(model_name, {"description": "Unknown model"})

    @staticmethod
    def auto_select_model(X: np.ndarray, y: np.ndarray = None, task_type: str = None) -> str:
        """Automatically select the best model based on data characteristics"""
        n_samples, n_features = X.shape

        if task_type == "classification":
            if n_samples < 1000:
                return "svm" if n_features > 50 else "random_forest"
            elif n_samples < 10000:
                return "random_forest"
            else:
                return "logistic_regression"

        elif task_type == "regression":
            if n_samples < 1000:
                return "random_forest"
            elif n_features > n_samples:
                return "ridge"
            else:
                return "linear_regression" if n_features < 20 else "random_forest"

        elif task_type == "clustering":
            if n_samples < 1000:
                return "kmeans"
            else:
                return "dbscan" if n_features > 10 else "kmeans"

        return "random_forest"  # Default fallback


# Convenience functions for quick model creation
def create_classifier(model_type: str = "random_forest", **kwargs) -> TabularClassifier:
    """Create a tabular classifier"""
    return TabularClassifier(model_type=model_type, **kwargs)


def create_regressor(model_type: str = "random_forest", **kwargs) -> TabularRegressor:
    """Create a tabular regressor"""
    return TabularRegressor(model_type=model_type, **kwargs)


def create_clusterer(model_type: str = "kmeans", **kwargs) -> TabularClusterer:
    """Create a tabular clusterer"""
    return TabularClusterer(model_type=model_type, **kwargs)


def save_model(model: Union[TabularClassifier, TabularRegressor, TabularClusterer],
               filepath: str) -> None:
    """Save a trained model to disk"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Union[TabularClassifier, TabularRegressor, TabularClusterer]:
    """Load a trained model from disk"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file {filepath} not found")

    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model
