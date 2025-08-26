"""
Database utilities for SQLite operations.
Handles data storage, experiment tracking, and model metadata.
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from config import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for the AutoML system"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        self.init_database()

    def init_database(self):
        """Initialize database and create tables if they don't exist"""
        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    dataset_name TEXT,
                    model_type TEXT,
                    status TEXT DEFAULT 'running',
                    config TEXT,
                    metrics TEXT,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    file_path TEXT,
                    onnx_path TEXT,
                    metrics TEXT,
                    parameters TEXT,
                    is_best BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)

            # Create datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    data_type TEXT NOT NULL,
                    file_path TEXT,
                    num_samples INTEGER,
                    num_features INTEGER,
                    num_classes INTEGER,
                    description TEXT,
                    preprocessing_info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    input_data TEXT,
                    prediction TEXT,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models (id)
                )
            """)

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def create_experiment(self, name: str, task_type: str, dataset_name: str = None,
                         model_type: str = None, config_dict: Dict = None) -> int:
        """Create a new experiment record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            config_json = json.dumps(config_dict) if config_dict else None

            cursor.execute("""
                INSERT INTO experiments (name, task_type, dataset_name, model_type, config)
                VALUES (?, ?, ?, ?, ?)
            """, (name, task_type, dataset_name, model_type, config_json))

            experiment_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Created experiment {experiment_id}: {name}")
            return experiment_id

    def update_experiment(self, experiment_id: int, status: str = None,
                         metrics: Dict = None, end_time: datetime = None):
        """Update experiment with results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            updates = []
            values = []

            if status:
                updates.append("status = ?")
                values.append(status)

            if metrics:
                updates.append("metrics = ?")
                values.append(json.dumps(metrics))

            if end_time:
                updates.append("end_time = ?")
                values.append(end_time.isoformat())

            if updates:
                values.append(experiment_id)
                query = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"
                cursor.execute(query, values)
                conn.commit()

                logger.info(f"Updated experiment {experiment_id}")

    def save_model(self, experiment_id: int, name: str, model_type: str,
                   task_type: str, file_path: str = None, onnx_path: str = None,
                   metrics: Dict = None, parameters: Dict = None,
                   is_best: bool = False) -> int:
        """Save model metadata to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            metrics_json = json.dumps(metrics) if metrics else None
            params_json = json.dumps(parameters) if parameters else None

            cursor.execute("""
                INSERT INTO models
                (experiment_id, name, model_type, task_type, file_path, onnx_path,
                 metrics, parameters, is_best)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, name, model_type, task_type, file_path,
                  onnx_path, metrics_json, params_json, is_best))

            model_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Saved model {model_id}: {name}")
            return model_id

    def save_dataset(self, name: str, data_type: str, file_path: str = None,
                     num_samples: int = None, num_features: int = None,
                     num_classes: int = None, description: str = None,
                     preprocessing_info: Dict = None) -> int:
        """Save dataset metadata"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            preprocessing_json = json.dumps(preprocessing_info) if preprocessing_info else None

            cursor.execute("""
                INSERT OR REPLACE INTO datasets
                (name, data_type, file_path, num_samples, num_features,
                 num_classes, description, preprocessing_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, data_type, file_path, num_samples, num_features,
                  num_classes, description, preprocessing_json))

            dataset_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Saved dataset {dataset_id}: {name}")
            return dataset_id

    def log_prediction(self, model_id: int, input_data: Any,
                      prediction: Any, confidence: float = None) -> int:
        """Log a prediction for tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            input_json = json.dumps(input_data, default=str)
            prediction_json = json.dumps(prediction, default=str)

            cursor.execute("""
                INSERT INTO predictions (model_id, input_data, prediction, confidence)
                VALUES (?, ?, ?, ?)
            """, (model_id, input_json, prediction_json, confidence))

            prediction_id = cursor.lastrowid
            conn.commit()

            return prediction_id

    def get_experiments(self, task_type: str = None, status: str = None) -> pd.DataFrame:
        """Get experiments as DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM experiments"
            conditions = []
            params = []

            if task_type:
                conditions.append("task_type = ?")
                params.append(task_type)

            if status:
                conditions.append("status = ?")
                params.append(status)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC"

            df = pd.read_sql_query(query, conn, params=params)
            return df

    def get_models(self, experiment_id: int = None, task_type: str = None,
                   is_best: bool = None) -> pd.DataFrame:
        """Get models as DataFrame"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT m.*, e.name as experiment_name
                FROM models m
                JOIN experiments e ON m.experiment_id = e.id
            """
            conditions = []
            params = []

            if experiment_id:
                conditions.append("m.experiment_id = ?")
                params.append(experiment_id)

            if task_type:
                conditions.append("m.task_type = ?")
                params.append(task_type)

            if is_best is not None:
                conditions.append("m.is_best = ?")
                params.append(is_best)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY m.created_at DESC"

            df = pd.read_sql_query(query, conn, params=params)
            return df

    def get_best_model(self, task_type: str, metric: str = None) -> Optional[Dict]:
        """Get the best model for a task type"""
        models_df = self.get_models(task_type=task_type)

        if models_df.empty:
            return None

        # If metric is specified, sort by that metric
        if metric and 'metrics' in models_df.columns:
            def extract_metric(metrics_str):
                try:
                    metrics = json.loads(metrics_str) if metrics_str else {}
                    return metrics.get(metric, 0)
                except:
                    return 0

            models_df['metric_value'] = models_df['metrics'].apply(extract_metric)
            best_model = models_df.loc[models_df['metric_value'].idxmax()]
        else:
            # Otherwise, get the most recent is_best model
            best_models = models_df[models_df['is_best'] == True]
            if not best_models.empty:
                best_model = best_models.iloc[0]
            else:
                best_model = models_df.iloc[0]

        return best_model.to_dict()

    def get_dataset_info(self, name: str) -> Optional[Dict]:
        """Get dataset information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None

    def get_experiment_summary(self) -> Dict:
        """Get summary statistics of all experiments"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total experiments
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total_experiments = cursor.fetchone()[0]

            # Experiments by status
            cursor.execute("SELECT status, COUNT(*) FROM experiments GROUP BY status")
            status_counts = dict(cursor.fetchall())

            # Experiments by task type
            cursor.execute("SELECT task_type, COUNT(*) FROM experiments GROUP BY task_type")
            task_counts = dict(cursor.fetchall())

            # Total models
            cursor.execute("SELECT COUNT(*) FROM models")
            total_models = cursor.fetchone()[0]

            return {
                "total_experiments": total_experiments,
                "status_distribution": status_counts,
                "task_distribution": task_counts,
                "total_models": total_models
            }

    def cleanup_old_predictions(self, days: int = 30):
        """Clean up old prediction logs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM predictions
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            deleted = cursor.rowcount
            conn.commit()

            logger.info(f"Cleaned up {deleted} old prediction records")
            return deleted


# Global database instance
db = DatabaseManager()
