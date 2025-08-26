"""
Utilities package for the AutoML system.
Contains database operations, helper functions, and common utilities.
"""

from .database import db, DatabaseManager
from .helpers import (
    setup_logging,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    calculate_model_size,
    create_confusion_matrix_plot,
    create_feature_importance_plot,
    create_learning_curve_plot,
    visualize_data_distribution,
    create_model_comparison_plot,
    generate_classification_report,
    validate_data_format,
    estimate_training_time,
    clean_experiment_name,
    get_system_info,
    create_experiment_report
)

__all__ = [
    # Database
    'db',
    'DatabaseManager',

    # File operations
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',

    # Logging and system
    'setup_logging',
    'get_system_info',

    # Model utilities
    'calculate_model_size',
    'validate_data_format',
    'estimate_training_time',
    'clean_experiment_name',

    # Visualization
    'create_confusion_matrix_plot',
    'create_feature_importance_plot',
    'create_learning_curve_plot',
    'visualize_data_distribution',
    'create_model_comparison_plot',

    # Evaluation
    'generate_classification_report',
    'create_experiment_report'
]
