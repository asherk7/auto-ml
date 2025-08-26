"""
Models package for the AutoML system.
Contains computer vision and tabular machine learning model definitions.
"""

from .cv_models import (
    ImageClassifier,
    SimpleObjectDetector,
    CVModelFactory,
    create_cv_model
)

from .tabular_models import (
    TabularClassifier,
    TabularRegressor,
    TabularClusterer,
    TabularModelFactory,
    create_classifier,
    create_regressor,
    create_clusterer,
    save_model,
    load_model
)

__all__ = [
    # Computer Vision Models
    'ImageClassifier',
    'SimpleObjectDetector',
    'CVModelFactory',
    'create_cv_model',

    # Tabular Models
    'TabularClassifier',
    'TabularRegressor',
    'TabularClusterer',
    'TabularModelFactory',
    'create_classifier',
    'create_regressor',
    'create_clusterer',
    'save_model',
    'load_model'
]
