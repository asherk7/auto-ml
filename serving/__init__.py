"""
Serving package for the AutoML system.
Contains FastAPI-based model serving, API endpoints, and inference utilities.
"""

from .api import app

__all__ = [
    'app'
]
