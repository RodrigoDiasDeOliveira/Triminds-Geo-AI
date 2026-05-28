"""
Satellite Land Classification Package

A scalable deep learning project for crop and land cover classification
using satellite imagery with PyTorch and Spark on GCP.
"""

__version__ = "0.1.0"
__author__ = "Rodrigo Dias de Oliveira"

# Optional: expose main modules
from . import data_loader
from . import models
from . import training