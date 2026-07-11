"""Pipeline module for orchestrating detection and tracking."""

from pipeline.base import Pipeline
from pipeline.classical_pipeline import ClassicalPipeline

__all__ = ["Pipeline", "ClassicalPipeline"]
