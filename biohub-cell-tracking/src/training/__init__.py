"""Training loops (Phase 4+)."""

from training.label_generator import LabelGenerator, WeightedLabelGenerator
from training.augmentation import AnisotropicAugmentation, TrainingAugmentation, ValidationAugmentation
from training.dataset import CellDetectionDataset, MultiSampleDataset, LazyZarrDataset, create_training_datasets

# Commented out to avoid monai dependency
# from training.trainer import CombinedLoss, CellDetectionTrainer, create_trainer

__all__ = [
    "LabelGenerator",
    "WeightedLabelGenerator",
    "AnisotropicAugmentation",
    "TrainingAugmentation",
    "ValidationAugmentation",
    "CellDetectionDataset",
    "MultiSampleDataset",
    "LazyZarrDataset",
    "create_training_datasets",
    # "CombinedLoss",
    # "CellDetectionTrainer",
    # "create_trainer",
]
