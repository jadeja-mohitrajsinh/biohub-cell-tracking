"""Cell detection module."""

from detection.base import Detector, DetectionResult
from detection.frangi_detector import FrangiDetector
from detection.watershed import WatershedSegmenter
from detection.improved_detector import ImprovedDetector
from detection.gt_guided_detector import GTGuidedDetector
from detection.tta import TTADetector

# Optional torch-based detector
try:
    from detection.unet_detector import UNetDetector, CellDetectionUNet
    _torch_available = True
except ImportError:
    _torch_available = False
    UNetDetector = None
    CellDetectionUNet = None

__all__ = [
    "Detector",
    "DetectionResult",
    "FrangiDetector",
    "WatershedSegmenter",
    "ImprovedDetector",
    "GTGuidedDetector",
    "TTADetector",
]

if _torch_available:
    __all__.extend(["UNetDetector", "CellDetectionUNet"])
