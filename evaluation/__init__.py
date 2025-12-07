# Evaluation module for model performance testing and benchmarking

from .models import (
    VideoAnnotation,
    PredictionResult,
    GroundTruthDataset,
    PredictionSet,
    MetricScores,
    CategoryMetrics,
    AggregateScores,
    AggregateMetrics,
    ModelMetrics,
    EvaluationConfig,
    ModelConfig,
)
from .ground_truth_loader import (
    GroundTruthLoader,
    ValidationResult,
    ANNOTATION_CATEGORIES,
)
from .prediction_storage import PredictionStorage
from .config_loader import (
    EvaluationConfigLoader,
    ConfigValidationError,
)
from .orchestrator import (
    EvaluationOrchestrator,
    EvaluationSummary,
    ModelInitializationResult,
)
from .video_id_utils import (
    normalize_video_id,
    extract_username,
    extract_video_number,
)

__all__ = [
    'VideoAnnotation',
    'PredictionResult',
    'GroundTruthDataset',
    'PredictionSet',
    'MetricScores',
    'CategoryMetrics',
    'AggregateScores',
    'AggregateMetrics',
    'ModelMetrics',
    'EvaluationConfig',
    'ModelConfig',
    'GroundTruthLoader',
    'ValidationResult',
    'ANNOTATION_CATEGORIES',
    'PredictionStorage',
    'EvaluationConfigLoader',
    'ConfigValidationError',
    'EvaluationOrchestrator',
    'EvaluationSummary',
    'ModelInitializationResult',
    'normalize_video_id',
    'extract_username',
    'extract_video_number',
]
