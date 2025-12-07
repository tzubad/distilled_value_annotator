# Data models for the evaluation module

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class VideoAnnotation:
    """
    Represents a video with its ground truth annotations.
    
    Attributes:
        video_id: Unique identifier for the video
        video_uri: GCS URI or path to the video file
        script_uri: Path to the script file (required for all models)
        script_text: Cached script content for efficiency (optional)
        annotations: Dictionary mapping category names to annotation values (-1, 0, 1, 2)
        has_sound: Whether the video has audio
    """
    video_id: str
    video_uri: str
    script_uri: str
    annotations: Dict[str, int]
    has_sound: bool
    script_text: Optional[str] = None
    
    def __post_init__(self):
        """Validate annotation values are in valid range."""
        valid_values = {-1, 0, 1, 2}
        for category, value in self.annotations.items():
            if value not in valid_values:
                raise ValueError(
                    f"Invalid annotation value {value} for category {category}. "
                    f"Must be one of {valid_values}"
                )


@dataclass
class PredictionResult:
    """
    Represents prediction results for a single video from a model.
    
    Attributes:
        video_id: Unique identifier for the video
        predictions: Dictionary mapping category names to predicted values
        success: Whether the prediction was successful
        error_message: Error message if prediction failed
        inference_time: Time taken for inference in seconds
    """
    video_id: str
    predictions: Dict[str, int]
    success: bool
    error_message: Optional[str] = None
    inference_time: float = 0.0
    
    def __post_init__(self):
        """Validate prediction values are in valid range if successful."""
        if self.success:
            valid_values = {-1, 0, 1, 2}
            for category, value in self.predictions.items():
                if value not in valid_values:
                    raise ValueError(
                        f"Invalid prediction value {value} for category {category}. "
                        f"Must be one of {valid_values}"
                    )


@dataclass
class GroundTruthDataset:
    """
    Represents the complete ground truth dataset.
    
    Attributes:
        videos: List of VideoAnnotation objects
        total_count: Total number of videos in the dataset
        valid_count: Number of valid videos after validation
        validation_errors: List of validation error messages
    """
    videos: List[VideoAnnotation]
    total_count: int
    valid_count: int
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class PredictionSet:
    """
    Represents all predictions from a single model.
    
    Attributes:
        model_name: Name of the model that generated predictions
        predictions: List of PredictionResult objects
        total_count: Total number of prediction attempts
        success_count: Number of successful predictions
        failure_count: Number of failed predictions
        failed_video_ids: List of video IDs that failed
    """
    model_name: str
    predictions: List[PredictionResult]
    total_count: int
    success_count: int
    failure_count: int
    failed_video_ids: List[str] = field(default_factory=list)


@dataclass
class MetricScores:
    """
    Represents precision, recall, and F1 scores for a specific classification.
    
    Attributes:
        precision: Precision score (0.0 to 1.0)
        recall: Recall score (0.0 to 1.0)
        f1: F1 score (0.0 to 1.0)
        support: Number of true instances in the ground truth
    """
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class CategoryMetrics:
    """
    Represents metrics for a single annotation category.
    
    Attributes:
        category_name: Name of the annotation category
        endorsed: Metrics treating endorsed (1,2) as positive class
        conflict: Metrics treating conflict (-1) as positive class
        combined: Metrics considering all value types
    """
    category_name: str
    endorsed: MetricScores
    conflict: MetricScores
    combined: MetricScores


@dataclass
class AggregateScores:
    """
    Represents aggregate metrics across all categories.
    
    Attributes:
        macro_f1: Unweighted mean of category F1 scores
        weighted_f1: Weighted mean of category F1 scores by support
        macro_precision: Unweighted mean of category precision scores
        macro_recall: Unweighted mean of category recall scores
        weighted_precision: Weighted mean of category precision scores
        weighted_recall: Weighted mean of category recall scores
    """
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    weighted_precision: float
    weighted_recall: float


@dataclass
class AggregateMetrics:
    """
    Represents aggregate metrics for endorsed, conflict, and combined classifications.
    
    Attributes:
        endorsed: Aggregate scores for endorsed values (1, 2)
        conflict: Aggregate scores for conflict values (-1)
        combined: Aggregate scores for all value types
    """
    endorsed: AggregateScores
    conflict: AggregateScores
    combined: AggregateScores


@dataclass
class ModelMetrics:
    """
    Represents complete evaluation metrics for a single model.
    
    Attributes:
        model_name: Name of the evaluated model
        per_category: Dictionary mapping category names to CategoryMetrics
        aggregate: Aggregate metrics across all categories
        excluded_categories: Categories excluded from aggregate metrics due to low frequency
        matched_videos: Number of videos with both ground truth and predictions
        unmatched_predictions: Number of predictions without corresponding ground truth
        missing_predictions: Number of ground truth videos without predictions
        success_rate: Proportion of successful predictions
        total_videos: Total number of videos in ground truth
        successful_predictions: Number of successful predictions
    """
    model_name: str
    per_category: Dict[str, CategoryMetrics]
    aggregate: AggregateMetrics
    excluded_categories: List[str]
    matched_videos: int
    unmatched_predictions: int
    missing_predictions: int
    success_rate: float
    total_videos: int
    successful_predictions: int


@dataclass
class ModelConfig:
    """
    Configuration for a single model to evaluate.
    
    Attributes:
        model_type: Type of model (e.g., 'gemini', 'roberta', 'deberta')
        model_name: Specific model identifier
        adapter_class: Python class name for the adapter
        config: Model-specific configuration parameters
    """
    model_type: str
    model_name: str
    adapter_class: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """
    Complete configuration for evaluation run.
    
    Attributes:
        ground_truth_path: Path to ground truth dataset file
        scripts_path: Path to folder containing video scripts
        output_dir: Directory for output reports
        models: List of ModelConfig objects
        sample_size: Optional number of videos to sample (None = use all)
        random_seed: Random seed for reproducible sampling
        min_frequency_threshold: Minimum frequency for including categories in aggregate metrics
        parallel_execution: Whether to run models in parallel
        max_workers: Maximum number of parallel workers
    """
    ground_truth_path: str
    scripts_path: str
    output_dir: str
    models: List[ModelConfig]
    sample_size: Optional[int] = None
    random_seed: int = 42
    min_frequency_threshold: float = 0.05
    parallel_execution: bool = False
    max_workers: int = 4
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_frequency_threshold < 0.0 or self.min_frequency_threshold > 1.0:
            raise ValueError(
                f"min_frequency_threshold must be between 0.0 and 1.0, "
                f"got {self.min_frequency_threshold}"
            )
        
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError(
                f"sample_size must be positive, got {self.sample_size}"
            )
        
        if self.max_workers <= 0:
            raise ValueError(
                f"max_workers must be positive, got {self.max_workers}"
            )
