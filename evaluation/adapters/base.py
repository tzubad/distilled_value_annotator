# Base model adapter interface for unified model evaluation

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging
import time

from evaluation.models import VideoAnnotation, PredictionResult


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Provides a unified interface for all model types (LLM, MLM, etc.)
    to enable consistent evaluation across different architectures.
    
    Requirements: 7.1, 7.4
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the model adapter.
        
        Args:
            model_name: Unique identifier for the model
            config: Model-specific configuration parameters
        """
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{model_name}]")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Load model and prepare for inference.
        
        This method should handle all model loading, authentication,
        and initialization required before predictions can be made.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        """
        Generate predictions for a single video.
        
        Args:
            video: VideoAnnotation object containing video metadata and script
        
        Returns:
            PredictionResult with predictions for all 19 categories,
            or None if prediction fails
        """
        pass
    
    @abstractmethod
    def get_model_type(self) -> str:
        """
        Return the type of model.
        
        Returns:
            Model type identifier (e.g., 'LLM', 'MLM')
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the specific model identifier.
        
        Returns:
            Model name (e.g., 'gemini-1.5-pro-002', 'roberta-large')
        """
        pass
    
    def batch_predict(self, videos: List[VideoAnnotation]) -> List[PredictionResult]:
        """
        Process multiple videos with error handling.
        
        This method provides a default implementation that processes videos
        sequentially with error handling. Subclasses can override for
        optimized batch processing.
        
        Args:
            videos: List of VideoAnnotation objects to process
        
        Returns:
            List of PredictionResult objects (includes both successful and failed predictions)
        """
        results = []
        
        self.logger.info(f"Starting batch prediction for {len(videos)} videos")
        
        for idx, video in enumerate(videos, 1):
            try:
                self.logger.debug(f"Processing video {idx}/{len(videos)}: {video.video_id}")
                
                start_time = time.time()
                result = self.predict(video)
                
                if result is None:
                    # Create a failed prediction result
                    result = PredictionResult(
                        video_id=video.video_id,
                        predictions={},
                        success=False,
                        error_message="predict() returned None",
                        inference_time=time.time() - start_time
                    )
                    self.logger.warning(f"Prediction returned None for video {video.video_id}")
                
                results.append(result)
                
                if result.success:
                    self.logger.debug(f"Successfully predicted video {video.video_id}")
                else:
                    self.logger.warning(
                        f"Failed to predict video {video.video_id}: {result.error_message}"
                    )
            
            except Exception as e:
                # Handle unexpected errors during prediction
                self.logger.error(
                    f"Unexpected error processing video {video.video_id}: {str(e)}",
                    exc_info=True
                )
                
                results.append(PredictionResult(
                    video_id=video.video_id,
                    predictions={},
                    success=False,
                    error_message=f"Unexpected error: {str(e)}",
                    inference_time=0.0
                ))
        
        success_count = sum(1 for r in results if r.success)
        self.logger.info(
            f"Batch prediction complete: {success_count}/{len(videos)} successful"
        )
        
        return results
