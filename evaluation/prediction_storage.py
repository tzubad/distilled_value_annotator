# Prediction storage for managing model predictions

import logging
from typing import Dict, List, Optional
from evaluation.models import PredictionResult, PredictionSet


class PredictionStorage:
    """
    Manages storage and retrieval of model predictions.
    
    Ensures isolation between different models' predictions and tracks
    success/failure counts per model.
    """
    
    def __init__(self):
        """Initialize empty prediction storage."""
        # Model name -> PredictionSet
        self._predictions: Dict[str, PredictionSet] = {}
        
        # Model name -> video_id -> PredictionResult (for fast lookup)
        self._prediction_index: Dict[str, Dict[str, PredictionResult]] = {}
        
        logging.info("PredictionStorage initialized")
    
    def store_predictions(
        self,
        model_name: str,
        predictions: List[PredictionResult]
    ) -> None:
        """
        Store predictions for a model.
        
        Args:
            model_name: Name of the model
            predictions: List of PredictionResult objects
        
        Raises:
            ValueError: If model_name is empty or predictions list is empty
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")
        
        if not predictions:
            raise ValueError("predictions list cannot be empty")
        
        # Calculate statistics
        success_count = sum(1 for p in predictions if p.success)
        failure_count = sum(1 for p in predictions if not p.success)
        failed_video_ids = [p.video_id for p in predictions if not p.success]
        
        # Create PredictionSet
        prediction_set = PredictionSet(
            model_name=model_name,
            predictions=predictions,
            total_count=len(predictions),
            success_count=success_count,
            failure_count=failure_count,
            failed_video_ids=failed_video_ids
        )
        
        # Store in main dictionary
        self._predictions[model_name] = prediction_set
        
        # Build index for fast video_id lookup
        self._prediction_index[model_name] = {
            pred.video_id: pred for pred in predictions
        }
        
        logging.info(
            f"Stored {len(predictions)} predictions for model '{model_name}' "
            f"({success_count} successful, {failure_count} failed)"
        )
    
    def get_predictions(self, model_name: str) -> Optional[PredictionSet]:
        """
        Retrieve all predictions for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            PredictionSet for the model, or None if not found
        """
        return self._predictions.get(model_name)
    
    def get_prediction_for_video(
        self,
        model_name: str,
        video_id: str
    ) -> Optional[PredictionResult]:
        """
        Retrieve prediction for a specific video from a model.
        
        Args:
            model_name: Name of the model
            video_id: ID of the video
        
        Returns:
            PredictionResult for the video, or None if not found
        """
        if model_name not in self._prediction_index:
            return None
        
        return self._prediction_index[model_name].get(video_id)
    
    def get_success_rate(self, model_name: str) -> float:
        """
        Get success rate for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Success rate (0.0 to 1.0), or 0.0 if model not found
        """
        prediction_set = self._predictions.get(model_name)
        if not prediction_set or prediction_set.total_count == 0:
            return 0.0
        
        return prediction_set.success_count / prediction_set.total_count
    
    def get_all_model_names(self) -> List[str]:
        """
        Get list of all models with stored predictions.
        
        Returns:
            List of model names
        """
        return list(self._predictions.keys())
    
    def has_predictions(self, model_name: str) -> bool:
        """
        Check if predictions exist for a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            True if predictions exist, False otherwise
        """
        return model_name in self._predictions
    
    def get_statistics(self, model_name: str) -> Optional[Dict[str, int]]:
        """
        Get statistics for a model's predictions.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dictionary with statistics, or None if model not found
        """
        prediction_set = self._predictions.get(model_name)
        if not prediction_set:
            return None
        
        return {
            'total_count': prediction_set.total_count,
            'success_count': prediction_set.success_count,
            'failure_count': prediction_set.failure_count,
            'success_rate': self.get_success_rate(model_name)
        }
    
    def clear(self) -> None:
        """Clear all stored predictions."""
        self._predictions.clear()
        self._prediction_index.clear()
        logging.info("PredictionStorage cleared")
    
    def remove_model_predictions(self, model_name: str) -> bool:
        """
        Remove predictions for a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            True if predictions were removed, False if model not found
        """
        if model_name not in self._predictions:
            return False
        
        del self._predictions[model_name]
        del self._prediction_index[model_name]
        
        logging.info(f"Removed predictions for model '{model_name}'")
        return True
