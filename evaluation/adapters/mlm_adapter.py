# MLM (Masked Language Model) adapter for video annotation evaluation

import logging
import time
from typing import Optional, Dict, Any
from abc import abstractmethod

from evaluation.adapters.base import ModelAdapter
from evaluation.adapters.script_loader import ScriptLoader
from evaluation.models import VideoAnnotation, PredictionResult


# Define the 19 annotation categories
ANNOTATION_CATEGORIES = [
    "Self_Direction_Thought",
    "Self_Direction_Action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power_Resources",
    "Power_Dominance",
    "Face",
    "Security_Personal",
    "Security_Social",
    "Conformity_Rules",
    "Conformity_Interpersonal",
    "Tradition",
    "Humility",
    "Benevolence_Dependability",
    "Benevolence_Care",
    "Universalism_Concern",
    "Universalism_Nature",
    "Universalism_Tolerance",
]


class MLMAdapter(ModelAdapter):
    """
    Base adapter for Masked Language Models (MLM) like RoBERTa and DeBERTa.
    
    This adapter loads models from HuggingFace and performs per-category
    classification on video scripts to generate annotations.
    
    Requirements: 2.1, 2.2, 2.3, 7.2, 7.3
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize the MLM adapter.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'roberta-large')
            config: Configuration including:
                - max_length: Maximum sequence length for tokenizer (default: 512)
                - device: Device to use ('cuda', 'cpu', or 'auto') (default: 'auto')
                - batch_size: Batch size for processing (default: 16)
                - padding: Padding strategy (default: True)
                - truncation: Truncation strategy (default: True)
        """
        super().__init__(model_name, config)
        
        self.max_length = config.get('max_length', 512)
        self.device = config.get('device', 'auto')
        self.batch_size = config.get('batch_size', 16)
        self.padding = config.get('padding', True)
        self.truncation = config.get('truncation', True)
        
        self.script_loader = ScriptLoader()
        self.model = None
        self.tokenizer = None
    
    def initialize(self) -> bool:
        """
        Load model and tokenizer from HuggingFace.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer()
            
            if model is None or tokenizer is None:
                self.logger.error("Failed to load model or tokenizer")
                return False
            
            self.model = model
            self.tokenizer = tokenizer
            
            # Move model to appropriate device
            self._setup_device()
            
            self.logger.info(
                f"Successfully initialized MLM model: {self.model_name} on {self.device}"
            )
            return True
        
        except Exception as e:
            self.logger.error(f"Error initializing MLM adapter: {str(e)}", exc_info=True)
            return False
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        """
        Generate predictions for a single video.
        
        Loads the video script and performs per-category classification
        to generate annotations for all 19 categories.
        
        Args:
            video: VideoAnnotation object containing video metadata and script
        
        Returns:
            PredictionResult with predictions for all 19 categories,
            or None if prediction fails
        """
        start_time = time.time()
        
        try:
            # Load script content
            script_text = self._load_script(video)
            if script_text is None:
                return PredictionResult(
                    video_id=video.video_id,
                    predictions={},
                    success=False,
                    error_message="Failed to load script",
                    inference_time=time.time() - start_time
                )
            
            # Prepare input text
            prepared_text = self._prepare_input(script_text)
            
            # Classify each category
            predictions = {}
            for category in ANNOTATION_CATEGORIES:
                try:
                    value = self._classify_category(prepared_text, category)
                    predictions[category] = value
                except Exception as e:
                    self.logger.warning(
                        f"Error classifying category {category} for video {video.video_id}: {str(e)}"
                    )
                    # Use default value of 0 (absent) on error
                    predictions[category] = 0
            
            # Verify we have all 19 categories
            if len(predictions) != 19:
                return PredictionResult(
                    video_id=video.video_id,
                    predictions=predictions,
                    success=False,
                    error_message=f"Incomplete predictions: got {len(predictions)}/19 categories",
                    inference_time=time.time() - start_time
                )
            
            return PredictionResult(
                video_id=video.video_id,
                predictions=predictions,
                success=True,
                inference_time=time.time() - start_time
            )
        
        except Exception as e:
            self.logger.error(
                f"Error predicting video {video.video_id}: {str(e)}",
                exc_info=True
            )
            return PredictionResult(
                video_id=video.video_id,
                predictions={},
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                inference_time=time.time() - start_time
            )
    
    def get_model_type(self) -> str:
        """Return the type of model."""
        return "MLM"
    
    def get_model_name(self) -> str:
        """Return the specific model identifier."""
        return self.model_name
    
    @abstractmethod
    def _load_model_and_tokenizer(self):
        """
        Load model and tokenizer from HuggingFace.
        
        This method should be implemented by subclasses to load
        model-specific implementations.
        
        Returns:
            Tuple of (model, tokenizer), or (None, None) if loading fails
        """
        pass
    
    def _setup_device(self):
        """
        Setup device for model inference.
        
        Moves model to the appropriate device (CPU or GPU).
        """
        try:
            import torch
            
            if self.device == 'auto':
                # Auto-detect GPU availability
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.logger.info("GPU detected, using CUDA")
                else:
                    self.device = 'cpu'
                    self.logger.info("No GPU detected, using CPU")
            
            # Move model to device
            if self.model is not None:
                self.model.to(self.device)
                self.logger.info(f"Model moved to {self.device}")
        
        except ImportError:
            self.logger.warning("PyTorch not available, using CPU")
            self.device = 'cpu'
        except Exception as e:
            self.logger.warning(f"Error setting up device: {str(e)}, defaulting to CPU")
            self.device = 'cpu'
    
    def _load_script(self, video: VideoAnnotation) -> Optional[str]:
        """
        Load script for a video.
        
        Uses cached script_text if available, otherwise loads from script_uri.
        
        Args:
            video: VideoAnnotation object
        
        Returns:
            Script text, or None if loading fails
        """
        # Use cached script if available
        if video.script_text is not None:
            return video.script_text
        
        # Load from script_uri
        return self.script_loader.load_script(video.script_uri)
    
    def _prepare_input(self, script_text: str) -> str:
        """
        Prepare input text for classification.
        
        This method formats the script text for model input.
        Subclasses can override for model-specific formatting.
        
        Args:
            script_text: Raw script text
        
        Returns:
            Formatted text ready for tokenization
        """
        # Basic preparation: strip whitespace and ensure non-empty
        text = script_text.strip()
        
        if not text:
            # Return a minimal text to avoid tokenization errors
            return "No content available."
        
        return text
    
    def _classify_category(self, text: str, category: str) -> int:
        """
        Classify a single category for the given text.
        
        This method performs classification to determine the annotation value
        for a specific category. The implementation maps model outputs to the
        standard annotation format {-1, 0, 1, 2}.
        
        Args:
            text: Prepared input text
            category: Category name to classify
        
        Returns:
            Annotation value: -1 (conflict), 0 (absent), 1 (present), or 2 (strong present)
        
        Requirements: 2.3, 7.3
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            import torch
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract logits (assuming classification head exists)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Map logits to annotation values
            # This is a simplified mapping - subclasses should override for specific models
            predicted_value = self._map_logits_to_annotation(logits, category)
            
            return predicted_value
        
        except Exception as e:
            self.logger.error(
                f"Error classifying category {category}: {str(e)}",
                exc_info=True
            )
            # Return default value on error
            return 0
    
    def _map_logits_to_annotation(self, logits, category: str) -> int:
        """
        Map model logits to annotation values.
        
        This is a default implementation that maps logits to {-1, 0, 1, 2}.
        Subclasses should override this method for model-specific mapping logic.
        
        Args:
            logits: Model output logits
            category: Category being classified
        
        Returns:
            Annotation value in {-1, 0, 1, 2}
        """
        import torch
        
        # Default implementation: use argmax over 4 classes
        # Assumes model has 4 output classes corresponding to {-1, 0, 1, 2}
        if logits.shape[-1] >= 4:
            predicted_class = torch.argmax(logits, dim=-1).item()
            # Map class indices to annotation values
            # 0 -> -1, 1 -> 0, 2 -> 1, 3 -> 2
            mapping = {0: -1, 1: 0, 2: 1, 3: 2}
            return mapping.get(predicted_class, 0)
        else:
            # If model doesn't have 4 classes, use binary classification
            # and map to {0, 1}
            predicted_class = torch.argmax(logits, dim=-1).item()
            return 1 if predicted_class == 1 else 0



class RoBERTaAdapter(MLMAdapter):
    """
    Adapter for RoBERTa models.
    
    Implements model-specific initialization and configuration for
    RoBERTa-based classification models.
    
    Requirements: 2.1
    """
    
    def _load_model_and_tokenizer(self):
        """
        Load RoBERTa model and tokenizer from HuggingFace.
        
        Returns:
            Tuple of (model, tokenizer), or (None, None) if loading fails
        """
        try:
            from transformers import RobertaForSequenceClassification, RobertaTokenizer
            
            self.logger.info(f"Loading RoBERTa model: {self.model_name}")
            
            # Load tokenizer
            tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            
            # Load model
            # Note: This assumes the model has a classification head
            # For fine-tuned models, use the appropriate model class
            model = RobertaForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=4  # 4 classes for {-1, 0, 1, 2}
            )
            
            self.logger.info(f"Successfully loaded RoBERTa model: {self.model_name}")
            return model, tokenizer
        
        except ImportError as e:
            self.logger.error(f"Failed to import transformers library: {str(e)}")
            return None, None
        except Exception as e:
            self.logger.error(f"Error loading RoBERTa model: {str(e)}", exc_info=True)
            return None, None


class DeBERTaAdapter(MLMAdapter):
    """
    Adapter for DeBERTa models.
    
    Implements model-specific initialization and configuration for
    DeBERTa-based classification models.
    
    Requirements: 2.1
    """
    
    def _load_model_and_tokenizer(self):
        """
        Load DeBERTa model and tokenizer from HuggingFace.
        
        Returns:
            Tuple of (model, tokenizer), or (None, None) if loading fails
        """
        try:
            from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
            
            self.logger.info(f"Loading DeBERTa model: {self.model_name}")
            
            # Load tokenizer
            tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
            
            # Load model
            # Note: This assumes the model has a classification head
            # For fine-tuned models, use the appropriate model class
            model = DebertaV2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=4  # 4 classes for {-1, 0, 1, 2}
            )
            
            self.logger.info(f"Successfully loaded DeBERTa model: {self.model_name}")
            return model, tokenizer
        
        except ImportError as e:
            self.logger.error(f"Failed to import transformers library: {str(e)}")
            return None, None
        except Exception as e:
            self.logger.error(f"Error loading DeBERTa model: {str(e)}", exc_info=True)
            return None, None
