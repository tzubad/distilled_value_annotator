# Evaluation orchestrator for coordinating the entire evaluation workflow

import logging
import time
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field

from evaluation.models import (
    EvaluationConfig,
    ModelConfig,
    GroundTruthDataset,
    PredictionSet,
    PredictionResult,
)
from evaluation.config_loader import EvaluationConfigLoader, ConfigValidationError
from evaluation.ground_truth_loader import GroundTruthLoader
from evaluation.prediction_storage import PredictionStorage
from evaluation.adapters.base import ModelAdapter
from evaluation.metrics.calculator import MetricsCalculator, ModelEvaluationResult
from evaluation.reports.generator import ReportGenerator


@dataclass
class ModelInitializationResult:
    """Result of initializing a model adapter."""
    model_name: str
    adapter: Optional[ModelAdapter] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary of the entire evaluation run."""
    total_models: int
    successful_models: int
    failed_models: int
    total_videos: int
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    elapsed_time: float
    model_results: Dict[str, ModelEvaluationResult] = field(default_factory=dict)
    model_errors: Dict[str, str] = field(default_factory=dict)
    generated_reports: Dict[str, Path] = field(default_factory=dict)


class EvaluationOrchestrator:
    """
    Orchestrates the entire evaluation workflow.
    
    Coordinates:
    - Configuration loading and validation
    - Ground truth loading
    - Model initialization
    - Prediction execution
    - Metrics calculation
    - Report generation
    
    Requirements: 1.1, 1.5, 2.5, 3.1, 5.9, 8.1, 8.4, 9.1, 10.1-10.4
    """
    
    # Registry of known adapter classes
    _adapter_registry: Dict[str, Type[ModelAdapter]] = {}
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config: EvaluationConfig object (takes precedence if provided)
            config_path: Path to configuration file (used if config not provided)
            
        Raises:
            ValueError: If neither config nor config_path is provided
            ConfigValidationError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist
        """
        if config is None and config_path is None:
            raise ValueError("Either config or config_path must be provided")
        
        if config is not None:
            self._config = config
        else:
            loader = EvaluationConfigLoader()
            self._config = loader.load(config_path)
        
        # Set up logging
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components (lazy loading)
        self._ground_truth: Optional[GroundTruthDataset] = None
        self._prediction_storage: Optional[PredictionStorage] = None
        self._adapters: Dict[str, ModelAdapter] = {}
        self._adapter_errors: Dict[str, str] = {}
        
        self._logger.info(
            f"EvaluationOrchestrator initialized with {len(self._config.models)} models"
        )
    
    @property
    def config(self) -> EvaluationConfig:
        """Get the evaluation configuration."""
        return self._config
    
    @property
    def ground_truth(self) -> Optional[GroundTruthDataset]:
        """Get the loaded ground truth dataset."""
        return self._ground_truth
    
    @property
    def adapters(self) -> Dict[str, ModelAdapter]:
        """Get the initialized adapters."""
        return dict(self._adapters)
    
    @property
    def adapter_errors(self) -> Dict[str, str]:
        """Get errors from adapter initialization."""
        return dict(self._adapter_errors)
    
    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[ModelAdapter]) -> None:
        """
        Register an adapter class for dynamic loading.
        
        Args:
            name: Name to register the adapter under
            adapter_class: The adapter class to register
        """
        cls._adapter_registry[name] = adapter_class
        logging.getLogger(cls.__name__).debug(f"Registered adapter: {name}")
    
    @classmethod
    def get_registered_adapters(cls) -> Dict[str, Type[ModelAdapter]]:
        """Get a copy of the adapter registry."""
        return dict(cls._adapter_registry)
    
    def load_ground_truth(self) -> GroundTruthDataset:
        """
        Load the ground truth dataset.
        
        Returns:
            Loaded and validated GroundTruthDataset
            
        Raises:
            FileNotFoundError: If ground truth file doesn't exist
            ValueError: If ground truth data is invalid
        """
        self._logger.info(f"Loading ground truth from: {self._config.ground_truth_path}")
        
        loader = GroundTruthLoader(
            dataset_path=self._config.ground_truth_path,
            sample_size=self._config.sample_size,
            random_seed=self._config.random_seed,
            scripts_path=self._config.scripts_path,
        )
        self._ground_truth = loader.load()
        
        self._logger.info(
            f"Loaded {self._ground_truth.valid_count} valid videos "
            f"(total: {self._ground_truth.total_count})"
        )
        
        if self._ground_truth.validation_errors:
            self._logger.warning(
                f"Ground truth has {len(self._ground_truth.validation_errors)} validation errors"
            )
        
        return self._ground_truth
    
    def initialize_adapters(self) -> List[ModelInitializationResult]:
        """
        Initialize all model adapters based on configuration.
        
        Failed adapters are logged but don't stop other adapters from initializing.
        
        Returns:
            List of initialization results for each model
        """
        results = []
        
        for model_config in self._config.models:
            result = self._initialize_single_adapter(model_config)
            results.append(result)
            
            if result.success:
                self._adapters[model_config.model_name] = result.adapter
                self._logger.info(f"Successfully initialized adapter: {model_config.model_name}")
            else:
                self._adapter_errors[model_config.model_name] = result.error_message
                self._logger.error(
                    f"Failed to initialize adapter {model_config.model_name}: {result.error_message}"
                )
        
        success_count = sum(1 for r in results if r.success)
        self._logger.info(
            f"Adapter initialization complete: {success_count}/{len(results)} successful"
        )
        
        return results
    
    def _initialize_single_adapter(self, model_config: ModelConfig) -> ModelInitializationResult:
        """
        Initialize a single model adapter.
        
        Args:
            model_config: Configuration for the model
            
        Returns:
            ModelInitializationResult with success/failure status
        """
        result = ModelInitializationResult(model_name=model_config.model_name)
        
        try:
            # Try to get adapter class from registry first
            adapter_class = self._adapter_registry.get(model_config.adapter_class)
            
            # If not in registry, try dynamic import
            if adapter_class is None:
                adapter_class = self._import_adapter_class(model_config.adapter_class)
            
            if adapter_class is None:
                result.error_message = f"Could not find adapter class: {model_config.adapter_class}"
                return result
            
            # Create adapter instance
            adapter = adapter_class(
                model_name=model_config.model_name,
                config=model_config.config,
            )
            
            # Initialize the adapter
            if adapter.initialize():
                result.adapter = adapter
                result.success = True
            else:
                result.error_message = "Adapter initialize() returned False"
        
        except Exception as e:
            self._logger.error(
                f"Exception initializing adapter {model_config.model_name}: {e}",
                exc_info=True
            )
            result.error_message = str(e)
        
        return result
    
    def _import_adapter_class(self, class_path: str) -> Optional[Type[ModelAdapter]]:
        """
        Dynamically import an adapter class.
        
        Args:
            class_path: Either a simple class name or module.path.ClassName
            
        Returns:
            The adapter class, or None if import fails
        """
        try:
            if '.' in class_path:
                # Full module path provided
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            else:
                # Try common locations
                search_paths = [
                    f"evaluation.adapters.{class_path.lower()}",
                    f"evaluation.adapters.gemini_adapter",
                    f"evaluation.adapters.mlm_adapter",
                ]
                
                for path in search_paths:
                    try:
                        module = importlib.import_module(path)
                        if hasattr(module, class_path):
                            return getattr(module, class_path)
                    except ImportError:
                        continue
                
                return None
        
        except (ImportError, AttributeError) as e:
            self._logger.warning(f"Could not import adapter class {class_path}: {e}")
            return None
    
    def run_predictions(
        self,
        model_names: Optional[List[str]] = None,
    ) -> PredictionStorage:
        """
        Run predictions for all videos through specified models.
        
        Args:
            model_names: Optional list of model names to run (default: all initialized)
            
        Returns:
            PredictionStorage containing all prediction results
            
        Raises:
            RuntimeError: If ground truth not loaded or no adapters initialized
        """
        if self._ground_truth is None:
            raise RuntimeError("Ground truth not loaded. Call load_ground_truth() first.")
        
        if not self._adapters:
            raise RuntimeError("No adapters initialized. Call initialize_adapters() first.")
        
        # Initialize prediction storage if needed
        if self._prediction_storage is None:
            self._prediction_storage = PredictionStorage()
        
        # Determine which models to run
        models_to_run = model_names or list(self._adapters.keys())
        
        self._logger.info(
            f"Starting predictions for {len(self._ground_truth.videos)} videos "
            f"across {len(models_to_run)} models"
        )
        
        for model_name in models_to_run:
            if model_name not in self._adapters:
                self._logger.warning(f"Model {model_name} not found in initialized adapters")
                continue
            
            adapter = self._adapters[model_name]
            self._run_predictions_for_model(adapter)
        
        return self._prediction_storage
    
    def _run_predictions_for_model(self, adapter: ModelAdapter) -> PredictionSet:
        """
        Run predictions for a single model.
        
        Args:
            adapter: The model adapter to use
            
        Returns:
            PredictionSet with all prediction results
        """
        model_name = adapter.get_model_name()
        
        self._logger.info(f"Running predictions for model: {model_name}")
        start_time = time.time()
        
        # Get predictions using batch_predict (handles errors internally)
        predictions = adapter.batch_predict(self._ground_truth.videos)
        
        elapsed = time.time() - start_time
        
        # Count results
        success_count = sum(1 for p in predictions if p.success)
        failure_count = len(predictions) - success_count
        failed_ids = [p.video_id for p in predictions if not p.success]
        
        # Store predictions (only if not empty - storage rejects empty lists)
        if predictions:
            self._prediction_storage.store_predictions(model_name, predictions)
            prediction_set = self._prediction_storage.get_predictions(model_name)
        else:
            # Create an empty prediction set for models with no videos to process
            prediction_set = PredictionSet(
                model_name=model_name,
                predictions=[],
                total_count=0,
                success_count=0,
                failure_count=0,
                failed_video_ids=[],
            )
        
        self._logger.info(
            f"Model {model_name}: {success_count}/{len(predictions)} successful "
            f"in {elapsed:.2f}s"
        )
        
        # Log errors with context (Property 21)
        for pred in predictions:
            if not pred.success:
                self._logger.error(
                    f"Prediction failed - model: {model_name}, "
                    f"video_id: {pred.video_id}, "
                    f"error: {pred.error_message}"
                )
        
        return prediction_set
    
    def calculate_metrics(
        self,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, ModelEvaluationResult]:
        """
        Calculate evaluation metrics for specified models.
        
        Args:
            model_names: Optional list of model names to evaluate (default: all with predictions)
            
        Returns:
            Dictionary mapping model names to their evaluation results
            
        Raises:
            RuntimeError: If ground truth not loaded or no predictions available
        """
        if self._ground_truth is None:
            raise RuntimeError("Ground truth not loaded. Call load_ground_truth() first.")
        
        if self._prediction_storage is None:
            raise RuntimeError("No predictions available. Call run_predictions() first.")
        
        # Determine which models to evaluate
        available_models = self._prediction_storage.get_all_model_names()
        models_to_evaluate = model_names or available_models
        
        results = {}
        
        for model_name in models_to_evaluate:
            if model_name not in available_models:
                self._logger.warning(f"No predictions found for model: {model_name}")
                continue
            
            try:
                prediction_set = self._prediction_storage.get_predictions(model_name)
                result = self._calculate_metrics_for_model(prediction_set)
                results[model_name] = result
                
                self._logger.info(
                    f"Metrics calculated for {model_name}: "
                    f"macro_f1={result.endorsed_aggregate.macro_f1:.4f}"
                )
            
            except Exception as e:
                self._logger.error(
                    f"Failed to calculate metrics for {model_name}: {e}",
                    exc_info=True
                )
        
        return results
    
    def _calculate_metrics_for_model(
        self,
        prediction_set: PredictionSet,
    ) -> ModelEvaluationResult:
        """
        Calculate metrics for a single model's predictions.
        
        Args:
            prediction_set: The predictions to evaluate
            
        Returns:
            ModelEvaluationResult with all metrics
        """
        calculator = MetricsCalculator(
            ground_truth=self._ground_truth,
            min_frequency_threshold=self._config.min_frequency_threshold,
        )
        
        return calculator.calculate_model_metrics(prediction_set)
    
    def generate_reports(
        self,
        results: Dict[str, ModelEvaluationResult],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Generate reports for the evaluation results.
        
        Args:
            results: Dictionary of model evaluation results
            timestamp: Optional timestamp for filenames
            
        Returns:
            Dictionary mapping report type to file path
        """
        if not results:
            self._logger.warning("No results to generate reports for")
            return {}
        
        generator = ReportGenerator(self._config.output_dir)
        
        # Convert dict to list for report generator
        result_list = list(results.values())
        
        generated = generator.generate_all_reports(result_list, timestamp)
        
        self._logger.info(f"Generated {len(generated)} reports in {self._config.output_dir}")
        
        return generated
    
    def run(self) -> EvaluationSummary:
        """
        Run the complete evaluation workflow.
        
        Executes:
        1. Load ground truth
        2. Initialize adapters
        3. Run predictions
        4. Calculate metrics
        5. Generate reports
        
        Returns:
            EvaluationSummary with complete results and statistics
        """
        start_time = time.time()
        
        self._logger.info("Starting evaluation workflow")
        
        # Step 1: Load ground truth
        self.load_ground_truth()
        
        # Step 2: Initialize adapters
        init_results = self.initialize_adapters()
        
        if not self._adapters:
            self._logger.error("No adapters initialized successfully. Aborting.")
            return EvaluationSummary(
                total_models=len(self._config.models),
                successful_models=0,
                failed_models=len(self._config.models),
                total_videos=self._ground_truth.valid_count if self._ground_truth else 0,
                total_predictions=0,
                successful_predictions=0,
                failed_predictions=0,
                elapsed_time=time.time() - start_time,
                model_errors=self._adapter_errors,
            )
        
        # Step 3: Run predictions
        self.run_predictions()
        
        # Step 4: Calculate metrics
        model_results = self.calculate_metrics()
        
        # Step 5: Generate reports
        generated_reports = self.generate_reports(model_results)
        
        # Compile summary
        total_predictions = 0
        successful_predictions = 0
        failed_predictions = 0
        
        for model_name in self._prediction_storage.get_all_model_names():
            pred_set = self._prediction_storage.get_predictions(model_name)
            if pred_set:
                total_predictions += pred_set.total_count
                successful_predictions += pred_set.success_count
                failed_predictions += pred_set.failure_count
        
        elapsed_time = time.time() - start_time
        
        summary = EvaluationSummary(
            total_models=len(self._config.models),
            successful_models=len(self._adapters),
            failed_models=len(self._config.models) - len(self._adapters),
            total_videos=self._ground_truth.valid_count,
            total_predictions=total_predictions,
            successful_predictions=successful_predictions,
            failed_predictions=failed_predictions,
            elapsed_time=elapsed_time,
            model_results=model_results,
            model_errors=self._adapter_errors,
            generated_reports=generated_reports,
        )
        
        self._logger.info(
            f"Evaluation complete in {elapsed_time:.2f}s. "
            f"Models: {summary.successful_models}/{summary.total_models}, "
            f"Predictions: {summary.successful_predictions}/{summary.total_predictions}"
        )
        
        return summary
    
    def get_success_rates(self) -> Dict[str, float]:
        """
        Get prediction success rates for all models.
        
        Returns:
            Dictionary mapping model names to their success rates (0.0 to 1.0)
        """
        if self._prediction_storage is None:
            return {}
        
        rates = {}
        for model_name in self._prediction_storage.get_all_model_names():
            pred_set = self._prediction_storage.get_predictions(model_name)
            if pred_set and pred_set.total_count > 0:
                rates[model_name] = pred_set.success_count / pred_set.total_count
            else:
                rates[model_name] = 0.0
        
        return rates
    
    def get_prediction_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get prediction success/failure counts for all models.
        
        Returns:
            Dictionary mapping model names to {"success": N, "failure": M, "total": T}
        """
        if self._prediction_storage is None:
            return {}
        
        counts = {}
        for model_name in self._prediction_storage.get_all_model_names():
            pred_set = self._prediction_storage.get_predictions(model_name)
            if pred_set:
                counts[model_name] = {
                    "success": pred_set.success_count,
                    "failure": pred_set.failure_count,
                    "total": pred_set.total_count,
                }
            else:
                counts[model_name] = {
                    "success": 0,
                    "failure": 0,
                    "total": 0,
                }
        
        return counts
