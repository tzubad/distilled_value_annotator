# Tests for EvaluationOrchestrator

import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, MagicMock, patch
from hypothesis import given, strategies as st, assume, settings

from evaluation.orchestrator import (
    EvaluationOrchestrator,
    EvaluationSummary,
    ModelInitializationResult,
)
from evaluation.config_loader import EvaluationConfigLoader, ConfigValidationError
from evaluation.models import (
    EvaluationConfig,
    ModelConfig,
    VideoAnnotation,
    PredictionResult,
    PredictionSet,
    GroundTruthDataset,
)
from evaluation.adapters.base import ModelAdapter
from evaluation.metrics.calculator import ModelEvaluationResult, CategoryResult, AggregateResult
from evaluation.ground_truth_loader import ANNOTATION_CATEGORIES


# ==============================================================================
# Test Fixtures
# ==============================================================================

class MockAdapter(ModelAdapter):
    """Mock adapter for testing."""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._should_fail_init = config.get('fail_init', False)
        self._should_fail_predict = config.get('fail_predict', False)
        self._fail_video_ids = config.get('fail_video_ids', [])
    
    def initialize(self) -> bool:
        return not self._should_fail_init
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        if self._should_fail_predict:
            return PredictionResult(
                video_id=video.video_id,
                predictions={},
                success=False,
                error_message="Mock failure"
            )
        
        if video.video_id in self._fail_video_ids:
            return PredictionResult(
                video_id=video.video_id,
                predictions={},
                success=False,
                error_message=f"Simulated failure for {video.video_id}"
            )
        
        # Generate mock predictions
        predictions = {cat: 0 for cat in ANNOTATION_CATEGORIES}
        return PredictionResult(
            video_id=video.video_id,
            predictions=predictions,
            success=True
        )
    
    def get_model_type(self) -> str:
        return "mock"
    
    def get_model_name(self) -> str:
        return self.model_name


class FailingAdapter(ModelAdapter):
    """Adapter that always fails initialization."""
    
    def initialize(self) -> bool:
        return False
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        return None
    
    def get_model_type(self) -> str:
        return "failing"
    
    def get_model_name(self) -> str:
        return self.model_name


class ExceptionAdapter(ModelAdapter):
    """Adapter that throws exception during initialization."""
    
    def initialize(self) -> bool:
        raise RuntimeError("Initialization explosion")
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        return None
    
    def get_model_type(self) -> str:
        return "exception"
    
    def get_model_name(self) -> str:
        return self.model_name


@pytest.fixture
def sample_ground_truth():
    """Create a sample ground truth dataset."""
    videos = []
    for i in range(10):
        annotations = {cat: (i % 4) - 1 for cat in ANNOTATION_CATEGORIES}  # -1, 0, 1, 2
        video = VideoAnnotation(
            video_id=f"video_{i}",
            video_uri=f"gs://bucket/video_{i}.mp4",
            script_uri=f"gs://bucket/scripts/video_{i}.txt",
            annotations=annotations,
            has_sound=True,
        )
        videos.append(video)
    
    return GroundTruthDataset(
        videos=videos,
        total_count=len(videos),
        valid_count=len(videos),
    )


@pytest.fixture
def basic_config(tmp_path):
    """Create a basic evaluation config."""
    return EvaluationConfig(
        ground_truth_path=str(tmp_path / "ground_truth.csv"),
        scripts_path=str(tmp_path / "scripts"),
        output_dir=str(tmp_path / "output"),
        models=[
            ModelConfig(
                model_type="mock",
                model_name="test_model",
                adapter_class="MockAdapter",
                config={},
            )
        ],
    )


@pytest.fixture
def multi_model_config(tmp_path):
    """Create config with multiple models."""
    return EvaluationConfig(
        ground_truth_path=str(tmp_path / "ground_truth.csv"),
        scripts_path=str(tmp_path / "scripts"),
        output_dir=str(tmp_path / "output"),
        models=[
            ModelConfig(
                model_type="mock",
                model_name="model_a",
                adapter_class="MockAdapter",
                config={},
            ),
            ModelConfig(
                model_type="mock",
                model_name="model_b",
                adapter_class="MockAdapter",
                config={},
            ),
            ModelConfig(
                model_type="mock",
                model_name="model_c",
                adapter_class="MockAdapter",
                config={},
            ),
        ],
    )


# ==============================================================================
# Test Class: Initialization
# ==============================================================================

class TestEvaluationOrchestratorInitialization:
    """Tests for evaluation orchestrator initialization."""
    
    def test_init_with_config(self, basic_config):
        """Test initialization with EvaluationConfig object."""
        orchestrator = EvaluationOrchestrator(config=basic_config)
        
        assert orchestrator.config == basic_config
        assert len(orchestrator.config.models) == 1
    
    def test_init_with_config_path(self, basic_config, tmp_path):
        """Test initialization with config file path."""
        # Save config to file
        config_path = tmp_path / "config.yaml"
        config_yaml = EvaluationConfigLoader.to_yaml(basic_config)
        config_path.write_text(config_yaml)
        
        orchestrator = EvaluationOrchestrator(config_path=str(config_path))
        
        assert orchestrator.config.ground_truth_path == basic_config.ground_truth_path
    
    def test_init_requires_config_or_path(self):
        """Test that ValueError is raised if neither config nor path provided."""
        with pytest.raises(ValueError, match="Either config or config_path"):
            EvaluationOrchestrator()
    
    def test_config_takes_precedence(self, basic_config, tmp_path):
        """Test that config parameter takes precedence over config_path."""
        # Create different config in file
        other_config = EvaluationConfig(
            ground_truth_path="/other/path.csv",
            scripts_path="/other/scripts",
            output_dir="/other/output",
            models=[
                ModelConfig(
                    model_type="other",
                    model_name="other_model",
                    adapter_class="OtherAdapter",
                    config={},
                )
            ],
        )
        config_path = tmp_path / "other_config.yaml"
        config_path.write_text(EvaluationConfigLoader.to_yaml(other_config))
        
        # Both provided - config should take precedence
        orchestrator = EvaluationOrchestrator(
            config=basic_config,
            config_path=str(config_path),
        )
        
        assert orchestrator.config.ground_truth_path == basic_config.ground_truth_path


# ==============================================================================
# Test Class: Adapter Registration and Initialization
# ==============================================================================

class TestAdapterManagement:
    """Tests for adapter registration and initialization."""
    
    def test_register_adapter(self, basic_config):
        """Test registering an adapter class."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        registry = EvaluationOrchestrator.get_registered_adapters()
        assert "MockAdapter" in registry
        assert registry["MockAdapter"] == MockAdapter
    
    def test_initialize_adapters_success(self, basic_config):
        """Test successful adapter initialization."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        orchestrator = EvaluationOrchestrator(config=basic_config)
        results = orchestrator.initialize_adapters()
        
        assert len(results) == 1
        assert results[0].success
        assert results[0].model_name == "test_model"
        assert "test_model" in orchestrator.adapters
    
    def test_initialize_adapters_failure_isolated(self, tmp_path):
        """Test that one adapter failure doesn't affect others."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        EvaluationOrchestrator.register_adapter("FailingAdapter", FailingAdapter)
        
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="good_model",
                    adapter_class="MockAdapter",
                    config={},
                ),
                ModelConfig(
                    model_type="failing",
                    model_name="bad_model",
                    adapter_class="FailingAdapter",
                    config={},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        results = orchestrator.initialize_adapters()
        
        # One should succeed, one should fail
        success_count = sum(1 for r in results if r.success)
        assert success_count == 1
        
        # Good model should be available
        assert "good_model" in orchestrator.adapters
        
        # Bad model should be in errors
        assert "bad_model" in orchestrator.adapter_errors
    
    def test_initialize_adapter_exception_handled(self, tmp_path):
        """Test that adapter initialization exceptions are caught."""
        EvaluationOrchestrator.register_adapter("ExceptionAdapter", ExceptionAdapter)
        
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="exception",
                    model_name="exploding_model",
                    adapter_class="ExceptionAdapter",
                    config={},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        results = orchestrator.initialize_adapters()
        
        assert len(results) == 1
        assert not results[0].success
        assert "exploding_model" in orchestrator.adapter_errors


# ==============================================================================
# Test Class: Prediction Execution
# ==============================================================================

class TestPredictionExecution:
    """Tests for running predictions."""
    
    def test_run_predictions_success(self, basic_config, sample_ground_truth):
        """Test successful prediction execution."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        orchestrator = EvaluationOrchestrator(config=basic_config)
        orchestrator._ground_truth = sample_ground_truth
        orchestrator.initialize_adapters()
        
        storage = orchestrator.run_predictions()
        
        assert "test_model" in storage.get_all_model_names()
        pred_set = storage.get_predictions("test_model")
        assert pred_set.total_count == len(sample_ground_truth.videos)
        assert pred_set.success_count == len(sample_ground_truth.videos)
    
    def test_run_predictions_partial_failure(self, tmp_path, sample_ground_truth):
        """Test predictions with some videos failing."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        # Configure adapter to fail for specific videos
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="partial_fail_model",
                    adapter_class="MockAdapter",
                    config={"fail_video_ids": ["video_0", "video_1", "video_2"]},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        orchestrator._ground_truth = sample_ground_truth
        orchestrator.initialize_adapters()
        
        storage = orchestrator.run_predictions()
        
        pred_set = storage.get_predictions("partial_fail_model")
        assert pred_set.failure_count == 3
        assert pred_set.success_count == 7
        assert "video_0" in pred_set.failed_video_ids
    
    def test_run_predictions_requires_ground_truth(self, basic_config):
        """Test that predictions require ground truth to be loaded."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        orchestrator = EvaluationOrchestrator(config=basic_config)
        orchestrator.initialize_adapters()
        
        with pytest.raises(RuntimeError, match="Ground truth not loaded"):
            orchestrator.run_predictions()
    
    def test_run_predictions_requires_adapters(self, basic_config, sample_ground_truth):
        """Test that predictions require adapters to be initialized."""
        orchestrator = EvaluationOrchestrator(config=basic_config)
        orchestrator._ground_truth = sample_ground_truth
        
        with pytest.raises(RuntimeError, match="No adapters initialized"):
            orchestrator.run_predictions()


# ==============================================================================
# Test Class: Success Rate Reporting
# ==============================================================================

class TestSuccessRateReporting:
    """Tests for success rate reporting."""
    
    def test_get_success_rates(self, tmp_path, sample_ground_truth):
        """Test getting success rates for all models."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="model_100",
                    adapter_class="MockAdapter",
                    config={},
                ),
                ModelConfig(
                    model_type="mock",
                    model_name="model_70",
                    adapter_class="MockAdapter",
                    config={"fail_video_ids": ["video_0", "video_1", "video_2"]},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        orchestrator._ground_truth = sample_ground_truth
        orchestrator.initialize_adapters()
        orchestrator.run_predictions()
        
        rates = orchestrator.get_success_rates()
        
        assert rates["model_100"] == 1.0
        assert rates["model_70"] == 0.7
    
    def test_get_prediction_counts(self, tmp_path, sample_ground_truth):
        """Test getting prediction counts."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="count_model",
                    adapter_class="MockAdapter",
                    config={"fail_video_ids": ["video_0", "video_1"]},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        orchestrator._ground_truth = sample_ground_truth
        orchestrator.initialize_adapters()
        orchestrator.run_predictions()
        
        counts = orchestrator.get_prediction_counts()
        
        assert counts["count_model"]["total"] == 10
        assert counts["count_model"]["success"] == 8
        assert counts["count_model"]["failure"] == 2
    
    def test_success_rates_empty_before_predictions(self, basic_config):
        """Test that success rates are empty before predictions."""
        orchestrator = EvaluationOrchestrator(config=basic_config)
        
        rates = orchestrator.get_success_rates()
        assert rates == {}


# ==============================================================================
# Property Tests
# ==============================================================================

class TestModelFailureIsolationProperty:
    """
    Property 22: Model Failure Isolation
    
    Validates: Requirements 10.3
    Verify: Failed model doesn't stop evaluation
    """
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=10)
    def test_failed_models_dont_stop_others(self, num_good_models):
        """
        Property 22: A failed model initialization should not prevent
        other models from being initialized and run.
        """
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        EvaluationOrchestrator.register_adapter("FailingAdapter", FailingAdapter)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create config with mix of good and failing models
            models = []
            
            # Add good models
            for i in range(num_good_models):
                models.append(ModelConfig(
                    model_type="mock",
                    model_name=f"good_model_{i}",
                    adapter_class="MockAdapter",
                    config={},
                ))
            
            # Add a failing model
            models.append(ModelConfig(
                model_type="failing",
                model_name="failing_model",
                adapter_class="FailingAdapter",
                config={},
            ))
            
            config = EvaluationConfig(
                ground_truth_path=f"{tmp_dir}/gt.csv",
                scripts_path=f"{tmp_dir}/scripts",
                output_dir=f"{tmp_dir}/output",
                models=models,
            )
            
            orchestrator = EvaluationOrchestrator(config=config)
            results = orchestrator.initialize_adapters()
            
            # All good models should succeed
            success_count = sum(1 for r in results if r.success)
            assert success_count == num_good_models
            
            # Good models should be available
            assert len(orchestrator.adapters) == num_good_models
            
            # Failing model should be in errors
            assert "failing_model" in orchestrator.adapter_errors


class TestVideoProcessingCompletenessProperty:
    """
    Property 1: Video Processing Completeness
    
    Validates: Requirements 1.1
    Verify: N videos → N prediction attempts
    """
    
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=10)
    def test_all_videos_get_prediction_attempt(self, num_videos):
        """
        Property 1: Every video in the ground truth should get
        a prediction attempt (successful or not).
        """
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        # Create ground truth with N videos
        videos = []
        for i in range(num_videos):
            annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=videos,
            total_count=num_videos,
            valid_count=num_videos,
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvaluationConfig(
                ground_truth_path=f"{tmp_dir}/gt.csv",
                scripts_path=f"{tmp_dir}/scripts",
                output_dir=f"{tmp_dir}/output",
                models=[ModelConfig(
                    model_type="mock",
                    model_name="test_model",
                    adapter_class="MockAdapter",
                    config={},
                )],
            )
            
            orchestrator = EvaluationOrchestrator(config=config)
            orchestrator._ground_truth = ground_truth
            orchestrator.initialize_adapters()
            storage = orchestrator.run_predictions()
            
            pred_set = storage.get_predictions("test_model")
            
            # Total predictions should equal number of videos
            assert pred_set.total_count == num_videos


class TestErrorResilienceProperty:
    """
    Property 5: Error Resilience
    
    Validates: Requirements 1.5, 10.2
    Verify: One error doesn't stop processing
    """
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=10)
    def test_processing_continues_after_failures(self, num_failures):
        """
        Property 5: Processing should continue even when some predictions fail.
        """
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        total_videos = 10
        fail_ids = [f"video_{i}" for i in range(min(num_failures, total_videos))]
        
        # Create ground truth
        videos = []
        for i in range(total_videos):
            annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=videos,
            total_count=total_videos,
            valid_count=total_videos,
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvaluationConfig(
                ground_truth_path=f"{tmp_dir}/gt.csv",
                scripts_path=f"{tmp_dir}/scripts",
                output_dir=f"{tmp_dir}/output",
                models=[ModelConfig(
                    model_type="mock",
                    model_name="resilient_model",
                    adapter_class="MockAdapter",
                    config={"fail_video_ids": fail_ids},
                )],
            )
            
            orchestrator = EvaluationOrchestrator(config=config)
            orchestrator._ground_truth = ground_truth
            orchestrator.initialize_adapters()
            storage = orchestrator.run_predictions()
            
            pred_set = storage.get_predictions("resilient_model")
            
            # All videos should have been attempted
            assert pred_set.total_count == total_videos
            
            # Correct number of failures
            assert pred_set.failure_count == len(fail_ids)
            
            # Correct number of successes (processing continued)
            expected_successes = total_videos - len(fail_ids)
            assert pred_set.success_count == expected_successes


class TestErrorLoggingContextProperty:
    """
    Property 21: Error Logging with Context
    
    Validates: Requirements 10.1
    Verify: Errors logged with video_id and model_name
    """
    
    def test_errors_include_context(self, tmp_path, sample_ground_truth, caplog):
        """
        Property 21: Error messages should include video_id and model_name
        for debugging purposes.
        """
        import logging
        caplog.set_level(logging.ERROR)
        
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        fail_video_ids = ["video_0", "video_1"]
        
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="contextual_model",
                    adapter_class="MockAdapter",
                    config={"fail_video_ids": fail_video_ids},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        orchestrator._ground_truth = sample_ground_truth
        orchestrator.initialize_adapters()
        orchestrator.run_predictions()
        
        # Check that error logs contain context
        error_logs = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
        
        for video_id in fail_video_ids:
            # At least one log should mention this video
            has_video_context = any(video_id in log for log in error_logs)
            assert has_video_context, f"Expected {video_id} in error logs"
        
        # Check that model name is in logs
        has_model_context = any("contextual_model" in log for log in error_logs)
        assert has_model_context, "Expected model_name in error logs"


class TestSuccessRateReportingProperty:
    """
    Property 23: Success Rate Reporting
    
    Validates: Requirements 10.4
    Verify: Success/failure counts reported
    """
    
    @given(
        st.integers(min_value=0, max_value=10),  # num_failures
    )
    @settings(max_examples=10)
    def test_success_failure_counts_accurate(self, num_failures):
        """
        Property 23: Success and failure counts should be accurately reported.
        """
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        total_videos = 10
        actual_failures = min(num_failures, total_videos)
        fail_ids = [f"video_{i}" for i in range(actual_failures)]
        
        # Create ground truth
        videos = []
        for i in range(total_videos):
            annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=videos,
            total_count=total_videos,
            valid_count=total_videos,
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvaluationConfig(
                ground_truth_path=f"{tmp_dir}/gt.csv",
                scripts_path=f"{tmp_dir}/scripts",
                output_dir=f"{tmp_dir}/output",
                models=[ModelConfig(
                    model_type="mock",
                    model_name="counting_model",
                    adapter_class="MockAdapter",
                    config={"fail_video_ids": fail_ids},
                )],
            )
            
            orchestrator = EvaluationOrchestrator(config=config)
            orchestrator._ground_truth = ground_truth
            orchestrator.initialize_adapters()
            orchestrator.run_predictions()
            
            counts = orchestrator.get_prediction_counts()
            rates = orchestrator.get_success_rates()
            
            # Verify counts
            assert counts["counting_model"]["total"] == total_videos
            assert counts["counting_model"]["failure"] == actual_failures
            assert counts["counting_model"]["success"] == total_videos - actual_failures
            
            # Verify rate
            expected_rate = (total_videos - actual_failures) / total_videos
            assert abs(rates["counting_model"] - expected_rate) < 0.001


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEvaluationOrchestratorEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_ground_truth(self, basic_config):
        """Test handling of empty ground truth dataset."""
        EvaluationOrchestrator.register_adapter("MockAdapter", MockAdapter)
        
        empty_gt = GroundTruthDataset(videos=[], total_count=0, valid_count=0)
        
        orchestrator = EvaluationOrchestrator(config=basic_config)
        orchestrator._ground_truth = empty_gt
        orchestrator.initialize_adapters()
        
        storage = orchestrator.run_predictions()
        
        # With empty ground truth, no predictions are stored (storage rejects empty lists)
        # The model should not be in storage
        pred_set = storage.get_predictions("test_model")
        # When there are no videos, either no predictions are stored (None)
        # or an empty prediction set is stored
        if pred_set is None:
            # This is acceptable - no predictions were made
            pass
        else:
            assert pred_set.total_count == 0
    
    def test_all_adapters_fail(self, tmp_path):
        """Test handling when all adapters fail to initialize."""
        EvaluationOrchestrator.register_adapter("FailingAdapter", FailingAdapter)
        
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="failing",
                    model_name="fail1",
                    adapter_class="FailingAdapter",
                    config={},
                ),
                ModelConfig(
                    model_type="failing",
                    model_name="fail2",
                    adapter_class="FailingAdapter",
                    config={},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        orchestrator.initialize_adapters()
        
        assert len(orchestrator.adapters) == 0
        assert len(orchestrator.adapter_errors) == 2
    
    def test_unregistered_adapter_class(self, tmp_path):
        """Test handling of unregistered adapter class."""
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "gt.csv"),
            scripts_path=str(tmp_path / "scripts"),
            output_dir=str(tmp_path / "output"),
            models=[
                ModelConfig(
                    model_type="unknown",
                    model_name="mystery_model",
                    adapter_class="NonExistentAdapter",
                    config={},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        results = orchestrator.initialize_adapters()
        
        assert len(results) == 1
        assert not results[0].success
        assert "mystery_model" in orchestrator.adapter_errors
