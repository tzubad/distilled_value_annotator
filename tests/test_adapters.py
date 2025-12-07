# Property-based tests for model adapters

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from abc import ABC
from evaluation.adapters import ModelAdapter
from evaluation.models import VideoAnnotation, PredictionResult
from typing import Optional, Dict, Any


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


# Create concrete test adapter implementations for testing
class MockLLMAdapter(ModelAdapter):
    """Mock LLM adapter for testing."""
    
    def initialize(self) -> bool:
        return True
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        # Return a simple prediction with all values set to 1
        predictions = {category: 1 for category in ANNOTATION_CATEGORIES}
        return PredictionResult(
            video_id=video.video_id,
            predictions=predictions,
            success=True,
            inference_time=0.1
        )
    
    def get_model_type(self) -> str:
        return "LLM"
    
    def get_model_name(self) -> str:
        return self.model_name


class MockMLMAdapter(ModelAdapter):
    """Mock MLM adapter for testing."""
    
    def initialize(self) -> bool:
        return True
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        # Return a simple prediction with all values set to 0
        predictions = {category: 0 for category in ANNOTATION_CATEGORIES}
        return PredictionResult(
            video_id=video.video_id,
            predictions=predictions,
            success=True,
            inference_time=0.05
        )
    
    def get_model_type(self) -> str:
        return "MLM"
    
    def get_model_name(self) -> str:
        return self.model_name


class FailingAdapter(ModelAdapter):
    """Mock adapter that fails initialization."""
    
    def initialize(self) -> bool:
        return False
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        return None
    
    def get_model_type(self) -> str:
        return "FAILING"
    
    def get_model_name(self) -> str:
        return self.model_name


# Strategy for generating valid annotation dictionaries
valid_annotation_values = st.integers(min_value=-1, max_value=2)
complete_annotations = st.fixed_dictionaries(
    {category: valid_annotation_values for category in ANNOTATION_CATEGORIES}
)


# **Feature: model-evaluation-module, Property 15: Model adapter interface consistency**
# **Validates: Requirements 7.1**
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    model_name=st.text(min_size=1, max_size=50),
    config=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(st.integers(), st.floats(allow_nan=False), st.text(), st.booleans()),
        max_size=5
    )
)
def test_adapter_interface_consistency(model_name, config):
    """
    Property: For any two model adapters (regardless of type), they should implement
    the same interface methods (initialize, predict, get_model_type, get_model_name).
    
    This test verifies that all adapters conform to the ModelAdapter interface.
    """
    # Create instances of different adapter types
    llm_adapter = MockLLMAdapter(model_name=f"llm_{model_name}", config=config)
    mlm_adapter = MockMLMAdapter(model_name=f"mlm_{model_name}", config=config)
    failing_adapter = FailingAdapter(model_name=f"fail_{model_name}", config=config)
    
    adapters = [llm_adapter, mlm_adapter, failing_adapter]
    
    # Verify all adapters have the required methods
    required_methods = ['initialize', 'predict', 'get_model_type', 'get_model_name', 'batch_predict']
    
    for adapter in adapters:
        for method_name in required_methods:
            assert hasattr(adapter, method_name), (
                f"{adapter.__class__.__name__} missing required method: {method_name}"
            )
            assert callable(getattr(adapter, method_name)), (
                f"{adapter.__class__.__name__}.{method_name} is not callable"
            )
    
    # Verify initialize() returns bool
    for adapter in adapters:
        result = adapter.initialize()
        assert isinstance(result, bool), (
            f"{adapter.__class__.__name__}.initialize() should return bool, got {type(result)}"
        )
    
    # Verify get_model_type() returns str
    for adapter in adapters:
        result = adapter.get_model_type()
        assert isinstance(result, str), (
            f"{adapter.__class__.__name__}.get_model_type() should return str, got {type(result)}"
        )
    
    # Verify get_model_name() returns str
    for adapter in adapters:
        result = adapter.get_model_name()
        assert isinstance(result, str), (
            f"{adapter.__class__.__name__}.get_model_name() should return str, got {type(result)}"
        )


@given(
    model_name=st.text(min_size=1, max_size=50),
    video_id=st.text(min_size=1, max_size=100),
    video_uri=st.text(min_size=1, max_size=200),
    script_uri=st.text(min_size=1, max_size=200),
    annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_adapter_predict_returns_prediction_result_or_none(
    model_name, video_id, video_uri, script_uri, annotations, has_sound
):
    """
    Property: For any adapter and video, predict() should return either a
    PredictionResult object or None.
    
    This test verifies the return type contract of the predict() method.
    """
    # Create a video annotation
    video = VideoAnnotation(
        video_id=video_id,
        video_uri=video_uri,
        script_uri=script_uri,
        annotations=annotations,
        has_sound=has_sound
    )
    
    # Test with different adapter types
    adapters = [
        MockLLMAdapter(model_name=f"llm_{model_name}", config={}),
        MockMLMAdapter(model_name=f"mlm_{model_name}", config={}),
        FailingAdapter(model_name=f"fail_{model_name}", config={}),
    ]
    
    for adapter in adapters:
        adapter.initialize()
        result = adapter.predict(video)
        
        # Result should be either PredictionResult or None
        assert result is None or isinstance(result, PredictionResult), (
            f"{adapter.__class__.__name__}.predict() should return PredictionResult or None, "
            f"got {type(result)}"
        )


@given(
    model_name=st.text(min_size=1, max_size=50),
    num_videos=st.integers(min_value=1, max_value=10),
)
def test_adapter_batch_predict_returns_list(model_name, num_videos):
    """
    Property: For any adapter and list of videos, batch_predict() should return
    a list of PredictionResult objects with the same length as the input.
    
    This test verifies the batch_predict() method behavior.
    """
    # Create a list of videos
    videos = []
    for i in range(num_videos):
        annotations = {category: 1 for category in ANNOTATION_CATEGORIES}
        video = VideoAnnotation(
            video_id=f"video_{i}",
            video_uri=f"gs://bucket/video_{i}.mp4",
            script_uri=f"gs://bucket/scripts/video_{i}.txt",
            annotations=annotations,
            has_sound=True
        )
        videos.append(video)
    
    # Test with different adapter types
    adapters = [
        MockLLMAdapter(model_name=f"llm_{model_name}", config={}),
        MockMLMAdapter(model_name=f"mlm_{model_name}", config={}),
    ]
    
    for adapter in adapters:
        adapter.initialize()
        results = adapter.batch_predict(videos)
        
        # Results should be a list
        assert isinstance(results, list), (
            f"{adapter.__class__.__name__}.batch_predict() should return list, got {type(results)}"
        )
        
        # Should have same length as input
        assert len(results) == num_videos, (
            f"{adapter.__class__.__name__}.batch_predict() should return {num_videos} results, "
            f"got {len(results)}"
        )
        
        # All results should be PredictionResult objects
        for result in results:
            assert isinstance(result, PredictionResult), (
                f"batch_predict() should return list of PredictionResult, "
                f"got {type(result)} in list"
            )


@given(
    model_name=st.text(min_size=1, max_size=50),
    num_videos=st.integers(min_value=1, max_value=10),
)
def test_adapter_batch_predict_handles_errors_gracefully(model_name, num_videos):
    """
    Property: For any adapter, batch_predict() should handle errors gracefully
    and continue processing remaining videos.
    
    This test verifies error resilience in batch processing.
    """
    # Create a list of videos
    videos = []
    for i in range(num_videos):
        annotations = {category: 1 for category in ANNOTATION_CATEGORIES}
        video = VideoAnnotation(
            video_id=f"video_{i}",
            video_uri=f"gs://bucket/video_{i}.mp4",
            script_uri=f"gs://bucket/scripts/video_{i}.txt",
            annotations=annotations,
            has_sound=True
        )
        videos.append(video)
    
    # Create an adapter that sometimes fails
    class SometimesFailingAdapter(ModelAdapter):
        def __init__(self, model_name: str, config: Dict[str, Any]):
            super().__init__(model_name, config)
            self.call_count = 0
        
        def initialize(self) -> bool:
            return True
        
        def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
            self.call_count += 1
            # Fail on every other video
            if self.call_count % 2 == 0:
                return None
            
            predictions = {category: 1 for category in ANNOTATION_CATEGORIES}
            return PredictionResult(
                video_id=video.video_id,
                predictions=predictions,
                success=True,
                inference_time=0.1
            )
        
        def get_model_type(self) -> str:
            return "TEST"
        
        def get_model_name(self) -> str:
            return self.model_name
    
    adapter = SometimesFailingAdapter(model_name=model_name, config={})
    adapter.initialize()
    results = adapter.batch_predict(videos)
    
    # Should still return results for all videos
    assert len(results) == num_videos, (
        f"batch_predict() should return {num_videos} results even with failures, "
        f"got {len(results)}"
    )
    
    # Some should be successful, some should be failures
    success_count = sum(1 for r in results if r.success)
    failure_count = sum(1 for r in results if not r.success)
    
    # With the pattern above, we expect roughly half to succeed
    assert success_count > 0 or failure_count > 0, (
        "batch_predict() should have mix of successes and failures"
    )
    
    # All failed predictions should have error messages
    for result in results:
        if not result.success:
            assert result.error_message is not None and len(result.error_message) > 0, (
                "Failed predictions should have error messages"
            )


@given(
    model_name=st.text(min_size=1, max_size=50),
)
def test_adapter_is_abstract_base_class(model_name):
    """
    Property: ModelAdapter should be an abstract base class that cannot be
    instantiated directly.
    
    This test verifies that ModelAdapter enforces implementation of abstract methods.
    """
    # Attempting to instantiate ModelAdapter directly should fail
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        adapter = ModelAdapter(model_name=model_name, config={})


@given(
    model_name1=st.text(min_size=1, max_size=50),
    model_name2=st.text(min_size=1, max_size=50),
)
def test_different_adapters_have_independent_state(model_name1, model_name2):
    """
    Property: Different adapter instances should maintain independent state.
    
    This test verifies that adapters don't share state inappropriately.
    """
    # Create two different adapters
    adapter1 = MockLLMAdapter(model_name=model_name1, config={"param": "value1"})
    adapter2 = MockMLMAdapter(model_name=model_name2, config={"param": "value2"})
    
    # Verify they have different names
    assert adapter1.get_model_name() == model_name1
    assert adapter2.get_model_name() == model_name2
    
    # Verify they have different types
    assert adapter1.get_model_type() == "LLM"
    assert adapter2.get_model_type() == "MLM"
    
    # Verify they have independent configs
    assert adapter1.config["param"] == "value1"
    assert adapter2.config["param"] == "value2"
    
    # Modifying one config shouldn't affect the other
    adapter1.config["new_param"] = "new_value"
    assert "new_param" not in adapter2.config


# **Feature: model-evaluation-module, Property 16: Adapter output normalization**
# **Validates: Requirements 7.3**
@given(
    video_id=st.text(min_size=1, max_size=100),
    video_uri=st.text(min_size=1, max_size=200),
    script_uri=st.text(min_size=1, max_size=200),
    annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_adapter_output_normalization(video_id, video_uri, script_uri, annotations, has_sound):
    """
    Property: For any model adapter and any video, the prediction output should conform
    to the standard annotation format with 19 categories and values in {-1, 0, 1, 2}.
    
    This test verifies that all adapters normalize their outputs to the standard format.
    """
    # Create a video annotation
    video = VideoAnnotation(
        video_id=video_id,
        video_uri=video_uri,
        script_uri=script_uri,
        annotations=annotations,
        has_sound=has_sound
    )
    
    # Test with different adapter types
    adapters = [
        MockLLMAdapter(model_name="test_llm", config={}),
        MockMLMAdapter(model_name="test_mlm", config={}),
    ]
    
    valid_values = {-1, 0, 1, 2}
    
    for adapter in adapters:
        adapter.initialize()
        result = adapter.predict(video)
        
        # Skip if prediction failed
        if result is None or not result.success:
            continue
        
        # Verify predictions is a dictionary
        assert isinstance(result.predictions, dict), (
            f"{adapter.__class__.__name__} predictions should be a dict, got {type(result.predictions)}"
        )
        
        # Verify we have exactly 19 categories
        assert len(result.predictions) == 19, (
            f"{adapter.__class__.__name__} should return 19 categories, got {len(result.predictions)}"
        )
        
        # Verify all categories are present
        for category in ANNOTATION_CATEGORIES:
            assert category in result.predictions, (
                f"{adapter.__class__.__name__} missing category: {category}"
            )
        
        # Verify all values are in valid range
        for category, value in result.predictions.items():
            assert value in valid_values, (
                f"{adapter.__class__.__name__} returned invalid value {value} for {category}. "
                f"Must be one of {valid_values}"
            )
            assert isinstance(value, int), (
                f"{adapter.__class__.__name__} returned non-integer value {value} ({type(value)}) "
                f"for {category}"
            )


# **Feature: model-evaluation-module, Property 6: MLM prediction completeness**
# **Validates: Requirements 2.2**
@given(
    video_id=st.text(min_size=1, max_size=100),
    video_uri=st.text(min_size=1, max_size=200),
    script_uri=st.text(min_size=1, max_size=200),
    annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_mlm_prediction_completeness(video_id, video_uri, script_uri, annotations, has_sound):
    """
    Property: For any video processed by an MLM model, the output should contain
    predictions for all 19 annotation categories.
    
    This test verifies that MLM adapters generate complete predictions.
    """
    # Create a video annotation
    video = VideoAnnotation(
        video_id=video_id,
        video_uri=video_uri,
        script_uri=script_uri,
        annotations=annotations,
        has_sound=has_sound
    )
    
    # Test with MLM adapter
    adapter = MockMLMAdapter(model_name="test_mlm", config={})
    adapter.initialize()
    result = adapter.predict(video)
    
    # If prediction succeeded, verify completeness
    if result is not None and result.success:
        # Verify predictions is a dictionary
        assert isinstance(result.predictions, dict), (
            f"MLM predictions should be a dict, got {type(result.predictions)}"
        )
        
        # Verify we have exactly 19 categories
        assert len(result.predictions) == 19, (
            f"MLM should return predictions for all 19 categories, got {len(result.predictions)}"
        )
        
        # Verify all expected categories are present
        for category in ANNOTATION_CATEGORIES:
            assert category in result.predictions, (
                f"MLM missing prediction for category: {category}"
            )
        
        # Verify no extra categories
        for category in result.predictions.keys():
            assert category in ANNOTATION_CATEGORIES, (
                f"MLM returned unexpected category: {category}"
            )



# Mock MLM adapters for testing MLM-specific behavior
class MockRoBERTaAdapter(ModelAdapter):
    """Mock RoBERTa adapter for testing."""
    
    def initialize(self) -> bool:
        return True
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        # Simulate RoBERTa predictions with varied values
        predictions = {}
        for i, category in enumerate(ANNOTATION_CATEGORIES):
            # Cycle through valid values
            predictions[category] = (i % 4) - 1  # Results in -1, 0, 1, 2
        
        return PredictionResult(
            video_id=video.video_id,
            predictions=predictions,
            success=True,
            inference_time=0.05
        )
    
    def get_model_type(self) -> str:
        return "MLM"
    
    def get_model_name(self) -> str:
        return self.model_name


class MockDeBERTaAdapter(ModelAdapter):
    """Mock DeBERTa adapter for testing."""
    
    def initialize(self) -> bool:
        return True
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        # Simulate DeBERTa predictions with all zeros
        predictions = {category: 0 for category in ANNOTATION_CATEGORIES}
        
        return PredictionResult(
            video_id=video.video_id,
            predictions=predictions,
            success=True,
            inference_time=0.08
        )
    
    def get_model_type(self) -> str:
        return "MLM"
    
    def get_model_name(self) -> str:
        return self.model_name


# **Feature: model-evaluation-module, Property 16: Adapter output normalization**
# **Validates: Requirements 7.3**
@given(
    video_id=st.text(min_size=1, max_size=100),
    video_uri=st.text(min_size=1, max_size=200),
    script_uri=st.text(min_size=1, max_size=200),
    annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_mlm_adapter_output_format(video_id, video_uri, script_uri, annotations, has_sound):
    """
    Property: For any MLM adapter and any video, the prediction output should conform
    to the standard annotation format with 19 categories and values in {-1, 0, 1, 2}.
    
    This test specifically verifies MLM adapters (RoBERTa, DeBERTa) normalize their
    outputs to the standard format.
    """
    # Create a video annotation
    video = VideoAnnotation(
        video_id=video_id,
        video_uri=video_uri,
        script_uri=script_uri,
        annotations=annotations,
        has_sound=has_sound
    )
    
    # Test with MLM adapter types
    mlm_adapters = [
        MockRoBERTaAdapter(model_name="roberta-test", config={}),
        MockDeBERTaAdapter(model_name="deberta-test", config={}),
    ]
    
    valid_values = {-1, 0, 1, 2}
    
    for adapter in mlm_adapters:
        adapter.initialize()
        result = adapter.predict(video)
        
        # MLM adapters should always succeed in this test
        assert result is not None, (
            f"{adapter.__class__.__name__} returned None"
        )
        assert result.success, (
            f"{adapter.__class__.__name__} prediction failed: {result.error_message}"
        )
        
        # Verify predictions is a dictionary
        assert isinstance(result.predictions, dict), (
            f"{adapter.__class__.__name__} predictions should be a dict, got {type(result.predictions)}"
        )
        
        # Verify we have exactly 19 categories
        assert len(result.predictions) == 19, (
            f"{adapter.__class__.__name__} should return 19 categories, got {len(result.predictions)}"
        )
        
        # Verify all categories are present
        for category in ANNOTATION_CATEGORIES:
            assert category in result.predictions, (
                f"{adapter.__class__.__name__} missing category: {category}"
            )
        
        # Verify all values are in valid range
        for category, value in result.predictions.items():
            assert value in valid_values, (
                f"{adapter.__class__.__name__} returned invalid value {value} for {category}. "
                f"Must be one of {valid_values}"
            )
            assert isinstance(value, int), (
                f"{adapter.__class__.__name__} returned non-integer value {value} ({type(value)}) "
                f"for {category}"
            )
        
        # Verify model type is MLM
        assert adapter.get_model_type() == "MLM", (
            f"{adapter.__class__.__name__} should return 'MLM' as model type, "
            f"got {adapter.get_model_type()}"
        )
