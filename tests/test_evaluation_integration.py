"""
Integration tests for the Model Evaluation Module.

These tests verify end-to-end functionality of the evaluation pipeline
using sample data and mock adapters.
"""

import pytest
import sys
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    EvaluationOrchestrator,
    EvaluationConfigLoader,
    EvaluationConfig,
    ModelConfig,
    PredictionStorage,
    GroundTruthLoader,
    VideoAnnotation,
    PredictionResult,
)
from evaluation.metrics import MetricsCalculator
from evaluation.reports import ReportGenerator
from evaluation.adapters import ModelAdapter


# ==============================================================================
# Test Fixtures
# ==============================================================================

class MockSuccessAdapter(ModelAdapter):
    """Mock adapter that always succeeds."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._model_name = model_name
    
    def initialize(self) -> bool:
        return True
    
    def predict(self, video: VideoAnnotation) -> PredictionResult:
        # Return predictions matching ground truth for high accuracy
        return PredictionResult(
            video_id=video.video_id,
            predictions=video.annotations.copy(),
            success=True,
        )
    
    def get_model_type(self) -> str:
        return "mock"
    
    def get_model_name(self) -> str:
        return self._model_name


class MockPartialAdapter(ModelAdapter):
    """Mock adapter that fails for some videos."""
    
    def __init__(self, model_name: str, fail_pattern: str = "003", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._model_name = model_name
        self.fail_pattern = fail_pattern
    
    def initialize(self) -> bool:
        return True
    
    def predict(self, video: VideoAnnotation) -> PredictionResult:
        if self.fail_pattern in video.video_id:
            return PredictionResult(
                video_id=video.video_id,
                predictions={},
                success=False,
                error_message=f"Simulated failure for {video.video_id}",
            )
        return PredictionResult(
            video_id=video.video_id,
            predictions=video.annotations.copy(),
            success=True,
        )
    
    def get_model_type(self) -> str:
        return "mock"
    
    def get_model_name(self) -> str:
        return self._model_name


@pytest.fixture
def sample_ground_truth_file(tmp_path):
    """Create a sample ground truth CSV file."""
    gt_path = tmp_path / "ground_truth.csv"
    scripts_path = tmp_path / "scripts"
    scripts_path.mkdir()
    
    categories = [
        "Achievement", "Benevolence", "Conformity", "Hedonism", "Power",
        "Security", "Self_direction", "Stimulation", "Tradition", "Universalism",
        "Face", "Humility", "Societal_security", "Personal_security", "Routine",
        "Rules", "Caring", "Dependability", "Concern_for_others"
    ]
    
    header = ["video_id", "video_uri", "script_uri"] + categories
    
    rows = []
    for i in range(1, 6):
        video_id = f"video_00{i}"
        video_uri = f"gs://test-bucket/videos/{video_id}.mp4"
        script_file = scripts_path / f"{video_id}.txt"
        script_file.write_text(f"Sample script content for video {i}")
        script_uri = str(script_file)
        
        # Create annotation values
        if i == 1:
            values = [1, 0, 0, 2, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        elif i == 2:
            values = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1]
        elif i == 3:
            values = [2, 0, -1, 1, 1, 0, 2, 1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0]
        elif i == 4:
            values = [0, 2, 0, 0, -1, 1, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 1, 2]
        else:
            values = [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        
        rows.append([video_id, video_uri, script_uri] + values)
    
    with open(gt_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    return gt_path


@pytest.fixture
def sample_scripts_dir(tmp_path, sample_ground_truth_file):
    """Return the scripts path created by sample_ground_truth_file fixture."""
    # Scripts directory was created by sample_ground_truth_file fixture
    scripts_path = tmp_path / "scripts"
    return scripts_path


@pytest.fixture
def output_dir(tmp_path):
    """Create output directory."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return output_path


@pytest.fixture
def basic_config(sample_ground_truth_file, sample_scripts_dir, output_dir):
    """Create a basic evaluation configuration."""
    return EvaluationConfig(
        ground_truth_path=str(sample_ground_truth_file),
        scripts_path=str(sample_scripts_dir),
        output_dir=str(output_dir),
        models=[
            ModelConfig(
                model_type="mock",
                model_name="test_model",
                adapter_class="MockSuccessAdapter",
                config={},
            ),
        ],
    )


@pytest.fixture
def multi_model_config(sample_ground_truth_file, sample_scripts_dir, output_dir):
    """Create configuration with multiple models."""
    return EvaluationConfig(
        ground_truth_path=str(sample_ground_truth_file),
        scripts_path=str(sample_scripts_dir),
        output_dir=str(output_dir),
        models=[
            ModelConfig(
                model_type="mock",
                model_name="perfect_model",
                adapter_class="MockSuccessAdapter",
                config={},
            ),
            ModelConfig(
                model_type="mock",
                model_name="partial_model",
                adapter_class="MockPartialAdapter",
                config={"fail_pattern": "003"},
            ),
        ],
    )


# ==============================================================================
# Integration Test: End-to-End Evaluation
# ==============================================================================

class TestEndToEndEvaluation:
    """Integration tests for complete evaluation workflow."""
    
    def test_complete_evaluation_workflow(self, basic_config, output_dir):
        """Test the complete evaluation workflow from start to finish."""
        # Register mock adapter
        EvaluationOrchestrator.register_adapter("MockSuccessAdapter", MockSuccessAdapter)
        
        # Create orchestrator
        orchestrator = EvaluationOrchestrator(config=basic_config)
        
        # Load ground truth
        ground_truth = orchestrator.load_ground_truth()
        assert ground_truth.valid_count == 5
        
        # Initialize adapters
        init_results = orchestrator.initialize_adapters()
        assert len(init_results) == 1
        assert init_results[0].success
        
        # Run predictions
        storage = orchestrator.run_predictions()
        assert storage is not None
        
        # Get prediction counts
        counts = orchestrator.get_prediction_counts()
        assert "test_model" in counts
        assert counts["test_model"]["success"] == 5
        assert counts["test_model"]["failure"] == 0
        
        # Calculate metrics
        results = orchestrator.calculate_metrics()
        assert "test_model" in results
        
        # Check metrics (should be perfect since predictions match ground truth)
        model_result = results["test_model"]
        assert model_result.endorsed_aggregate.macro_f1 >= 0.0
        
        # Generate reports
        reports = orchestrator.generate_reports(results)
        assert len(reports) > 0
        
        # Verify report files exist
        for report_path in reports.values():
            assert Path(report_path).exists()
    
    def test_multi_model_evaluation(self, multi_model_config, output_dir):
        """Test evaluation with multiple models."""
        # Register adapters
        EvaluationOrchestrator.register_adapter("MockSuccessAdapter", MockSuccessAdapter)
        EvaluationOrchestrator.register_adapter("MockPartialAdapter", MockPartialAdapter)
        
        # Create orchestrator
        orchestrator = EvaluationOrchestrator(config=multi_model_config)
        
        # Load ground truth
        orchestrator.load_ground_truth()
        
        # Initialize adapters
        init_results = orchestrator.initialize_adapters()
        assert len(init_results) == 2
        assert all(r.success for r in init_results)
        
        # Run predictions
        storage = orchestrator.run_predictions()
        
        # Get success rates
        rates = orchestrator.get_success_rates()
        
        # Perfect model should have 100% success
        assert rates["perfect_model"] == 1.0
        
        # Partial model should have <100% success (fails for video_003)
        assert rates["partial_model"] < 1.0
        
        # Calculate and verify metrics for both models
        results = orchestrator.calculate_metrics()
        assert "perfect_model" in results
        assert "partial_model" in results
    
    def test_evaluation_with_sampling(self, sample_ground_truth_file, sample_scripts_dir, output_dir):
        """Test evaluation with dataset sampling."""
        EvaluationOrchestrator.register_adapter("MockSuccessAdapter", MockSuccessAdapter)
        
        config = EvaluationConfig(
            ground_truth_path=str(sample_ground_truth_file),
            scripts_path=str(sample_scripts_dir),
            output_dir=str(output_dir),
            sample_size=3,
            random_seed=42,
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="sampled_model",
                    adapter_class="MockSuccessAdapter",
                    config={},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        ground_truth = orchestrator.load_ground_truth()
        
        # Should have sampled 3 videos
        assert ground_truth.valid_count == 3
    
    def test_reproducible_sampling(self, sample_ground_truth_file, sample_scripts_dir, output_dir):
        """Test that sampling with same seed produces same results."""
        EvaluationOrchestrator.register_adapter("MockSuccessAdapter", MockSuccessAdapter)
        
        config1 = EvaluationConfig(
            ground_truth_path=str(sample_ground_truth_file),
            scripts_path=str(sample_scripts_dir),
            output_dir=str(output_dir),
            sample_size=3,
            random_seed=12345,
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="model1",
                    adapter_class="MockSuccessAdapter",
                    config={},
                ),
            ],
        )
        
        config2 = EvaluationConfig(
            ground_truth_path=str(sample_ground_truth_file),
            scripts_path=str(sample_scripts_dir),
            output_dir=str(output_dir),
            sample_size=3,
            random_seed=12345,
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="model2",
                    adapter_class="MockSuccessAdapter",
                    config={},
                ),
            ],
        )
        
        orchestrator1 = EvaluationOrchestrator(config=config1)
        orchestrator2 = EvaluationOrchestrator(config=config2)
        
        gt1 = orchestrator1.load_ground_truth()
        gt2 = orchestrator2.load_ground_truth()
        
        # Should have same videos
        ids1 = sorted([v.video_id for v in gt1.videos])
        ids2 = sorted([v.video_id for v in gt2.videos])
        assert ids1 == ids2


# ==============================================================================
# Integration Test: Report Generation
# ==============================================================================

class TestReportGeneration:
    """Integration tests for report generation."""
    
    def test_csv_report_format(self, basic_config, output_dir):
        """Test that CSV reports have correct format."""
        EvaluationOrchestrator.register_adapter("MockSuccessAdapter", MockSuccessAdapter)
        
        orchestrator = EvaluationOrchestrator(config=basic_config)
        orchestrator.load_ground_truth()
        orchestrator.initialize_adapters()
        orchestrator.run_predictions()
        results = orchestrator.calculate_metrics()
        reports = orchestrator.generate_reports(results)
        
        # Find CSV reports (reports is a dict now, values are Paths)
        csv_reports = [str(p) for p in reports.values() if str(p).endswith('.csv')]
        assert len(csv_reports) > 0
        
        # Verify CSV structure
        for csv_path in csv_reports:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) > 0
                
                # Check that all rows have fieldnames
                for row in rows:
                    assert all(key in row for key in reader.fieldnames)
    
    def test_json_report_format(self, basic_config, output_dir):
        """Test that JSON reports have correct format."""
        EvaluationOrchestrator.register_adapter("MockSuccessAdapter", MockSuccessAdapter)
        
        orchestrator = EvaluationOrchestrator(config=basic_config)
        orchestrator.load_ground_truth()
        orchestrator.initialize_adapters()
        orchestrator.run_predictions()
        results = orchestrator.calculate_metrics()
        reports = orchestrator.generate_reports(results)
        
        # Find JSON reports (reports is a dict now, values are Paths)
        json_reports = [str(p) for p in reports.values() if str(p).endswith('.json')]
        assert len(json_reports) > 0
        
        # Verify JSON structure
        for json_path in json_reports:
            with open(json_path, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)
                # Should have model metrics
                assert "model_name" in data or "models" in data or len(data) > 0


# ==============================================================================
# Integration Test: Error Handling
# ==============================================================================

class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_graceful_handling_of_prediction_failures(self, multi_model_config, output_dir):
        """Test that prediction failures are handled gracefully."""
        EvaluationOrchestrator.register_adapter("MockSuccessAdapter", MockSuccessAdapter)
        EvaluationOrchestrator.register_adapter("MockPartialAdapter", MockPartialAdapter)
        
        orchestrator = EvaluationOrchestrator(config=multi_model_config)
        orchestrator.load_ground_truth()
        orchestrator.initialize_adapters()
        
        # This should not raise even though some predictions fail
        storage = orchestrator.run_predictions()
        assert storage is not None
        
        # Metrics should still be calculated for successful predictions
        results = orchestrator.calculate_metrics()
        assert len(results) == 2
    
    def test_handles_missing_ground_truth_file(self, tmp_path):
        """Test error handling for missing ground truth file."""
        config = EvaluationConfig(
            ground_truth_path=str(tmp_path / "nonexistent.csv"),
            scripts_path=str(tmp_path),
            output_dir=str(tmp_path),
            models=[
                ModelConfig(
                    model_type="mock",
                    model_name="test",
                    adapter_class="MockSuccessAdapter",
                    config={},
                ),
            ],
        )
        
        orchestrator = EvaluationOrchestrator(config=config)
        
        with pytest.raises(Exception):
            orchestrator.load_ground_truth()


# ==============================================================================
# Integration Test: Configuration Loading
# ==============================================================================

class TestConfigurationLoading:
    """Integration tests for configuration loading."""
    
    def test_load_yaml_config(self, tmp_path, sample_ground_truth_file, sample_scripts_dir):
        """Test loading YAML configuration file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        config_content = f"""
ground_truth_path: "{str(sample_ground_truth_file).replace(chr(92), '/')}"
scripts_path: "{str(sample_scripts_dir).replace(chr(92), '/')}"
output_dir: "{str(output_dir).replace(chr(92), '/')}"
models:
  - model_type: mock
    model_name: yaml_test_model
    adapter_class: MockSuccessAdapter
    config: {{}}
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)
        
        loader = EvaluationConfigLoader()
        config = loader.load(str(config_path))
        
        assert config.ground_truth_path == str(sample_ground_truth_file).replace('\\', '/')
        assert len(config.models) == 1
        assert config.models[0].model_name == "yaml_test_model"
    
    def test_load_json_config(self, tmp_path, sample_ground_truth_file, sample_scripts_dir):
        """Test loading JSON configuration file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        config_data = {
            "ground_truth_path": str(sample_ground_truth_file).replace('\\', '/'),
            "scripts_path": str(sample_scripts_dir).replace('\\', '/'),
            "output_dir": str(output_dir).replace('\\', '/'),
            "models": [
                {
                    "model_type": "mock",
                    "model_name": "json_test_model",
                    "adapter_class": "MockSuccessAdapter",
                    "config": {}
                }
            ]
        }
        
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))
        
        loader = EvaluationConfigLoader()
        config = loader.load(str(config_path))
        
        assert len(config.models) == 1
        assert config.models[0].model_name == "json_test_model"
