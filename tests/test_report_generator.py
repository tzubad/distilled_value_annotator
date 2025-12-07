# Tests for ReportGenerator

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

from evaluation.models import (
    VideoAnnotation,
    PredictionResult,
    PredictionSet,
    GroundTruthDataset,
)
from evaluation.metrics.calculator import (
    ANNOTATION_CATEGORIES,
    MetricsCalculator,
    ModelEvaluationResult,
    CategoryResult,
    AggregateResult,
)
from evaluation.reports import ReportGenerator


# Helper to create a valid ModelEvaluationResult for testing
def create_test_evaluation_result(model_name: str = "test_model") -> ModelEvaluationResult:
    """Create a minimal valid ModelEvaluationResult for testing."""
    # Create per-category metrics
    per_category_endorsed = {}
    per_category_conflict = {}
    per_category_combined = {}
    
    for i, category in enumerate(ANNOTATION_CATEGORIES):
        per_category_endorsed[category] = CategoryResult(
            category=category,
            precision=0.8 + (i % 3) * 0.05,
            recall=0.7 + (i % 3) * 0.05,
            f1=0.75 + (i % 3) * 0.05,
            support=10 + i,
            true_positives=8,
            false_positives=2,
            false_negatives=3,
        )
        per_category_conflict[category] = CategoryResult(
            category=category,
            precision=0.6,
            recall=0.5,
            f1=0.55,
            support=5,
            true_positives=3,
            false_positives=2,
            false_negatives=2,
        )
        per_category_combined[category] = CategoryResult(
            category=category,
            precision=0.7,
            recall=0.65,
            f1=0.67,
            support=15 + i,
            true_positives=10,
            false_positives=4,
            false_negatives=5,
        )
    
    return ModelEvaluationResult(
        model_name=model_name,
        total_predictions=100,
        successful_predictions=95,
        failed_predictions=5,
        matched_with_ground_truth=90,
        unmatched_count=5,
        missing_count=10,
        endorsed_aggregate=AggregateResult(
            macro_f1=0.78,
            weighted_f1=0.80,
            macro_precision=0.82,
            macro_recall=0.75,
            categories_evaluated=19,
        ),
        conflict_aggregate=AggregateResult(
            macro_f1=0.55,
            weighted_f1=0.52,
            macro_precision=0.60,
            macro_recall=0.50,
            categories_evaluated=19,
        ),
        combined_aggregate=AggregateResult(
            macro_f1=0.67,
            weighted_f1=0.68,
            macro_precision=0.70,
            macro_recall=0.65,
            categories_evaluated=19,
        ),
        per_category_endorsed=per_category_endorsed,
        per_category_conflict=per_category_conflict,
        per_category_combined=per_category_combined,
    )


class TestReportGeneratorInitialization:
    """Tests for ReportGenerator initialization."""
    
    def test_creates_output_directory(self):
        """ReportGenerator should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_reports_dir"
            assert not output_dir.exists()
            
            generator = ReportGenerator(str(output_dir))
            
            assert output_dir.exists()
            assert output_dir.is_dir()
    
    def test_uses_existing_directory(self):
        """ReportGenerator should work with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            assert generator.output_dir == Path(tmpdir)


class TestCSVReportGeneration:
    """Tests for CSV report generation."""
    
    def test_generates_category_csv(self):
        """Should generate per-category metrics CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            csv_path = generator.generate_csv_report(result, "test_timestamp")
            
            assert csv_path.exists()
            assert csv_path.suffix == '.csv'
            assert 'category_metrics' in csv_path.name
    
    def test_category_csv_has_all_categories(self):
        """Category CSV should include all 19 annotation categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            csv_path = generator.generate_csv_report(result, "test")
            
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 19
            categories_in_csv = {row['category'] for row in rows}
            assert categories_in_csv == set(ANNOTATION_CATEGORIES)
    
    def test_category_csv_has_all_metric_columns(self):
        """Category CSV should have columns for all value types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            csv_path = generator.generate_csv_report(result, "test")
            
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
            
            expected_columns = [
                'category',
                'endorsed_precision', 'endorsed_recall', 'endorsed_f1', 'endorsed_support',
                'conflict_precision', 'conflict_recall', 'conflict_f1', 'conflict_support',
                'combined_precision', 'combined_recall', 'combined_f1', 'combined_support',
            ]
            assert header == expected_columns
    
    def test_generates_aggregate_csv(self):
        """Should generate aggregate metrics CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            generator.generate_csv_report(result, "test_timestamp")
            
            # Check that aggregate CSV was also created
            aggregate_path = Path(tmpdir) / "test_model_aggregate_metrics_test_timestamp.csv"
            assert aggregate_path.exists()
    
    def test_aggregate_csv_has_all_value_types(self):
        """Aggregate CSV should have rows for endorsed, conflict, and combined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            generator.generate_csv_report(result, "test")
            
            aggregate_path = Path(tmpdir) / "test_model_aggregate_metrics_test.csv"
            with open(aggregate_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            metric_types = {row['metric_type'] for row in rows}
            assert metric_types == {'endorsed', 'conflict', 'combined'}


class TestJSONReportGeneration:
    """Tests for JSON report generation."""
    
    def test_generates_json_file(self):
        """Should generate JSON report file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            json_path = generator.generate_json_report(result, "test_timestamp")
            
            assert json_path.exists()
            assert json_path.suffix == '.json'
    
    def test_json_is_valid(self):
        """JSON file should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            json_path = generator.generate_json_report(result, "test")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)  # Should not raise
            
            assert isinstance(data, dict)
    
    def test_json_contains_model_name(self):
        """JSON should contain correct model name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result("my_test_model")
            
            json_path = generator.generate_json_report(result, "test")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data['model_name'] == "my_test_model"
    
    def test_json_contains_summary_stats(self):
        """JSON should contain summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            json_path = generator.generate_json_report(result, "test")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'summary' in data
            assert data['summary']['total_predictions'] == 100
            assert data['summary']['successful_predictions'] == 95
    
    def test_json_contains_aggregate_metrics(self):
        """JSON should contain aggregate metrics for all value types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            json_path = generator.generate_json_report(result, "test")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'aggregate_metrics' in data
            assert 'endorsed' in data['aggregate_metrics']
            assert 'conflict' in data['aggregate_metrics']
            assert 'combined' in data['aggregate_metrics']
            
            # Check specific values
            assert data['aggregate_metrics']['endorsed']['macro_f1'] == 0.78
    
    def test_json_contains_per_category_metrics(self):
        """JSON should contain per-category metrics for all categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            json_path = generator.generate_json_report(result, "test")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'per_category_metrics' in data
            assert 'endorsed' in data['per_category_metrics']
            
            # Check all categories present
            endorsed_cats = set(data['per_category_metrics']['endorsed'].keys())
            assert endorsed_cats == set(ANNOTATION_CATEGORIES)
    
    def test_json_contains_metadata(self):
        """JSON should contain report metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            json_path = generator.generate_json_report(result, "test")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'generated_at' in data['metadata']
            assert 'report_version' in data['metadata']


class TestComparisonReportGeneration:
    """Tests for comparison report generation."""
    
    def test_generates_comparison_csv(self):
        """Should generate comparison CSV for multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            results = [
                create_test_evaluation_result("model_a"),
                create_test_evaluation_result("model_b"),
            ]
            
            comparison_path = generator.generate_comparison_report(results, "test")
            
            assert comparison_path.exists()
            assert comparison_path.suffix == '.csv'
            assert 'comparison' in comparison_path.name
    
    def test_comparison_includes_all_models(self):
        """Comparison report should include all model names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            results = [
                create_test_evaluation_result("model_a"),
                create_test_evaluation_result("model_b"),
                create_test_evaluation_result("model_c"),
            ]
            
            comparison_path = generator.generate_comparison_report(results, "test")
            
            with open(comparison_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert 'model_a' in content
            assert 'model_b' in content
            assert 'model_c' in content
    
    def test_comparison_includes_rankings(self):
        """Comparison report should include model rankings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            results = [
                create_test_evaluation_result("model_a"),
                create_test_evaluation_result("model_b"),
            ]
            
            comparison_path = generator.generate_comparison_report(results, "test")
            
            with open(comparison_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert 'RANKINGS' in content


class TestGenerateAllReports:
    """Tests for generating all report formats at once."""
    
    def test_generates_csv_and_json_for_single_model(self):
        """Should generate both CSV and JSON for a single model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            files = generator.generate_all_reports([result], "test")
            
            assert f"{result.model_name}_csv" in files
            assert f"{result.model_name}_json" in files
            assert files[f"{result.model_name}_csv"].exists()
            assert files[f"{result.model_name}_json"].exists()
    
    def test_generates_comparison_for_multiple_models(self):
        """Should generate comparison report when multiple models provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            results = [
                create_test_evaluation_result("model_a"),
                create_test_evaluation_result("model_b"),
            ]
            
            files = generator.generate_all_reports(results, "test")
            
            assert 'comparison' in files
            assert files['comparison'].exists()
    
    def test_no_comparison_for_single_model(self):
        """Should not generate comparison report for single model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            files = generator.generate_all_reports([result], "test")
            
            assert 'comparison' not in files


class TestReportFormatCompleteness:
    """Property test for report format completeness (Property 17)."""
    
    def test_both_csv_and_json_generated(self):
        """
        Property 17: Report format completeness
        
        Validates: Requirements 8.4
        Verifies: Both CSV and JSON files generated for each model evaluation
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result()
            
            files = generator.generate_all_reports([result], "test")
            
            # Check CSV exists
            csv_files = [f for f in files.values() if f.suffix == '.csv']
            assert len(csv_files) >= 1, "At least one CSV file should be generated"
            
            # Check JSON exists
            json_files = [f for f in files.values() if f.suffix == '.json']
            assert len(json_files) == 1, "Exactly one JSON file should be generated per model"
            
            # Verify files have content
            for path in files.values():
                assert path.stat().st_size > 0, f"File {path} should not be empty"
    
    @given(model_count=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10)
    def test_report_count_matches_model_count(self, model_count):
        """
        Property: Number of generated report sets should match number of models.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            results = [
                create_test_evaluation_result(f"model_{i}") 
                for i in range(model_count)
            ]
            
            files = generator.generate_all_reports(results, "test")
            
            # Each model should have CSV and JSON
            csv_count = sum(1 for k in files if k.endswith('_csv'))
            json_count = sum(1 for k in files if k.endswith('_json'))
            
            assert csv_count == model_count
            assert json_count == model_count
            
            # Comparison only for multiple models
            if model_count > 1:
                assert 'comparison' in files
            else:
                assert 'comparison' not in files


class TestFilenameSanitization:
    """Tests for filename sanitization."""
    
    def test_sanitizes_special_characters(self):
        """Should sanitize special characters in model names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = ReportGenerator(tmpdir)
            result = create_test_evaluation_result("models/gemini:1.5-pro")
            
            # Should not raise
            json_path = generator.generate_json_report(result, "test")
            
            assert json_path.exists()
            assert '/' not in json_path.name
            assert ':' not in json_path.name
