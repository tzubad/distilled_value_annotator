"""
End-to-end integration test for the Model Evaluation Module.

This test verifies:
1. Ground truth loading from real CSV data
2. Predictions loading from real CSV data  
3. Metrics calculation (precision, recall, F1)
4. Report generation (CSV and JSON)

Run this test with: python test_e2e_real_data.py
"""

import sys
import os
import tempfile
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation import (
    GroundTruthLoader, 
    normalize_video_id,
    PredictionSet,
    ANNOTATION_CATEGORIES
)
from evaluation.adapters.gemini_adapter import GeminiAdapter
from evaluation.metrics.calculator import MetricsCalculator
from evaluation.reports.generator import ReportGenerator


def run_e2e_test():
    """Run end-to-end test with real data."""
    
    # File paths
    ground_truth_path = r"C:\Users\User\Desktop\deValuating_TikTok - freeze copy\Data\cleaned_groundtruth_values_only.csv"
    predictions_path = r"C:\Users\User\Desktop\deValuating_TikTok - freeze copy\Data\gemini_script2value_raw.csv"
    
    # Check if files exist
    if not os.path.exists(ground_truth_path):
        print(f"⚠ Ground truth file not found: {ground_truth_path}")
        print("  Skipping end-to-end test.")
        return True  # Not a failure, just skip
    
    if not os.path.exists(predictions_path):
        print(f"⚠ Predictions file not found: {predictions_path}")
        print("  Skipping end-to-end test.")
        return True  # Not a failure, just skip
    
    print("=" * 70)
    print("END-TO-END INTEGRATION TEST - Model Evaluation Module")
    print("=" * 70)
    
    start_time = time.time()
    
    # Step 1: Load Ground Truth
    print("\n📥 Step 1: Loading Ground Truth Dataset...")
    step_start = time.time()
    try:
        loader = GroundTruthLoader(ground_truth_path)
        dataset = loader.load()
        print(f"   ✓ Loaded {dataset.valid_count} valid videos ({time.time() - step_start:.2f}s)")
        
        # Validate dataset structure
        assert len(dataset.videos) > 0, "No videos loaded"
        for video in dataset.videos[:5]:  # Check first 5
            assert len(video.annotations) == 19, f"Video {video.video_id} has {len(video.annotations)} annotations, expected 19"
            for cat, val in video.annotations.items():
                assert val in {-1, 0, 1, 2}, f"Invalid value {val} for {cat}"
        print(f"   ✓ Dataset structure validated")
                
    except Exception as e:
        print(f"   ✗ Error loading ground truth: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Load Predictions
    print("\n📥 Step 2: Loading Gemini Predictions...")
    step_start = time.time()
    try:
        prediction_set = GeminiAdapter.load_predictions_from_csv(
            predictions_path, 
            model_name="gemini-1.5-pro"
        )
        print(f"   ✓ Loaded {prediction_set.success_count} successful predictions ({time.time() - step_start:.2f}s)")
        
        # Validate prediction structure
        assert prediction_set.success_count > 0, "No successful predictions"
        for pred in prediction_set.predictions[:5]:  # Check first 5
            if pred.success:
                assert len(pred.predictions) == 19, f"Prediction for {pred.video_id} has {len(pred.predictions)} values, expected 19"
                for cat, val in pred.predictions.items():
                    assert val in {-1, 0, 1, 2}, f"Invalid prediction value {val} for {cat}"
        print(f"   ✓ Prediction structure validated")
        
    except Exception as e:
        print(f"   ✗ Error loading predictions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Check Video ID Matching
    print("\n🔗 Step 3: Checking Video ID Matching...")
    gt_ids = set(v.video_id for v in dataset.videos)
    pred_ids = set(p.video_id for p in prediction_set.predictions if p.success)
    matched = gt_ids & pred_ids
    
    match_rate = len(matched) / max(len(gt_ids), 1) * 100
    print(f"   Ground truth: {len(gt_ids)} videos")
    print(f"   Predictions:  {len(pred_ids)} videos")
    print(f"   Matched:      {len(matched)} videos ({match_rate:.1f}%)")
    
    if match_rate < 50:
        print(f"   ✗ Match rate too low: {match_rate:.1f}%")
        return False
    print(f"   ✓ Match rate acceptable: {match_rate:.1f}%")
    
    # Step 4: Calculate Metrics
    print("\n📊 Step 4: Calculating Metrics...")
    step_start = time.time()
    try:
        calculator = MetricsCalculator(
            ground_truth=dataset,
            min_frequency_threshold=0.05
        )
        model_metrics = calculator.calculate_model_metrics(
            predictions=prediction_set
        )
        print(f"   ✓ Metrics calculated ({time.time() - step_start:.2f}s)")
        
        # Validate metrics structure
        print(f"\n   📈 Metrics Summary for '{model_metrics.model_name}':")
        print(f"      - Matched videos:      {model_metrics.matched_with_ground_truth}")
        print(f"      - Missing predictions: {model_metrics.missing_count}")
        print(f"      - Unmatched predictions: {model_metrics.unmatched_count}")
        print(f"      - Success rate:        {model_metrics.successful_predictions / model_metrics.total_predictions:.2%}")
        print(f"      - Categories evaluated: {len(model_metrics.per_category_endorsed)}")
        
        # Show aggregate metrics
        print(f"\n   📊 Aggregate Metrics:")
        print(f"      Endorsed:")
        print(f"        - Macro F1:    {model_metrics.endorsed_aggregate.macro_f1:.4f}")
        print(f"        - Weighted F1: {model_metrics.endorsed_aggregate.weighted_f1:.4f}")
        print(f"      Conflict:")
        print(f"        - Macro F1:    {model_metrics.conflict_aggregate.macro_f1:.4f}")
        print(f"        - Weighted F1: {model_metrics.conflict_aggregate.weighted_f1:.4f}")
        print(f"      Combined:")
        print(f"        - Macro F1:    {model_metrics.combined_aggregate.macro_f1:.4f}")
        print(f"        - Weighted F1: {model_metrics.combined_aggregate.weighted_f1:.4f}")
        
        # Validate F1 scores are in valid range
        assert 0 <= model_metrics.endorsed_aggregate.macro_f1 <= 1, "Invalid endorsed macro F1"
        assert 0 <= model_metrics.conflict_aggregate.macro_f1 <= 1, "Invalid conflict macro F1"
        assert 0 <= model_metrics.combined_aggregate.macro_f1 <= 1, "Invalid combined macro F1"
        print(f"   ✓ Metrics values validated")
        
    except Exception as e:
        print(f"   ✗ Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Generate Reports
    print("\n📝 Step 5: Generating Reports...")
    step_start = time.time()
    try:
        with tempfile.TemporaryDirectory() as output_dir:
            generator = ReportGenerator(output_dir)
            
            # Generate all reports
            report_paths = generator.generate_all_reports(
                results=[model_metrics]
            )
            
            print(f"   ✓ Reports generated ({time.time() - step_start:.2f}s)")
            print(f"   Generated files:")
            for report_type, path in report_paths.items():
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    print(f"      - {report_type}: {os.path.basename(path)} ({size:,} bytes)")
                else:
                    print(f"      - {report_type}: NOT FOUND")
            
            # Validate report contents
            for report_type, path in report_paths.items():
                assert os.path.exists(path), f"Report not generated: {report_type}"
                assert os.path.getsize(path) > 0, f"Report is empty: {report_type}"
            print(f"   ✓ All reports validated")
            
    except Exception as e:
        print(f"   ✗ Error generating reports: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ END-TO-END TEST PASSED ({total_time:.2f}s)")
    print("=" * 70)
    
    print(f"\n📋 Test Summary:")
    print(f"   - Ground truth videos: {dataset.valid_count}")
    print(f"   - Predictions loaded:  {prediction_set.success_count}")
    print(f"   - Videos matched:      {model_metrics.matched_with_ground_truth}")
    print(f"   - Aggregate macro F1:  {model_metrics.combined_aggregate.macro_f1:.4f}")
    print(f"   - Reports generated:   {len(report_paths)}")
    
    return True


def run_performance_test():
    """Run performance test on the metrics calculation."""
    
    ground_truth_path = r"C:\Users\User\Desktop\deValuating_TikTok - freeze copy\Data\cleaned_groundtruth_values_only.csv"
    predictions_path = r"C:\Users\User\Desktop\deValuating_TikTok - freeze copy\Data\gemini_script2value_raw.csv"
    
    if not (os.path.exists(ground_truth_path) and os.path.exists(predictions_path)):
        print("⚠ Data files not found, skipping performance test.")
        return True
    
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST")
    print("=" * 70)
    
    # Load data
    loader = GroundTruthLoader(ground_truth_path)
    dataset = loader.load()
    
    prediction_set = GeminiAdapter.load_predictions_from_csv(
        predictions_path, 
        model_name="gemini-1.5-pro"
    )
    
    # Run metrics calculation multiple times
    iterations = 5
    times = []
    
    print(f"\nRunning metrics calculation {iterations} times...")
    
    for i in range(iterations):
        calculator = MetricsCalculator(
            ground_truth=dataset,
            min_frequency_threshold=0.05
        )
        start = time.time()
        model_metrics = calculator.calculate_model_metrics(
            predictions=prediction_set
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Performance Results:")
    print(f"   - Average time: {avg_time:.3f}s")
    print(f"   - Min time:     {min_time:.3f}s")
    print(f"   - Max time:     {max_time:.3f}s")
    print(f"   - Videos/sec:   {dataset.valid_count / avg_time:.1f}")
    
    return True


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("MODEL EVALUATION MODULE - INTEGRATION TEST SUITE")
    print("🚀 " * 20 + "\n")
    
    e2e_success = run_e2e_test()
    perf_success = run_performance_test()
    
    if e2e_success and perf_success:
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)
