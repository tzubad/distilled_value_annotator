"""
Manual Testing Script for Evaluation Module
============================================

This script allows you to manually test the evaluation module in two ways:
1. LOCAL MODE: Test metrics calculation with actual/mock predictions
2. CLOUD MODE: Test the full pipeline with scripts from GCS

Usage:
    python manual_test_evaluation.py --mode local
    python manual_test_evaluation.py --mode cloud
    python manual_test_evaluation.py --mode both
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.models import (
    VideoAnnotation,
    PredictionResult,
    PredictionSet,
    GroundTruthDataset,
    EvaluationConfig,
    ModelConfig,
)
from evaluation.ground_truth_loader import GroundTruthLoader, ANNOTATION_CATEGORIES
from evaluation.metrics.calculator import MetricsCalculator, ModelEvaluationResult
from evaluation.reports.generator import ReportGenerator


def create_mock_predictions(
    ground_truth: GroundTruthDataset,
    model_name: str = "mock_model",
    accuracy_rate: float = 0.7,
    failure_rate: float = 0.05,
) -> PredictionSet:
    """
    Create mock predictions based on ground truth with configurable accuracy.
    
    Args:
        ground_truth: Ground truth dataset to base predictions on
        model_name: Name for the mock model
        accuracy_rate: Rate at which predictions match ground truth (0.0-1.0)
        failure_rate: Rate at which predictions fail (0.0-1.0)
    
    Returns:
        PredictionSet with mock predictions
    """
    import random
    
    predictions = []
    success_count = 0
    failure_count = 0
    failed_ids = []
    
    for video in ground_truth.videos:
        # Simulate occasional failures
        if random.random() < failure_rate:
            predictions.append(PredictionResult(
                video_id=video.video_id,
                predictions={},
                success=False,
                error_message="Simulated prediction failure",
                inference_time=0.0,
            ))
            failure_count += 1
            failed_ids.append(video.video_id)
            continue
        
        # Create predictions with some noise
        pred_annotations = {}
        for category, gt_value in video.annotations.items():
            if random.random() < accuracy_rate:
                # Correct prediction
                pred_annotations[category] = gt_value
            else:
                # Random incorrect prediction
                possible_values = [-1, 0, 1, 2]
                possible_values.remove(gt_value) if gt_value in possible_values else None
                pred_annotations[category] = random.choice(possible_values)
        
        predictions.append(PredictionResult(
            video_id=video.video_id,
            predictions=pred_annotations,
            success=True,
            error_message=None,
            inference_time=random.uniform(0.1, 2.0),
        ))
        success_count += 1
    
    return PredictionSet(
        model_name=model_name,
        predictions=predictions,
        total_count=len(predictions),
        success_count=success_count,
        failure_count=failure_count,
        failed_video_ids=failed_ids,
    )


def create_manual_predictions(
    ground_truth: GroundTruthDataset,
    model_name: str = "manual_test_model",
    predictions_file: Optional[str] = None,
) -> PredictionSet:
    """
    Create predictions from a JSON file or return a template.
    
    If predictions_file is provided, loads predictions from it.
    Otherwise, prints a template and creates predictions matching ground truth.
    
    Args:
        ground_truth: Ground truth dataset
        model_name: Name for the model
        predictions_file: Optional path to JSON file with predictions
    
    Returns:
        PredictionSet with predictions
    """
    if predictions_file and Path(predictions_file).exists():
        # Load from file
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        predictions = []
        for item in data:
            pred = PredictionResult(
                video_id=item['video_id'],
                predictions=item['predictions'],
                success=item.get('success', True),
                error_message=item.get('error_message'),
                inference_time=item.get('inference_time', 0.0),
            )
            predictions.append(pred)
        
        success_count = sum(1 for p in predictions if p.success)
        failed_ids = [p.video_id for p in predictions if not p.success]
        
        return PredictionSet(
            model_name=model_name,
            predictions=predictions,
            total_count=len(predictions),
            success_count=success_count,
            failure_count=len(predictions) - success_count,
            failed_video_ids=failed_ids,
        )
    
    # Create template file for user to fill in
    template_path = PROJECT_ROOT / "test_predictions_template.json"
    template = []
    
    for video in ground_truth.videos[:5]:  # Just first 5 for template
        template.append({
            "video_id": video.video_id,
            "predictions": {cat: 0 for cat in ANNOTATION_CATEGORIES},
            "success": True,
            "error_message": None,
            "inference_time": 1.0,
        })
    
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created prediction template at: {template_path}")
    logger.info("Fill in your predictions and re-run with: --predictions test_predictions_template.json")
    
    # Return predictions matching ground truth (for demo)
    predictions = []
    for video in ground_truth.videos:
        predictions.append(PredictionResult(
            video_id=video.video_id,
            predictions=dict(video.annotations),  # Copy ground truth
            success=True,
            error_message=None,
            inference_time=1.0,
        ))
    
    return PredictionSet(
        model_name=model_name,
        predictions=predictions,
        total_count=len(predictions),
        success_count=len(predictions),
        failure_count=0,
        failed_video_ids=[],
    )


def test_local_metrics(
    ground_truth_path: str,
    predictions_file: Optional[str] = None,
    output_dir: str = "./test_results",
    sample_size: Optional[int] = None,
    use_mock: bool = True,
    mock_accuracy: float = 0.7,
) -> Dict[str, ModelEvaluationResult]:
    """
    Test metrics calculation locally with actual or mock predictions.
    
    Args:
        ground_truth_path: Path to ground truth CSV
        predictions_file: Optional path to predictions JSON
        output_dir: Directory for output reports
        sample_size: Optional sample size for testing
        use_mock: Whether to use mock predictions
        mock_accuracy: Accuracy rate for mock predictions
    
    Returns:
        Dictionary of model evaluation results
    """
    logger.info("=" * 60)
    logger.info("LOCAL METRICS TEST")
    logger.info("=" * 60)
    
    # Step 1: Load ground truth
    logger.info(f"\n1. Loading ground truth from: {ground_truth_path}")
    loader = GroundTruthLoader(
        dataset_path=ground_truth_path,
        sample_size=sample_size,
        random_seed=42,
    )
    ground_truth = loader.load()
    
    logger.info(f"   Loaded {ground_truth.valid_count} valid videos out of {ground_truth.total_count} total")
    if ground_truth.validation_errors:
        logger.warning(f"   Found {len(ground_truth.validation_errors)} validation errors")
        for error in ground_truth.validation_errors[:5]:
            logger.warning(f"   - {error}")
    
    # Show sample of loaded data
    logger.info("\n   Sample videos loaded:")
    for video in ground_truth.videos[:3]:
        endorsed_count = sum(1 for v in video.annotations.values() if v in {1, 2})
        conflict_count = sum(1 for v in video.annotations.values() if v == -1)
        logger.info(f"   - {video.video_id}: {endorsed_count} endorsed, {conflict_count} conflicts")
    
    # Step 2: Create predictions
    logger.info("\n2. Creating predictions...")
    results = {}
    
    if use_mock:
        # Create multiple mock models with different accuracies
        mock_configs = [
            ("high_accuracy_model", 0.85, 0.02),
            ("medium_accuracy_model", 0.65, 0.05),
            ("low_accuracy_model", 0.45, 0.10),
        ]
        
        for model_name, accuracy, failure_rate in mock_configs:
            logger.info(f"   Creating {model_name} (accuracy={accuracy}, failure_rate={failure_rate})")
            predictions = create_mock_predictions(
                ground_truth,
                model_name=model_name,
                accuracy_rate=accuracy,
                failure_rate=failure_rate,
            )
            
            # Calculate metrics
            logger.info(f"   Calculating metrics for {model_name}...")
            calculator = MetricsCalculator(
                ground_truth=ground_truth,
                min_frequency_threshold=0.05,
            )
            result = calculator.calculate_model_metrics(predictions)
            results[model_name] = result
            
            # Print summary
            logger.info(f"   Results for {model_name}:")
            logger.info(f"     - Matched: {result.matched_with_ground_truth}/{result.total_predictions}")
            logger.info(f"     - Endorsed Macro F1: {result.endorsed_aggregate.macro_f1:.4f}")
            logger.info(f"     - Conflict Macro F1: {result.conflict_aggregate.macro_f1:.4f}")
    else:
        # Use manual predictions
        predictions = create_manual_predictions(
            ground_truth,
            model_name="manual_test",
            predictions_file=predictions_file,
        )
        
        calculator = MetricsCalculator(
            ground_truth=ground_truth,
            min_frequency_threshold=0.05,
        )
        result = calculator.calculate_model_metrics(predictions)
        results["manual_test"] = result
    
    # Step 3: Generate reports
    logger.info(f"\n3. Generating reports to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    generator = ReportGenerator(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generated_files = generator.generate_all_reports(list(results.values()), timestamp)
    
    logger.info("   Generated reports:")
    for report_type, path in generated_files.items():
        logger.info(f"   - {report_type}: {path}")
    
    # Step 4: Print detailed category breakdown
    logger.info("\n4. Per-Category Metrics (first model):")
    first_result = list(results.values())[0]
    logger.info(f"\n{'Category':<30} {'Endorsed F1':>12} {'Conflict F1':>12} {'Support':>10}")
    logger.info("-" * 66)
    
    for category in ANNOTATION_CATEGORIES:
        endorsed = first_result.per_category_endorsed.get(category)
        conflict = first_result.per_category_conflict.get(category)
        if endorsed and conflict:
            logger.info(f"{category:<30} {endorsed.f1:>12.4f} {conflict.f1:>12.4f} {endorsed.support:>10}")
    
    logger.info("\n" + "=" * 60)
    logger.info("LOCAL TEST COMPLETE")
    logger.info("=" * 60)
    
    return results


def test_cloud_pipeline(
    bucket_name: str,
    scripts_prefix: str,
    ground_truth_path: str,
    sample_size: int = 5,
) -> None:
    """
    Test the pipeline with scripts from GCS.
    
    Args:
        bucket_name: GCS bucket name
        scripts_prefix: Path prefix for scripts in bucket
        ground_truth_path: Local path to ground truth CSV
        sample_size: Number of videos to test
    """
    logger.info("=" * 60)
    logger.info("CLOUD PIPELINE TEST")
    logger.info("=" * 60)
    
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        return
    
    # Step 1: Connect to GCS
    logger.info(f"\n1. Connecting to GCS bucket: {bucket_name}")
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List scripts
        blobs = list(bucket.list_blobs(prefix=scripts_prefix, max_results=sample_size * 2))
        logger.info(f"   Found {len(blobs)} files in {scripts_prefix}")
        
        if not blobs:
            logger.warning("   No scripts found! Please upload some scripts first.")
            return
        
        # Show sample
        for blob in blobs[:5]:
            logger.info(f"   - {blob.name} ({blob.size} bytes)")
    
    except Exception as e:
        logger.error(f"   Failed to connect to GCS: {e}")
        logger.info("\n   Make sure you're authenticated with Google Cloud:")
        logger.info("   gcloud auth application-default login")
        return
    
    # Step 2: Load ground truth
    logger.info(f"\n2. Loading ground truth from: {ground_truth_path}")
    loader = GroundTruthLoader(
        dataset_path=ground_truth_path,
        sample_size=sample_size,
        random_seed=42,
    )
    ground_truth = loader.load()
    logger.info(f"   Loaded {ground_truth.valid_count} videos")
    
    # Step 3: Test script loading
    logger.info("\n3. Testing script loading from GCS...")
    from evaluation.adapters.script_loader import ScriptLoader
    
    script_loader = ScriptLoader()
    successful_loads = 0
    failed_loads = 0
    
    for video in ground_truth.videos[:sample_size]:
        # Construct GCS URI
        script_uri = f"gs://{bucket_name}/{scripts_prefix}/{video.video_id}.txt"
        
        content = script_loader.load_script(script_uri)
        if content:
            logger.info(f"   ✓ Loaded script for {video.video_id} ({len(content)} chars)")
            successful_loads += 1
        else:
            logger.warning(f"   ✗ Failed to load script for {video.video_id}")
            failed_loads += 1
    
    logger.info(f"\n   Script loading: {successful_loads} successful, {failed_loads} failed")
    
    # Step 4: Summary
    logger.info("\n" + "=" * 60)
    logger.info("CLOUD PIPELINE TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Bucket: {bucket_name}")
    logger.info(f"Scripts prefix: {scripts_prefix}")
    logger.info(f"Scripts loaded: {successful_loads}/{sample_size}")


def upload_test_scripts(
    bucket_name: str,
    scripts_prefix: str,
    local_scripts_dir: str,
) -> None:
    """
    Upload local scripts to GCS for testing.
    
    Args:
        bucket_name: GCS bucket name
        scripts_prefix: Path prefix for scripts in bucket
        local_scripts_dir: Local directory containing scripts
    """
    logger.info("=" * 60)
    logger.info("UPLOADING TEST SCRIPTS TO GCS")
    logger.info("=" * 60)
    
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        return
    
    local_path = Path(local_scripts_dir)
    if not local_path.exists():
        logger.error(f"Local scripts directory not found: {local_scripts_dir}")
        return
    
    # Find script files
    script_files = list(local_path.glob("*.txt"))
    if not script_files:
        logger.warning(f"No .txt files found in {local_scripts_dir}")
        return
    
    logger.info(f"Found {len(script_files)} script files to upload")
    
    # Connect and upload
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        uploaded = 0
        for script_file in script_files:
            blob_path = f"{scripts_prefix}/{script_file.name}"
            blob = bucket.blob(blob_path)
            
            blob.upload_from_filename(str(script_file))
            logger.info(f"   ✓ Uploaded: {blob_path}")
            uploaded += 1
        
        logger.info(f"\nSuccessfully uploaded {uploaded} scripts to gs://{bucket_name}/{scripts_prefix}/")
    
    except Exception as e:
        logger.error(f"Failed to upload: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Manual testing script for the evaluation module"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "cloud", "both", "upload"],
        default="local",
        help="Test mode: local (metrics only), cloud (GCS pipeline), both, or upload (upload scripts to GCS)"
    )
    parser.add_argument(
        "--ground-truth",
        default="cleaned_groundtruth_values_only.csv",
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "--predictions",
        help="Path to predictions JSON file (for local mode)"
    )
    parser.add_argument(
        "--output-dir",
        default="./manual_test_results",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of videos to sample (default: use all)"
    )
    parser.add_argument(
        "--bucket",
        default="videos-scripts-and-annotations",
        help="GCS bucket name (for cloud mode)"
    )
    parser.add_argument(
        "--scripts-prefix",
        default="saved_scripts/POC_scripts",
        help="Scripts prefix in GCS bucket"
    )
    parser.add_argument(
        "--local-scripts",
        help="Local directory with scripts to upload (for upload mode)"
    )
    parser.add_argument(
        "--no-mock",
        action="store_true",
        help="Don't use mock predictions, create template instead"
    )
    parser.add_argument(
        "--mock-accuracy",
        type=float,
        default=0.7,
        help="Accuracy rate for mock predictions (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Validate ground truth path
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        # Try relative to project root
        gt_path = PROJECT_ROOT / args.ground_truth
    
    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        logger.info("Available CSV files:")
        for csv_file in PROJECT_ROOT.glob("*.csv"):
            logger.info(f"  - {csv_file.name}")
        return 1
    
    # Run requested tests
    if args.mode in ["local", "both"]:
        test_local_metrics(
            ground_truth_path=str(gt_path),
            predictions_file=args.predictions,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            use_mock=not args.no_mock,
            mock_accuracy=args.mock_accuracy,
        )
    
    if args.mode in ["cloud", "both"]:
        test_cloud_pipeline(
            bucket_name=args.bucket,
            scripts_prefix=args.scripts_prefix,
            ground_truth_path=str(gt_path),
            sample_size=args.sample_size or 5,
        )
    
    if args.mode == "upload":
        if not args.local_scripts:
            logger.error("--local-scripts is required for upload mode")
            return 1
        
        upload_test_scripts(
            bucket_name=args.bucket,
            scripts_prefix=args.scripts_prefix,
            local_scripts_dir=args.local_scripts,
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
