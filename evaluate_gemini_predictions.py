"""
Evaluate Predictions from CSV
==============================

This script loads predictions from a CSV file and evaluates them
against the ground truth using the evaluation module.

Uses PredictionLoader which reuses GroundTruthLoader's parsing logic.

Usage:
    python evaluate_gemini_predictions.py
    python evaluate_gemini_predictions.py --predictions "path/to/predictions.csv"
    python evaluate_gemini_predictions.py --sample-size 50
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from evaluation module - reusing implemented components
from evaluation.ground_truth_loader import GroundTruthLoader, ANNOTATION_CATEGORIES
from evaluation.prediction_loader import PredictionLoader
from evaluation.metrics.calculator import MetricsCalculator
from evaluation.reports.generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against ground truth"
    )
    parser.add_argument(
        "--predictions",
        default=r"C:\Users\User\Desktop\deValuating_TikTok - freeze copy\Data\gemini_script2value_raw.csv",
        help="Path to predictions CSV"
    )
    parser.add_argument(
        "--model-name",
        default="gemini-1.5",
        help="Name for the model being evaluated"
    )
    parser.add_argument(
        "--ground-truth",
        default="cleaned_groundtruth_values_only.csv",
        help="Path to ground truth CSV"
    )
    parser.add_argument(
        "--output-dir",
        default="./gemini-1.5_evaluation_results",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for ground truth (default: use all)"
    )
    parser.add_argument(
        "--min-frequency",
        type=float,
        default=0.05,
        help="Minimum frequency threshold for categories (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PREDICTIONS EVALUATION")
    print("=" * 70)
    
    # Step 1: Load predictions using PredictionLoader (reuses GroundTruthLoader logic)
    print(f"\n📥 Loading predictions from:")
    print(f"   {args.predictions}")
    
    if not Path(args.predictions).exists():
        logger.error(f"Predictions file not found: {args.predictions}")
        return 1
    
    pred_loader = PredictionLoader(model_name=args.model_name)
    predictions = pred_loader.load(args.predictions)
    
    print(f"   ✓ Loaded {predictions.total_count} predictions")
    
    # Show sample predictions
    print("\n   Sample predictions:")
    for pred in predictions.predictions[:3]:
        endorsed = sum(1 for v in pred.predictions.values() if v in {1, 2})
        conflict = sum(1 for v in pred.predictions.values() if v == -1)
        print(f"   - {pred.video_id}: {endorsed} endorsed, {conflict} conflicts")
    
    # Step 2: Load ground truth
    print(f"\n📥 Loading ground truth from:")
    print(f"   {args.ground_truth}")
    
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        gt_path = PROJECT_ROOT / args.ground_truth
    
    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {args.ground_truth}")
        return 1
    
    loader = GroundTruthLoader(
        dataset_path=str(gt_path),
        sample_size=args.sample_size,
        random_seed=42,
    )
    ground_truth = loader.load()
    
    print(f"   ✓ Loaded {ground_truth.valid_count} valid videos (total: {ground_truth.total_count})")
    
    # Step 3: Calculate metrics
    print(f"\n🔍 Calculating metrics...")
    
    calculator = MetricsCalculator(
        ground_truth=ground_truth,
        min_frequency_threshold=args.min_frequency,
    )
    
    result = calculator.calculate_model_metrics(predictions)
    
    print(f"   ✓ Matched: {result.matched_with_ground_truth} videos")
    print(f"   ✓ Unmatched predictions: {result.unmatched_count}")
    print(f"   ✓ Missing from predictions: {result.missing_count}")
    
    # Step 4: Display results
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Endorsed':>12} {'Conflict':>12} {'Combined':>12}")
    print("-" * 70)
    print(f"{'Macro F1':<30} {result.endorsed_aggregate.macro_f1:>12.4f} "
          f"{result.conflict_aggregate.macro_f1:>12.4f} "
          f"{result.combined_aggregate.macro_f1:>12.4f}")
    print(f"{'Weighted F1':<30} {result.endorsed_aggregate.weighted_f1:>12.4f} "
          f"{result.conflict_aggregate.weighted_f1:>12.4f} "
          f"{result.combined_aggregate.weighted_f1:>12.4f}")
    print(f"{'Macro Precision':<30} {result.endorsed_aggregate.macro_precision:>12.4f} "
          f"{result.conflict_aggregate.macro_precision:>12.4f} "
          f"{result.combined_aggregate.macro_precision:>12.4f}")
    print(f"{'Macro Recall':<30} {result.endorsed_aggregate.macro_recall:>12.4f} "
          f"{result.conflict_aggregate.macro_recall:>12.4f} "
          f"{result.combined_aggregate.macro_recall:>12.4f}")
    print(f"{'Categories Evaluated':<30} {result.endorsed_aggregate.categories_evaluated:>12} "
          f"{result.conflict_aggregate.categories_evaluated:>12} "
          f"{result.combined_aggregate.categories_evaluated:>12}")
    
    # Per-category breakdown
    print("\n" + "-" * 70)
    print("📈 PER-CATEGORY ENDORSED METRICS")
    print("-" * 70)
    print(f"{'Category':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for category in ANNOTATION_CATEGORIES:
        cat_result = result.per_category_endorsed.get(category)
        if cat_result:
            print(f"{category:<30} {cat_result.precision:>10.4f} "
                  f"{cat_result.recall:>10.4f} {cat_result.f1:>10.4f} "
                  f"{cat_result.support:>10}")
    
    # Conflict breakdown
    print("\n" + "-" * 70)
    print("📈 PER-CATEGORY CONFLICT METRICS")
    print("-" * 70)
    print(f"{'Category':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for category in ANNOTATION_CATEGORIES:
        cat_result = result.per_category_conflict.get(category)
        if cat_result and cat_result.support > 0:
            print(f"{category:<30} {cat_result.precision:>10.4f} "
                  f"{cat_result.recall:>10.4f} {cat_result.f1:>10.4f} "
                  f"{cat_result.support:>10}")
    
    # Step 5: Generate reports
    print(f"\n📄 Generating reports to: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    generator = ReportGenerator(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generated_files = generator.generate_all_reports([result], timestamp)
    
    print("   Generated reports:")
    for report_type, path in generated_files.items():
        print(f"   - {report_type}: {path.name}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
