"""
Quick test script to verify real CSV data loading compatibility.

Tests that:
1. Ground truth CSV loads correctly with normalized video IDs
2. Gemini predictions CSV loads correctly with normalized video IDs  
3. Video IDs match between ground truth and predictions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation import GroundTruthLoader, normalize_video_id
from evaluation.adapters.gemini_adapter import GeminiAdapter


def test_real_data_loading():
    """Test loading real CSV data files."""
    
    # File paths
    ground_truth_path = r"C:\Users\User\Desktop\deValuating_TikTok - freeze copy\Data\cleaned_groundtruth_values_only.csv"
    predictions_path = r"C:\Users\User\Desktop\deValuating_TikTok - freeze copy\Data\gemini_script2value_raw.csv"
    
    print("=" * 60)
    print("Testing Real Data Loading Compatibility")
    print("=" * 60)
    
    # Test 1: Load ground truth
    print("\n1. Loading Ground Truth...")
    try:
        loader = GroundTruthLoader(ground_truth_path)
        dataset = loader.load()
        print(f"   ✓ Loaded {dataset.valid_count} valid videos from {dataset.total_count} total")
        
        if dataset.validation_errors:
            print(f"   ⚠ {len(dataset.validation_errors)} validation errors")
            for error in dataset.validation_errors[:3]:
                print(f"     - {error}")
            if len(dataset.validation_errors) > 3:
                print(f"     ... and {len(dataset.validation_errors) - 3} more")
        
        # Show sample video IDs
        print(f"\n   Sample normalized video IDs from ground truth:")
        for video in dataset.videos[:3]:
            print(f"     - {video.video_id}")
            
    except Exception as e:
        print(f"   ✗ Error loading ground truth: {e}")
        return False
    
    # Test 2: Load predictions
    print("\n2. Loading Gemini Predictions...")
    try:
        prediction_set = GeminiAdapter.load_predictions_from_csv(
            predictions_path, 
            model_name="gemini-1.5-pro"
        )
        print(f"   ✓ Loaded {prediction_set.success_count} successful predictions")
        print(f"   ✓ {prediction_set.failure_count} failed predictions")
        
        # Show sample prediction IDs
        print(f"\n   Sample normalized video IDs from predictions:")
        for pred in prediction_set.predictions[:3]:
            print(f"     - {pred.video_id} (success={pred.success})")
            if pred.success:
                # Show one sample value
                sample_key = list(pred.predictions.keys())[0]
                print(f"       Sample: {sample_key} = {pred.predictions[sample_key]}")
                
    except Exception as e:
        print(f"   ✗ Error loading predictions: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Check ID matching
    print("\n3. Checking Video ID Matching...")
    gt_ids = set(v.video_id for v in dataset.videos)
    pred_ids = set(p.video_id for p in prediction_set.predictions)
    
    matched = gt_ids & pred_ids
    gt_only = gt_ids - pred_ids
    pred_only = pred_ids - gt_ids
    
    print(f"   Ground truth IDs: {len(gt_ids)}")
    print(f"   Prediction IDs: {len(pred_ids)}")
    print(f"   ✓ Matched IDs: {len(matched)}")
    print(f"   ⚠ Ground truth only: {len(gt_only)}")
    print(f"   ⚠ Predictions only: {len(pred_only)}")
    
    if gt_only:
        print(f"\n   Sample unmatched ground truth IDs:")
        for vid in list(gt_only)[:3]:
            print(f"     - {vid}")
    
    if pred_only:
        print(f"\n   Sample unmatched prediction IDs:")
        for vid in list(pred_only)[:3]:
            print(f"     - {vid}")
    
    match_rate = len(matched) / max(len(gt_ids), 1) * 100
    print(f"\n   Match rate: {match_rate:.1f}%")
    
    print("\n" + "=" * 60)
    if match_rate >= 90:
        print("✓ SUCCESS: Real data loading compatibility verified!")
        return True
    elif match_rate >= 50:
        print("⚠ PARTIAL: Some videos matched, some didn't")
        return True
    else:
        print("✗ FAILURE: Low match rate between ground truth and predictions")
        return False


if __name__ == "__main__":
    success = test_real_data_loading()
    sys.exit(0 if success else 1)
