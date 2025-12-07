# Tests for MetricsCalculator

import pytest
from hypothesis import given, strategies as st, assume, settings

from evaluation.models import (
    VideoAnnotation,
    PredictionResult,
    PredictionSet,
    GroundTruthDataset,
)
from evaluation.metrics import (
    MetricsCalculator,
    AlignmentResult,
    CategoryResult,
    AggregateResult,
    ANNOTATION_CATEGORIES,
)


# Strategy for valid annotation values
valid_annotation_values = st.integers(min_value=-1, max_value=2)

# Strategy for complete annotation dictionaries
complete_annotations = st.fixed_dictionaries(
    {category: valid_annotation_values for category in ANNOTATION_CATEGORIES}
)

# Strategy for video IDs (alphanumeric)
video_id_strategy = st.text(
    min_size=1, max_size=30, 
    alphabet=st.characters(whitelist_categories=('L', 'N'))
)


class TestMetricsCalculatorAlignment:
    """Tests for prediction-ground truth alignment."""
    
    @given(
        video_ids=st.lists(video_id_strategy, min_size=1, max_size=10, unique=True),
        annotations=st.lists(complete_annotations, min_size=1, max_size=10),
    )
    def test_perfect_alignment(self, video_ids, annotations):
        """
        Property: When predictions and ground truth have identical video IDs,
        all should be matched.
        """
        # Make lists same length
        min_len = min(len(video_ids), len(annotations))
        video_ids = video_ids[:min_len]
        annotations = annotations[:min_len]
        
        # Create ground truth
        gt_videos = [
            VideoAnnotation(
                video_id=vid,
                video_uri=f"gs://bucket/{vid}.mp4",
                script_uri=f"gs://bucket/scripts/{vid}.txt",
                annotations=ann,
                has_sound=True,
            )
            for vid, ann in zip(video_ids, annotations)
        ]
        
        ground_truth = GroundTruthDataset(
            videos=gt_videos,
            total_count=len(gt_videos),
            valid_count=len(gt_videos),
        )
        
        # Create predictions with same video IDs
        predictions = [
            PredictionResult(
                video_id=vid,
                predictions=ann,
                success=True,
                inference_time=1.0,
            )
            for vid, ann in zip(video_ids, annotations)
        ]
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=predictions,
            total_count=len(predictions),
            success_count=len(predictions),
            failure_count=0,
        )
        
        # Align
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        
        # All should match
        assert alignment.matched_count == len(video_ids)
        assert len(alignment.unmatched_predictions) == 0
        assert len(alignment.missing_predictions) == 0
    
    @given(
        gt_video_ids=st.lists(video_id_strategy, min_size=2, max_size=5, unique=True),
        pred_video_ids=st.lists(video_id_strategy, min_size=2, max_size=5, unique=True),
        annotations=complete_annotations,
    )
    def test_partial_alignment(self, gt_video_ids, pred_video_ids, annotations):
        """
        Property: Only videos present in both sets should be matched.
        """
        # Create ground truth
        gt_videos = [
            VideoAnnotation(
                video_id=vid,
                video_uri=f"gs://bucket/{vid}.mp4",
                script_uri=f"gs://bucket/scripts/{vid}.txt",
                annotations=annotations,
                has_sound=True,
            )
            for vid in gt_video_ids
        ]
        
        ground_truth = GroundTruthDataset(
            videos=gt_videos,
            total_count=len(gt_videos),
            valid_count=len(gt_videos),
        )
        
        # Create predictions with potentially different video IDs
        predictions = [
            PredictionResult(
                video_id=vid,
                predictions=annotations,
                success=True,
                inference_time=1.0,
            )
            for vid in pred_video_ids
        ]
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=predictions,
            total_count=len(predictions),
            success_count=len(predictions),
            failure_count=0,
        )
        
        # Align
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        
        # Calculate expected matches
        gt_set = set(gt_video_ids)
        pred_set = set(pred_video_ids)
        expected_matched = len(gt_set & pred_set)
        expected_unmatched = len(pred_set - gt_set)
        expected_missing = len(gt_set - pred_set)
        
        assert alignment.matched_count == expected_matched
        assert len(alignment.unmatched_predictions) == expected_unmatched
        assert len(alignment.missing_predictions) == expected_missing
    
    @given(
        username=st.text(min_size=3, max_size=10, alphabet=st.characters(whitelist_categories=('L',))),
        video_num=st.integers(min_value=1000000000, max_value=9999999999),
        annotations=complete_annotations,
    )
    def test_normalized_video_id_matching(self, username, video_num, annotations):
        """
        Property: Video IDs should match even with different formats
        (e.g., "username_12345" matches "12345").
        """
        # Ground truth has username_videoid format
        gt_video_id = f"{username}_{video_num}"
        
        gt_video = VideoAnnotation(
            video_id=gt_video_id,
            video_uri=f"gs://bucket/{gt_video_id}.mp4",
            script_uri=f"gs://bucket/scripts/{gt_video_id}.txt",
            annotations=annotations,
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video],
            total_count=1,
            valid_count=1,
        )
        
        # Prediction has just the numeric ID
        pred_video_id = str(video_num)
        
        prediction = PredictionResult(
            video_id=pred_video_id,
            predictions=annotations,
            success=True,
            inference_time=1.0,
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[prediction],
            total_count=1,
            success_count=1,
            failure_count=0,
        )
        
        # Align
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        
        # Should match via normalized ID
        assert alignment.matched_count == 1
        assert len(alignment.unmatched_predictions) == 0
    
    @given(
        video_ids=st.lists(video_id_strategy, min_size=2, max_size=5, unique=True),
        annotations=complete_annotations,
    )
    def test_failed_predictions_excluded(self, video_ids, annotations):
        """
        Property: Failed predictions should not be included in alignment.
        """
        # Create ground truth
        gt_videos = [
            VideoAnnotation(
                video_id=vid,
                video_uri=f"gs://bucket/{vid}.mp4",
                script_uri=f"gs://bucket/scripts/{vid}.txt",
                annotations=annotations,
                has_sound=True,
            )
            for vid in video_ids
        ]
        
        ground_truth = GroundTruthDataset(
            videos=gt_videos,
            total_count=len(gt_videos),
            valid_count=len(gt_videos),
        )
        
        # Create predictions - first one fails, rest succeed
        predictions = []
        for i, vid in enumerate(video_ids):
            predictions.append(PredictionResult(
                video_id=vid,
                predictions=annotations if i > 0 else {},
                success=(i > 0),
                error_message="Test error" if i == 0 else None,
                inference_time=1.0,
            ))
        
        success_count = len(video_ids) - 1
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=predictions,
            total_count=len(predictions),
            success_count=success_count,
            failure_count=1,
            failed_video_ids=[video_ids[0]],
        )
        
        # Align
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        
        # Only successful predictions should be matched
        assert alignment.matched_count == success_count
        # The failed prediction's video should be in missing (no prediction matched)
        assert video_ids[0] in alignment.missing_predictions


class TestMetricsCalculatorMetrics:
    """Tests for metrics calculation."""
    
    def test_perfect_predictions_weighted_f1_is_one(self):
        """
        Property: When predictions exactly match ground truth, weighted F1 should be 1.0.
        Macro F1 may be lower due to categories with zero support.
        """
        video_id = "test_video"
        
        # Create annotations with all value types represented
        annotations = {cat: 1 for cat in ANNOTATION_CATEGORIES}  # All endorsed
        annotations["Self_Direction_Thought"] = -1  # One conflict
        annotations["Self_Direction_Action"] = 0   # One absent
        
        gt_video = VideoAnnotation(
            video_id=video_id,
            video_uri="gs://bucket/test.mp4",
            script_uri="gs://bucket/scripts/test.txt",
            annotations=annotations,
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video],
            total_count=1,
            valid_count=1,
        )
        
        prediction = PredictionResult(
            video_id=video_id,
            predictions=annotations,
            success=True,
            inference_time=1.0,
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[prediction],
            total_count=1,
            success_count=1,
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # Weighted F1 should be 1.0 for perfect match (only considers categories with support)
        assert result.endorsed_aggregate.weighted_f1 == 1.0
        assert result.conflict_aggregate.weighted_f1 == 1.0
        assert result.combined_aggregate.weighted_f1 == 1.0
        
        # Combined macro should also be 1.0 (exact match in all categories)
        assert result.combined_aggregate.macro_f1 == 1.0
    
    @given(annotations=complete_annotations)
    def test_perfect_match_combined_f1_is_one(self, annotations):
        """
        Property: When predictions exactly match ground truth, combined F1 should be 1.0.
        """
        video_id = "test_video"
        
        gt_video = VideoAnnotation(
            video_id=video_id,
            video_uri="gs://bucket/test.mp4",
            script_uri="gs://bucket/scripts/test.txt",
            annotations=annotations,
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video],
            total_count=1,
            valid_count=1,
        )
        
        prediction = PredictionResult(
            video_id=video_id,
            predictions=annotations,
            success=True,
            inference_time=1.0,
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[prediction],
            total_count=1,
            success_count=1,
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # Combined F1 should always be 1.0 for perfect match
        assert result.combined_aggregate.macro_f1 == 1.0
    
    def test_empty_predictions_returns_zero_metrics(self):
        """
        Property: Empty predictions should result in zero metrics.
        """
        gt_video = VideoAnnotation(
            video_id="test_video",
            video_uri="gs://bucket/test.mp4",
            script_uri="gs://bucket/scripts/test.txt",
            annotations={cat: 1 for cat in ANNOTATION_CATEGORIES},
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video],
            total_count=1,
            valid_count=1,
        )
        
        # Empty predictions (all failed)
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[],
            total_count=0,
            success_count=0,
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # No matches, so zero metrics
        assert result.matched_with_ground_truth == 0
        assert result.endorsed_aggregate.macro_f1 == 0.0


# **Feature: model-evaluation-module, Property 8: Endorsed metrics filtering**
# **Validates: Requirements 3.2, 4.1**
class TestEndorsedMetricsFiltering:
    """Tests for endorsed value metrics calculation."""
    
    @given(
        num_videos=st.integers(min_value=3, max_value=10),
    )
    def test_endorsed_only_counts_positive_values(self, num_videos):
        """
        Property: Endorsed metrics should only treat values 1 and 2 as positive class.
        Values 0 and -1 should be treated as negative class.
        """
        # Create ground truth with mix of values
        gt_videos = []
        for i in range(num_videos):
            annotations = {}
            for j, cat in enumerate(ANNOTATION_CATEGORIES):
                # Vary values: some endorsed (1,2), some not (0,-1)
                if (i + j) % 4 == 0:
                    annotations[cat] = 1  # Endorsed
                elif (i + j) % 4 == 1:
                    annotations[cat] = 2  # Endorsed
                elif (i + j) % 4 == 2:
                    annotations[cat] = 0  # Absent
                else:
                    annotations[cat] = -1  # Conflict
            
            gt_videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=gt_videos,
            total_count=len(gt_videos),
            valid_count=len(gt_videos),
        )
        
        # Create perfect predictions
        predictions = [
            PredictionResult(
                video_id=v.video_id,
                predictions=v.annotations.copy(),
                success=True,
                inference_time=1.0,
            )
            for v in gt_videos
        ]
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=predictions,
            total_count=len(predictions),
            success_count=len(predictions),
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        endorsed_metrics = calculator.calculate_category_metrics(alignment, "endorsed")
        
        # For each category, verify:
        # - True positives = count where both GT and pred are in {1, 2}
        # - False positives = count where GT not in {1, 2} but pred in {1, 2}
        # - False negatives = count where GT in {1, 2} but pred not in {1, 2}
        for cat in ANNOTATION_CATEGORIES:
            cat_metrics = endorsed_metrics[cat]
            
            # Count expected values manually
            expected_tp = 0
            expected_fp = 0
            expected_fn = 0
            
            for v in gt_videos:
                gt_val = v.annotations[cat]
                pred_val = v.annotations[cat]  # Perfect match
                
                gt_endorsed = gt_val in {1, 2}
                pred_endorsed = pred_val in {1, 2}
                
                if gt_endorsed and pred_endorsed:
                    expected_tp += 1
                elif not gt_endorsed and pred_endorsed:
                    expected_fp += 1
                elif gt_endorsed and not pred_endorsed:
                    expected_fn += 1
            
            assert cat_metrics.true_positives == expected_tp, (
                f"Category {cat}: expected TP={expected_tp}, got {cat_metrics.true_positives}"
            )
            assert cat_metrics.false_positives == expected_fp
            assert cat_metrics.false_negatives == expected_fn
    
    def test_endorsed_treats_1_and_2_as_equivalent(self):
        """
        Property: Values 1 and 2 should both be treated as endorsed (positive class).
        A prediction of 1 when ground truth is 2 should count as true positive.
        """
        # Ground truth with value 2
        gt_annotations = {cat: 2 for cat in ANNOTATION_CATEGORIES}
        gt_video = VideoAnnotation(
            video_id="test",
            video_uri="gs://bucket/test.mp4",
            script_uri="gs://bucket/scripts/test.txt",
            annotations=gt_annotations,
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video],
            total_count=1,
            valid_count=1,
        )
        
        # Prediction with value 1 (different endorsed value)
        pred_annotations = {cat: 1 for cat in ANNOTATION_CATEGORIES}
        prediction = PredictionResult(
            video_id="test",
            predictions=pred_annotations,
            success=True,
            inference_time=1.0,
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[prediction],
            total_count=1,
            success_count=1,
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # Should be perfect F1 for endorsed (1 and 2 are equivalent)
        assert result.endorsed_aggregate.macro_f1 == 1.0
        assert result.endorsed_aggregate.weighted_f1 == 1.0


# **Feature: model-evaluation-module, Property 9: Conflict metrics filtering**
# **Validates: Requirements 3.3, 4.2**
class TestConflictMetricsFiltering:
    """Tests for conflict value metrics calculation."""
    
    @given(
        num_videos=st.integers(min_value=3, max_value=10),
    )
    def test_conflict_only_counts_negative_one(self, num_videos):
        """
        Property: Conflict metrics should only treat value -1 as positive class.
        Values 0, 1, 2 should be treated as negative class.
        """
        # Create ground truth with mix of values
        gt_videos = []
        for i in range(num_videos):
            annotations = {}
            for j, cat in enumerate(ANNOTATION_CATEGORIES):
                # Vary values
                if (i + j) % 4 == 0:
                    annotations[cat] = -1  # Conflict (positive)
                elif (i + j) % 4 == 1:
                    annotations[cat] = 0   # Absent (negative)
                elif (i + j) % 4 == 2:
                    annotations[cat] = 1   # Endorsed (negative)
                else:
                    annotations[cat] = 2   # Endorsed (negative)
            
            gt_videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=gt_videos,
            total_count=len(gt_videos),
            valid_count=len(gt_videos),
        )
        
        # Create perfect predictions
        predictions = [
            PredictionResult(
                video_id=v.video_id,
                predictions=v.annotations.copy(),
                success=True,
                inference_time=1.0,
            )
            for v in gt_videos
        ]
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=predictions,
            total_count=len(predictions),
            success_count=len(predictions),
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        conflict_metrics = calculator.calculate_category_metrics(alignment, "conflict")
        
        # For each category, verify only -1 is counted as positive
        for cat in ANNOTATION_CATEGORIES:
            cat_metrics = conflict_metrics[cat]
            
            expected_tp = 0
            for v in gt_videos:
                if v.annotations[cat] == -1:
                    expected_tp += 1
            
            assert cat_metrics.true_positives == expected_tp
            assert cat_metrics.support == expected_tp  # Support = count of -1 in GT
    
    def test_conflict_distinguishes_from_other_values(self):
        """
        Property: Only -1 should be positive class for conflict metrics.
        Predicting 0 or 1 when GT is -1 should be false negative.
        """
        # Ground truth with all -1 (conflict)
        gt_annotations = {cat: -1 for cat in ANNOTATION_CATEGORIES}
        gt_video = VideoAnnotation(
            video_id="test",
            video_uri="gs://bucket/test.mp4",
            script_uri="gs://bucket/scripts/test.txt",
            annotations=gt_annotations,
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video],
            total_count=1,
            valid_count=1,
        )
        
        # Prediction with 0 (wrong - should be -1)
        pred_annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
        prediction = PredictionResult(
            video_id="test",
            predictions=pred_annotations,
            success=True,
            inference_time=1.0,
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[prediction],
            total_count=1,
            success_count=1,
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # Should have 0 F1 for conflict (all false negatives)
        assert result.conflict_aggregate.macro_f1 == 0.0
        
        # Check that all categories have false negatives
        for cat in ANNOTATION_CATEGORIES:
            cat_result = result.per_category_conflict[cat]
            assert cat_result.true_positives == 0
            assert cat_result.false_negatives == 1


# **Feature: model-evaluation-module, Property 11: Macro F1 calculation**
# **Validates: Requirements 5.1, 5.3, 5.4, 5.5**
class TestMacroF1Calculation:
    """Tests for macro F1 score calculation."""
    
    def test_macro_f1_is_unweighted_mean(self):
        """
        Property: Macro F1 should be the unweighted mean of all category F1 scores.
        """
        # Create dataset where different categories have different performance
        gt_videos = []
        pred_list = []
        
        for i in range(10):
            annotations = {}
            predictions_dict = {}
            
            for j, cat in enumerate(ANNOTATION_CATEGORIES):
                # First half of categories: perfect match
                # Second half: all wrong
                if j < len(ANNOTATION_CATEGORIES) // 2:
                    annotations[cat] = 1
                    predictions_dict[cat] = 1
                else:
                    annotations[cat] = 1
                    predictions_dict[cat] = 0  # Wrong prediction
            
            gt_videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
            
            pred_list.append(PredictionResult(
                video_id=f"video_{i}",
                predictions=predictions_dict,
                success=True,
                inference_time=1.0,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=gt_videos,
            total_count=len(gt_videos),
            valid_count=len(gt_videos),
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=pred_list,
            total_count=len(pred_list),
            success_count=len(pred_list),
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        endorsed_metrics = calculator.calculate_category_metrics(alignment, "endorsed")
        aggregate = calculator.calculate_aggregate_metrics(endorsed_metrics)
        
        # Calculate expected macro F1 manually
        f1_scores = [m.f1 for m in endorsed_metrics.values()]
        expected_macro_f1 = sum(f1_scores) / len(f1_scores)
        
        assert abs(aggregate.macro_f1 - expected_macro_f1) < 1e-10


# **Feature: model-evaluation-module, Property 12: Weighted F1 calculation**
# **Validates: Requirements 5.2, 5.6, 5.7, 5.8**
class TestWeightedF1Calculation:
    """Tests for weighted F1 score calculation."""
    
    def test_weighted_f1_uses_support(self):
        """
        Property: Weighted F1 should weight each category by its support.
        weighted_f1 = sum(f1 * support) / sum(support)
        """
        # Create dataset with varying support per category
        gt_videos = []
        pred_list = []
        
        for i in range(20):
            annotations = {}
            predictions_dict = {}
            
            for j, cat in enumerate(ANNOTATION_CATEGORIES):
                # Categories 0-4: high support (always endorsed)
                # Categories 5-9: medium support (sometimes endorsed)
                # Rest: low support (rarely endorsed)
                if j < 5:
                    annotations[cat] = 1  # High support
                    predictions_dict[cat] = 1  # Perfect match
                elif j < 10 and i < 10:
                    annotations[cat] = 1  # Medium support
                    predictions_dict[cat] = 0 if i < 5 else 1  # Half wrong
                else:
                    annotations[cat] = 0  # Low support (not endorsed)
                    predictions_dict[cat] = 0
            
            gt_videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
            
            pred_list.append(PredictionResult(
                video_id=f"video_{i}",
                predictions=predictions_dict,
                success=True,
                inference_time=1.0,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=gt_videos,
            total_count=len(gt_videos),
            valid_count=len(gt_videos),
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=pred_list,
            total_count=len(pred_list),
            success_count=len(pred_list),
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        alignment = calculator.align_predictions(prediction_set)
        endorsed_metrics = calculator.calculate_category_metrics(alignment, "endorsed")
        aggregate = calculator.calculate_aggregate_metrics(endorsed_metrics)
        
        # Calculate expected weighted F1 manually
        total_support = sum(m.support for m in endorsed_metrics.values())
        if total_support > 0:
            expected_weighted_f1 = sum(
                m.f1 * m.support for m in endorsed_metrics.values()
            ) / total_support
        else:
            expected_weighted_f1 = 0.0
        
        assert abs(aggregate.weighted_f1 - expected_weighted_f1) < 1e-10
    
    def test_weighted_f1_ignores_zero_support_categories(self):
        """
        Property: Categories with zero support should not affect weighted F1.
        """
        # Create dataset where most categories have no positive examples
        annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}  # All absent
        annotations[ANNOTATION_CATEGORIES[0]] = 1  # Only one category has endorsement
        
        gt_video = VideoAnnotation(
            video_id="test",
            video_uri="gs://bucket/test.mp4",
            script_uri="gs://bucket/scripts/test.txt",
            annotations=annotations,
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video],
            total_count=1,
            valid_count=1,
        )
        
        # Perfect prediction
        prediction = PredictionResult(
            video_id="test",
            predictions=annotations.copy(),
            success=True,
            inference_time=1.0,
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[prediction],
            total_count=1,
            success_count=1,
            failure_count=0,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # Weighted F1 should be 1.0 because the only category with support
        # has perfect F1
        assert result.endorsed_aggregate.weighted_f1 == 1.0


class TestRareCategoryFiltering:
    """Tests for rare category filtering based on frequency threshold."""
    
    def test_categories_below_threshold_excluded_from_aggregate(self):
        """
        Property: Categories with ground truth frequency below min_frequency_threshold
        should be excluded from aggregate metrics calculation.
        
        Validates that rare categories (appearing in few videos) don't skew
        overall model evaluation metrics.
        """
        # Create ground truth with one common category and one rare category
        # Using actual category names from ANNOTATION_CATEGORIES
        # 3 videos with "Achievement" (frequency = 1.0), 1 video with "Hedonism" (frequency = 0.33)
        videos = []
        
        # All 3 videos have Achievement endorsed
        for i in range(3):
            annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            annotations["Achievement"] = 1  # Common category - in all videos
            if i == 0:
                annotations["Hedonism"] = 1  # Rare category - only in 1/3 videos
            videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=videos,
            total_count=3,
            valid_count=3,
        )
        
        # Predictions: perfect for Achievement, wrong for Hedonism
        predictions = []
        for i, video in enumerate(videos):
            pred_annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            pred_annotations["Achievement"] = 1  # Correct
            # Hedonism: predict 1 for all videos (2 false positives)
            pred_annotations["Hedonism"] = 1
            predictions.append(PredictionResult(
                video_id=video.video_id,
                predictions=pred_annotations,
                success=True,
                inference_time=1.0,
            ))
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=predictions,
            total_count=3,
            success_count=3,
            failure_count=0,
        )
        
        # With threshold of 0.5, Hedonism (freq=0.33) should be excluded
        # Achievement (freq=1.0) should be included
        calculator_with_threshold = MetricsCalculator(
            ground_truth, 
            min_frequency_threshold=0.5
        )
        result_with_threshold = calculator_with_threshold.calculate_model_metrics(prediction_set)
        
        # Without threshold, both categories included
        calculator_no_threshold = MetricsCalculator(
            ground_truth,
            min_frequency_threshold=0.0
        )
        result_no_threshold = calculator_no_threshold.calculate_model_metrics(prediction_set)
        
        # Verify Hedonism is excluded when threshold is applied
        excluded = calculator_with_threshold.get_excluded_categories("endorsed")
        assert "Hedonism" in excluded, f"Hedonism should be excluded, got: {excluded}"
        assert "Achievement" not in excluded, f"Achievement should NOT be excluded, got: {excluded}"
        
        # With threshold: only Achievement (perfect predictions) included -> higher weighted F1
        # Without threshold: both categories included, Hedonism has false positives -> lower weighted F1
        # But weighted F1 is weighted by support, so let's check categories_evaluated instead
        assert result_with_threshold.endorsed_aggregate.categories_evaluated < result_no_threshold.endorsed_aggregate.categories_evaluated
    
    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_frequency_filtering_property(self, threshold):
        """
        Property: All excluded categories must have frequency strictly below the threshold.
        All included categories must have frequency >= threshold or have zero support.
        """
        # Create varied ground truth using valid category names
        videos = []
        for i in range(10):
            annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            # Vary which categories appear in each video
            if i < 8:  # 80% frequency
                annotations["Achievement"] = 1
            if i < 5:  # 50% frequency
                annotations["Hedonism"] = 1
            if i < 2:  # 20% frequency
                annotations["Power_Resources"] = 1
            videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=videos,
            total_count=10,
            valid_count=10,
        )
        
        calculator = MetricsCalculator(ground_truth, min_frequency_threshold=threshold)
        frequencies = calculator._category_frequencies
        excluded = calculator.get_excluded_categories("endorsed")
        
        # Verify all excluded categories are below threshold
        for cat in excluded:
            assert frequencies[cat]["endorsed"] < threshold, f"{cat} excluded but frequency {frequencies[cat]['endorsed']} >= {threshold}"
        
        # Verify all non-excluded categories with non-zero frequency are at or above threshold
        for cat in ANNOTATION_CATEGORIES:
            if cat not in excluded and frequencies[cat]["endorsed"] > 0:
                assert frequencies[cat]["endorsed"] >= threshold, f"{cat} included but frequency {frequencies[cat]['endorsed']} < {threshold}"


class TestFailedPredictionExclusion:
    """Tests for excluding failed predictions from metrics calculation."""
    
    def test_failed_predictions_excluded_from_metrics(self):
        """
        Property 24: Failed predictions (success=False) should be excluded
        from metrics calculation.
        
        Validates Requirement 10.5: Only successful predictions contribute
        to evaluation metrics.
        """
        # Create ground truth using valid category name
        annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
        annotations["Achievement"] = 1
        
        gt_video_1 = VideoAnnotation(
            video_id="video_1",
            video_uri="gs://bucket/video_1.mp4",
            script_uri="gs://bucket/scripts/video_1.txt",
            annotations=annotations.copy(),
            has_sound=True,
        )
        gt_video_2 = VideoAnnotation(
            video_id="video_2",
            video_uri="gs://bucket/video_2.mp4",
            script_uri="gs://bucket/scripts/video_2.txt",
            annotations=annotations.copy(),
            has_sound=True,
        )
        
        ground_truth = GroundTruthDataset(
            videos=[gt_video_1, gt_video_2],
            total_count=2,
            valid_count=2,
        )
        
        # Create predictions: one successful (correct), one failed (wrong but shouldn't count)
        successful_pred = PredictionResult(
            video_id="video_1",
            predictions=annotations.copy(),  # Correct prediction
            success=True,
            inference_time=1.0,
        )
        
        wrong_annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
        wrong_annotations["Achievement"] = 0  # Wrong - should be 1
        wrong_annotations["Hedonism"] = 1  # Wrong - should be 0
        
        failed_pred = PredictionResult(
            video_id="video_2",
            predictions=wrong_annotations,  # Wrong prediction
            success=False,  # But marked as failed
            inference_time=0.5,
            error_message="API timeout",
        )
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=[successful_pred, failed_pred],
            total_count=2,
            success_count=1,
            failure_count=1,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # Should only have 1 matched (the successful prediction)
        assert result.matched_with_ground_truth == 1
        
        # Metrics should reflect only the successful (correct) prediction
        # Perfect F1 for Achievement since the only matched prediction is correct
        achievement_result = result.per_category_endorsed.get("Achievement")
        assert achievement_result is not None
        assert achievement_result.f1 == 1.0
    
    @given(
        num_successful=st.integers(min_value=1, max_value=5),
        num_failed=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=30)
    def test_failed_prediction_count_property(self, num_successful, num_failed):
        """
        Property: The number of matched predictions should equal the number
        of successful predictions that have matching ground truth videos.
        """
        # Create ground truth for all videos using valid category name
        total = num_successful + num_failed
        videos = []
        for i in range(total):
            annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            annotations["Achievement"] = 1
            videos.append(VideoAnnotation(
                video_id=f"video_{i}",
                video_uri=f"gs://bucket/video_{i}.mp4",
                script_uri=f"gs://bucket/scripts/video_{i}.txt",
                annotations=annotations,
                has_sound=True,
            ))
        
        ground_truth = GroundTruthDataset(
            videos=videos,
            total_count=total,
            valid_count=total,
        )
        
        # Create predictions: first num_successful are successful, rest are failed
        predictions = []
        for i in range(total):
            is_success = i < num_successful
            pred_annotations = {cat: 0 for cat in ANNOTATION_CATEGORIES}
            pred_annotations["Achievement"] = 1
            predictions.append(PredictionResult(
                video_id=f"video_{i}",
                predictions=pred_annotations,
                success=is_success,
                inference_time=1.0,
                error_message=None if is_success else "API error",
            ))
        
        prediction_set = PredictionSet(
            model_name="test_model",
            predictions=predictions,
            total_count=total,
            success_count=num_successful,
            failure_count=num_failed,
        )
        
        calculator = MetricsCalculator(ground_truth)
        result = calculator.calculate_model_metrics(prediction_set)
        
        # Only successful predictions should be matched
        assert result.matched_with_ground_truth == num_successful
        # Missing count should be the number of failed predictions (GT videos without matching successful prediction)
        assert result.missing_count == num_failed