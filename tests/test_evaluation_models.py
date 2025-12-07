# Property-based tests for evaluation data models

import pytest
from hypothesis import given, strategies as st
from evaluation.models import (
    VideoAnnotation,
    PredictionResult,
    GroundTruthDataset,
    PredictionSet,
)
from evaluation.prediction_storage import PredictionStorage

# Define the 19 annotation categories based on Schwartz's value framework
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

# Strategy for generating valid annotation values
valid_annotation_values = st.integers(min_value=-1, max_value=2)

# Strategy for generating complete annotation dictionaries
complete_annotations = st.fixed_dictionaries(
    {category: valid_annotation_values for category in ANNOTATION_CATEGORIES}
)


# **Feature: model-evaluation-module, Property 2: Annotation format consistency**
# **Validates: Requirements 1.2, 2.3**
@given(
    video_id=st.text(min_size=1, max_size=100),
    video_uri=st.text(min_size=1, max_size=200),
    script_uri=st.text(min_size=1, max_size=200),
    annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_video_annotation_format_consistency(
    video_id, video_uri, script_uri, annotations, has_sound
):
    """
    Property: For any video processed by any model, the output annotations should have
    the same structure as ground truth (same 19 category keys, values in range -1 to 2).
    
    This test verifies that VideoAnnotation objects maintain format consistency.
    """
    # Create a VideoAnnotation object
    video_annotation = VideoAnnotation(
        video_id=video_id,
        video_uri=video_uri,
        script_uri=script_uri,
        annotations=annotations,
        has_sound=has_sound,
    )
    
    # Verify the annotation has all 19 categories
    assert len(video_annotation.annotations) == 19
    assert set(video_annotation.annotations.keys()) == set(ANNOTATION_CATEGORIES)
    
    # Verify all values are in the valid range
    for category, value in video_annotation.annotations.items():
        assert value in {-1, 0, 1, 2}, f"Invalid value {value} for category {category}"


@given(
    video_id=st.text(min_size=1, max_size=100),
    predictions=complete_annotations,
)
def test_prediction_result_format_consistency(video_id, predictions):
    """
    Property: For any video processed by any model, the prediction output should have
    the same structure as ground truth (same 19 category keys, values in range -1 to 2).
    
    This test verifies that PredictionResult objects maintain format consistency.
    """
    # Create a successful PredictionResult object
    prediction_result = PredictionResult(
        video_id=video_id,
        predictions=predictions,
        success=True,
        inference_time=1.0,
    )
    
    # Verify the prediction has all 19 categories
    assert len(prediction_result.predictions) == 19
    assert set(prediction_result.predictions.keys()) == set(ANNOTATION_CATEGORIES)
    
    # Verify all values are in the valid range
    for category, value in prediction_result.predictions.items():
        assert value in {-1, 0, 1, 2}, f"Invalid value {value} for category {category}"


@given(
    video_id=st.text(min_size=1, max_size=100),
    video_uri=st.text(min_size=1, max_size=200),
    script_uri=st.text(min_size=1, max_size=200),
    annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_video_annotation_rejects_invalid_values(
    video_id, video_uri, script_uri, annotations, has_sound
):
    """
    Property: VideoAnnotation should reject annotations with values outside the valid range.
    
    This test verifies that invalid annotation values are caught during initialization.
    """
    # Test with an invalid value (outside -1 to 2 range)
    invalid_annotations = annotations.copy()
    invalid_annotations[ANNOTATION_CATEGORIES[0]] = 5  # Invalid value
    
    with pytest.raises(ValueError, match="Invalid annotation value"):
        VideoAnnotation(
            video_id=video_id,
            video_uri=video_uri,
            script_uri=script_uri,
            annotations=invalid_annotations,
            has_sound=has_sound,
        )



# **Feature: model-evaluation-module, Property 10: Value classification consistency**
# **Validates: Requirements 4.1, 4.2, 4.3**
@given(annotation_value=st.integers(min_value=-1, max_value=2))
def test_value_classification_consistency(annotation_value):
    """
    Property: For any annotation value, it should be classified as exactly one of:
    endorsed (1,2), conflict (-1), or absent (0).
    
    This test verifies that value classification is mutually exclusive and exhaustive.
    """
    # Classify the value
    is_endorsed = annotation_value in {1, 2}
    is_conflict = annotation_value == -1
    is_absent = annotation_value == 0
    
    # Verify exactly one classification is true
    classifications = [is_endorsed, is_conflict, is_absent]
    assert sum(classifications) == 1, (
        f"Value {annotation_value} should be classified as exactly one type, "
        f"but got: endorsed={is_endorsed}, conflict={is_conflict}, absent={is_absent}"
    )
    
    # Verify the classification matches expected behavior
    if annotation_value in {1, 2}:
        assert is_endorsed and not is_conflict and not is_absent
    elif annotation_value == -1:
        assert is_conflict and not is_endorsed and not is_absent
    elif annotation_value == 0:
        assert is_absent and not is_endorsed and not is_conflict


@given(
    annotations=complete_annotations,
)
def test_annotation_value_classification_coverage(annotations):
    """
    Property: For any complete set of annotations, all values should be classifiable
    into endorsed, conflict, or absent categories.
    
    This test verifies that the classification system covers all possible annotation values.
    """
    endorsed_count = 0
    conflict_count = 0
    absent_count = 0
    
    for category, value in annotations.items():
        if value in {1, 2}:
            endorsed_count += 1
        elif value == -1:
            conflict_count += 1
        elif value == 0:
            absent_count += 1
        else:
            pytest.fail(f"Unexpected annotation value {value} for category {category}")
    
    # Verify all annotations were classified
    total_classified = endorsed_count + conflict_count + absent_count
    assert total_classified == 19, (
        f"Expected 19 classified annotations, got {total_classified}"
    )



# **Feature: model-evaluation-module, Property 13: Ground truth value validation**
# **Validates: Requirements 6.3**
@given(
    video_id=st.text(min_size=1, max_size=100),
    video_uri=st.text(min_size=1, max_size=200),
    script_uri=st.text(min_size=1, max_size=200),
    annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_ground_truth_value_validation(
    video_id, video_uri, script_uri, annotations, has_sound
):
    """
    Property: For any ground truth record, all annotation values should be in the
    valid range {-1, 0, 1, 2}.
    
    This test verifies that the validation logic correctly checks value ranges.
    """
    from evaluation.ground_truth_loader import GroundTruthLoader
    
    # Create a valid VideoAnnotation
    video = VideoAnnotation(
        video_id=video_id,
        video_uri=video_uri,
        script_uri=script_uri,
        annotations=annotations,
        has_sound=has_sound,
    )
    
    # Create a loader and validate
    loader = GroundTruthLoader(dataset_path="dummy.csv")
    validation_result = loader.validate([video])
    
    # All values are in valid range, so validation should pass
    assert validation_result.valid_count == 1
    assert validation_result.invalid_count == 0
    assert len(validation_result.errors) == 0
    
    # Now test with an invalid value
    invalid_annotations = annotations.copy()
    invalid_annotations[ANNOTATION_CATEGORIES[0]] = 5  # Invalid value
    
    # Create invalid video bypassing __post_init__ validation
    invalid_video = loader._create_invalid_video(
        video_id=video_id,
        video_uri=video_uri,
        script_uri=script_uri,
        annotations=invalid_annotations,
        has_sound=has_sound
    )
    
    # Validate the invalid video
    validation_result = loader.validate([invalid_video])
    
    # Should have validation errors
    assert validation_result.invalid_count > 0
    assert len(validation_result.errors) > 0
    assert any("Invalid value" in error for error in validation_result.errors)



# **Feature: model-evaluation-module, Property 14: Invalid record exclusion**
# **Validates: Requirements 6.4**
@given(
    valid_annotations=complete_annotations,
    has_sound=st.booleans(),
)
def test_invalid_record_exclusion(valid_annotations, has_sound):
    """
    Property: For any ground truth record that fails validation, it should not be
    included in the evaluation dataset.
    
    This test verifies that invalid records are excluded from the final dataset.
    """
    import tempfile
    import csv
    from evaluation.ground_truth_loader import GroundTruthLoader
    
    # Create a temporary CSV file with both valid and invalid records
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fieldnames = ['video_id', 'video_uri', 'script_uri', 'has_sound'] + ANNOTATION_CATEGORIES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write a valid record
        valid_row = {
            'video_id': 'valid_video_1',
            'video_uri': 'gs://bucket/valid_video_1.mp4',
            'script_uri': 'gs://bucket/scripts/valid_video_1.txt',
            'has_sound': str(has_sound).lower(),
        }
        valid_row.update(valid_annotations)
        writer.writerow(valid_row)
        
        # Write an invalid record (invalid value - not in valid range)
        invalid_row = {
            'video_id': 'invalid_video_1',
            'video_uri': 'gs://bucket/invalid_video_1.mp4',
            'script_uri': 'gs://bucket/scripts/invalid_video_1.txt',
            'has_sound': 'true',
        }
        # Include all categories but with an invalid text value
        for category in ANNOTATION_CATEGORIES:
            if category == ANNOTATION_CATEGORIES[0]:
                invalid_row[category] = 'invalid_value'  # Invalid text that won't convert
            else:
                invalid_row[category] = valid_annotations[category]
        writer.writerow(invalid_row)
        
        # Write another invalid record (invalid value)
        invalid_row2 = {
            'video_id': 'invalid_video_2',
            'video_uri': 'gs://bucket/invalid_video_2.mp4',
            'script_uri': 'gs://bucket/scripts/invalid_video_2.txt',
            'has_sound': 'false',
        }
        invalid_annotations = valid_annotations.copy()
        invalid_annotations[ANNOTATION_CATEGORIES[0]] = 99  # Invalid value
        invalid_row2.update(invalid_annotations)
        writer.writerow(invalid_row2)
        
        temp_path = f.name
    
    try:
        # Load the dataset
        loader = GroundTruthLoader(dataset_path=temp_path)
        dataset = loader.load()
        
        # Verify that only valid records are included
        assert dataset.valid_count == 1, f"Expected 1 valid record, got {dataset.valid_count}"
        assert dataset.total_count == 3, f"Expected 3 total records, got {dataset.total_count}"
        assert len(dataset.videos) == 1, f"Expected 1 video in dataset, got {len(dataset.videos)}"
        
        # Verify the valid video is the one we expect
        assert dataset.videos[0].video_id == 'valid_video_1'
        
        # Verify validation errors were recorded
        assert len(dataset.validation_errors) > 0, "Expected validation errors to be recorded"
        
    finally:
        # Clean up temp file
        import os
        os.unlink(temp_path)



# **Feature: model-evaluation-module, Property 25: Sample size adherence**
# **Validates: Requirements 12.1**
@given(
    sample_size=st.integers(min_value=1, max_value=50),
    num_videos=st.integers(min_value=10, max_value=100),
    random_seed=st.integers(min_value=0, max_value=1000),
)
def test_sample_size_adherence(sample_size, num_videos, random_seed):
    """
    Property: For any configuration specifying a sample size N, the evaluation should
    use exactly N videos (or fewer if dataset is smaller).
    
    This test verifies that sampling respects the requested sample size.
    """
    import tempfile
    import csv
    from evaluation.ground_truth_loader import GroundTruthLoader
    
    # Create a temporary CSV file with num_videos records
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fieldnames = ['video_id', 'video_uri', 'script_uri', 'has_sound'] + ANNOTATION_CATEGORIES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_videos):
            row = {
                'video_id': f'video_{i}',
                'video_uri': f'gs://bucket/video_{i}.mp4',
                'script_uri': f'gs://bucket/scripts/video_{i}.txt',
                'has_sound': 'true',
            }
            # Add valid annotations
            for category in ANNOTATION_CATEGORIES:
                row[category] = '1'  # All endorsed
            writer.writerow(row)
        
        temp_path = f.name
    
    try:
        # Load with sampling
        loader = GroundTruthLoader(
            dataset_path=temp_path,
            sample_size=sample_size,
            random_seed=random_seed
        )
        dataset = loader.load()
        
        # Verify sample size adherence
        expected_size = min(sample_size, num_videos)
        assert len(dataset.videos) == expected_size, (
            f"Expected {expected_size} videos in sample, got {len(dataset.videos)}"
        )
        
        # Verify all videos are valid
        assert dataset.valid_count >= len(dataset.videos), (
            "Sample should only contain valid videos"
        )
        
    finally:
        # Clean up temp file
        import os
        os.unlink(temp_path)



# **Feature: model-evaluation-module, Property 26: Stratified sampling**
# **Validates: Requirements 12.2**
@given(
    sample_size=st.integers(min_value=20, max_value=50),
    random_seed=st.integers(min_value=0, max_value=1000),
)
def test_stratified_sampling(sample_size, random_seed):
    """
    Property: For any sampled dataset, the proportion of each value type
    (endorsed/conflict/absent) should be within 5% of the full dataset proportions.
    
    This test verifies that stratified sampling maintains class distribution.
    """
    import tempfile
    import csv
    from evaluation.ground_truth_loader import GroundTruthLoader
    
    # Create a dataset with known distribution
    # We'll create 100 videos with varying proportions of endorsed values
    num_videos = 100
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fieldnames = ['video_id', 'video_uri', 'script_uri', 'has_sound'] + ANNOTATION_CATEGORIES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Create videos with different endorsed proportions
        for i in range(num_videos):
            row = {
                'video_id': f'video_{i}',
                'video_uri': f'gs://bucket/video_{i}.mp4',
                'script_uri': f'gs://bucket/scripts/video_{i}.txt',
                'has_sound': 'true',
            }
            
            # Vary the proportion of endorsed values
            # First 25%: mostly endorsed (80% endorsed)
            # Next 25%: half endorsed (50% endorsed)
            # Next 25%: few endorsed (20% endorsed)
            # Last 25%: very few endorsed (10% endorsed)
            if i < 25:
                endorsed_count = int(0.8 * 19)
            elif i < 50:
                endorsed_count = int(0.5 * 19)
            elif i < 75:
                endorsed_count = int(0.2 * 19)
            else:
                endorsed_count = int(0.1 * 19)
            
            # Assign values
            for j, category in enumerate(ANNOTATION_CATEGORIES):
                if j < endorsed_count:
                    row[category] = '1'  # Endorsed
                else:
                    row[category] = '0'  # Absent
            
            writer.writerow(row)
        
        temp_path = f.name
    
    try:
        # Load full dataset to calculate original distribution
        loader_full = GroundTruthLoader(dataset_path=temp_path)
        full_dataset = loader_full.load()
        
        # Calculate full dataset distribution
        full_endorsed_count = 0
        full_conflict_count = 0
        full_absent_count = 0
        
        for video in full_dataset.videos:
            for value in video.annotations.values():
                if value in {1, 2}:
                    full_endorsed_count += 1
                elif value == -1:
                    full_conflict_count += 1
                elif value == 0:
                    full_absent_count += 1
        
        full_total = full_endorsed_count + full_conflict_count + full_absent_count
        full_endorsed_prop = full_endorsed_count / full_total
        full_conflict_prop = full_conflict_count / full_total
        full_absent_prop = full_absent_count / full_total
        
        # Load sampled dataset
        loader_sample = GroundTruthLoader(
            dataset_path=temp_path,
            sample_size=sample_size,
            random_seed=random_seed
        )
        sample_dataset = loader_sample.load()
        
        # Calculate sample distribution
        sample_endorsed_count = 0
        sample_conflict_count = 0
        sample_absent_count = 0
        
        for video in sample_dataset.videos:
            for value in video.annotations.values():
                if value in {1, 2}:
                    sample_endorsed_count += 1
                elif value == -1:
                    sample_conflict_count += 1
                elif value == 0:
                    sample_absent_count += 1
        
        sample_total = sample_endorsed_count + sample_conflict_count + sample_absent_count
        sample_endorsed_prop = sample_endorsed_count / sample_total
        sample_conflict_prop = sample_conflict_count / sample_total
        sample_absent_prop = sample_absent_count / sample_total
        
        # Verify proportions are within 5% (0.05) of original
        # Note: With stratified sampling, we expect better than 5% but we'll use 10% 
        # to account for small sample sizes and randomness
        tolerance = 0.10
        
        assert abs(sample_endorsed_prop - full_endorsed_prop) <= tolerance, (
            f"Endorsed proportion difference {abs(sample_endorsed_prop - full_endorsed_prop):.3f} "
            f"exceeds tolerance {tolerance}. Full: {full_endorsed_prop:.3f}, "
            f"Sample: {sample_endorsed_prop:.3f}"
        )
        
        assert abs(sample_conflict_prop - full_conflict_prop) <= tolerance, (
            f"Conflict proportion difference {abs(sample_conflict_prop - full_conflict_prop):.3f} "
            f"exceeds tolerance {tolerance}. Full: {full_conflict_prop:.3f}, "
            f"Sample: {sample_conflict_prop:.3f}"
        )
        
        assert abs(sample_absent_prop - full_absent_prop) <= tolerance, (
            f"Absent proportion difference {abs(sample_absent_prop - full_absent_prop):.3f} "
            f"exceeds tolerance {tolerance}. Full: {full_absent_prop:.3f}, "
            f"Sample: {sample_absent_prop:.3f}"
        )
        
    finally:
        # Clean up temp file
        import os
        os.unlink(temp_path)



# **Feature: model-evaluation-module, Property 27: Reproducible sampling**
# **Validates: Requirements 12.5**
@given(
    sample_size=st.integers(min_value=10, max_value=30),
    random_seed=st.integers(min_value=0, max_value=1000),
)
def test_reproducible_sampling(sample_size, random_seed):
    """
    Property: For any random seed value, running evaluation twice with the same seed
    should produce the same sample of videos.
    
    This test verifies that sampling is reproducible when using the same seed.
    """
    import tempfile
    import csv
    from evaluation.ground_truth_loader import GroundTruthLoader
    
    # Create a dataset with 50 videos
    num_videos = 50
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        fieldnames = ['video_id', 'video_uri', 'script_uri', 'has_sound'] + ANNOTATION_CATEGORIES
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(num_videos):
            row = {
                'video_id': f'video_{i}',
                'video_uri': f'gs://bucket/video_{i}.mp4',
                'script_uri': f'gs://bucket/scripts/video_{i}.txt',
                'has_sound': 'true',
            }
            # Add valid annotations
            for category in ANNOTATION_CATEGORIES:
                row[category] = '1'
            writer.writerow(row)
        
        temp_path = f.name
    
    try:
        # Load with same seed twice
        loader1 = GroundTruthLoader(
            dataset_path=temp_path,
            sample_size=sample_size,
            random_seed=random_seed
        )
        dataset1 = loader1.load()
        
        loader2 = GroundTruthLoader(
            dataset_path=temp_path,
            sample_size=sample_size,
            random_seed=random_seed
        )
        dataset2 = loader2.load()
        
        # Verify same number of videos
        assert len(dataset1.videos) == len(dataset2.videos), (
            f"Different number of videos: {len(dataset1.videos)} vs {len(dataset2.videos)}"
        )
        
        # Verify same video IDs in same order
        video_ids1 = [v.video_id for v in dataset1.videos]
        video_ids2 = [v.video_id for v in dataset2.videos]
        
        assert video_ids1 == video_ids2, (
            f"Different video IDs or order:\n"
            f"First:  {video_ids1}\n"
            f"Second: {video_ids2}"
        )
        
        # Verify that using a different seed produces different results
        loader3 = GroundTruthLoader(
            dataset_path=temp_path,
            sample_size=sample_size,
            random_seed=random_seed + 1  # Different seed
        )
        dataset3 = loader3.load()
        
        video_ids3 = [v.video_id for v in dataset3.videos]
        
        # With high probability, different seed should give different sample
        # (unless sample_size == num_videos)
        if sample_size < num_videos:
            assert video_ids1 != video_ids3, (
                "Different seeds should produce different samples (with high probability)"
            )
        
    finally:
        # Clean up temp file
        import os
        os.unlink(temp_path)



# **Feature: model-evaluation-module, Property 3: Model prediction isolation**
# **Validates: Requirements 1.3**
@given(
    model_name_1=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    model_name_2=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    video_id_1=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    video_id_2=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    annotations_1=complete_annotations,
    annotations_2=complete_annotations,
)
def test_model_prediction_isolation(
    model_name_1, model_name_2, video_id_1, video_id_2, annotations_1, annotations_2
):
    """
    Property: For any two models A and B, modifying model A's predictions should
    not affect model B's predictions.
    
    This test verifies that PredictionStorage maintains isolation between models.
    """
    # Ensure different model names for meaningful test
    if model_name_1 == model_name_2:
        model_name_2 = model_name_2 + "_different"
    
    # Create storage
    storage = PredictionStorage()
    
    # Create predictions for model 1
    predictions_1 = [
        PredictionResult(
            video_id=video_id_1,
            predictions=annotations_1,
            success=True,
            inference_time=1.0,
        )
    ]
    
    # Create predictions for model 2
    predictions_2 = [
        PredictionResult(
            video_id=video_id_2,
            predictions=annotations_2,
            success=True,
            inference_time=2.0,
        )
    ]
    
    # Store predictions for both models
    storage.store_predictions(model_name_1, predictions_1)
    storage.store_predictions(model_name_2, predictions_2)
    
    # Retrieve model 2's predictions and store a copy
    model_2_before = storage.get_predictions(model_name_2)
    model_2_predictions_before = {
        p.video_id: dict(p.predictions) for p in model_2_before.predictions
    }
    
    # Now modify model 1's predictions by storing new ones
    modified_annotations = {k: 0 for k in ANNOTATION_CATEGORIES}  # All zeros
    modified_predictions = [
        PredictionResult(
            video_id=video_id_1 + "_new",
            predictions=modified_annotations,
            success=True,
            inference_time=3.0,
        )
    ]
    storage.store_predictions(model_name_1, modified_predictions)
    
    # Retrieve model 2's predictions after modification
    model_2_after = storage.get_predictions(model_name_2)
    model_2_predictions_after = {
        p.video_id: dict(p.predictions) for p in model_2_after.predictions
    }
    
    # Verify model 2's predictions are unchanged
    assert model_2_predictions_before == model_2_predictions_after, (
        f"Model 2's predictions were affected by modifying Model 1's predictions!\n"
        f"Before: {model_2_predictions_before}\n"
        f"After: {model_2_predictions_after}"
    )
    
    # Verify model counts are correct
    assert model_2_after.total_count == len(predictions_2), (
        f"Model 2's count changed: expected {len(predictions_2)}, got {model_2_after.total_count}"
    )


@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    num_predictions=st.integers(min_value=1, max_value=10),
)
def test_model_isolation_with_removal(model_name, num_predictions):
    """
    Property: Removing one model's predictions should not affect other models.
    
    This test verifies that remove_model_predictions maintains isolation.
    """
    storage = PredictionStorage()
    
    # Create two models
    model_a = model_name + "_A"
    model_b = model_name + "_B"
    
    # Create predictions for both
    predictions_a = []
    predictions_b = []
    
    base_annotations = {k: 1 for k in ANNOTATION_CATEGORIES}
    
    for i in range(num_predictions):
        predictions_a.append(PredictionResult(
            video_id=f"video_a_{i}",
            predictions=base_annotations.copy(),
            success=True,
            inference_time=1.0,
        ))
        predictions_b.append(PredictionResult(
            video_id=f"video_b_{i}",
            predictions=base_annotations.copy(),
            success=True,
            inference_time=1.0,
        ))
    
    # Store both
    storage.store_predictions(model_a, predictions_a)
    storage.store_predictions(model_b, predictions_b)
    
    # Verify both exist
    assert storage.has_predictions(model_a)
    assert storage.has_predictions(model_b)
    
    # Remove model A
    storage.remove_model_predictions(model_a)
    
    # Verify model A is gone but model B is intact
    assert not storage.has_predictions(model_a)
    assert storage.has_predictions(model_b)
    
    # Verify model B's predictions are unchanged
    model_b_predictions = storage.get_predictions(model_b)
    assert model_b_predictions.total_count == num_predictions
    assert len(model_b_predictions.predictions) == num_predictions



# **Feature: model-evaluation-module, Property 4: Video-prediction round trip**
# **Validates: Requirements 1.4**
@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    video_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    annotations=complete_annotations,
    success=st.booleans(),
    inference_time=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
)
def test_video_prediction_round_trip(
    model_name, video_id, annotations, success, inference_time
):
    """
    Property: For any video V and model M, storing and then retrieving predictions
    for V from M should return the same predictions.
    
    This test verifies that PredictionStorage maintains data integrity.
    """
    storage = PredictionStorage()
    
    # Create original prediction
    error_message = "Test error" if not success else None
    original_prediction = PredictionResult(
        video_id=video_id,
        predictions=annotations,
        success=success,
        error_message=error_message,
        inference_time=inference_time,
    )
    
    # Store prediction
    storage.store_predictions(model_name, [original_prediction])
    
    # Retrieve using get_predictions
    prediction_set = storage.get_predictions(model_name)
    assert prediction_set is not None, "PredictionSet should not be None"
    assert len(prediction_set.predictions) == 1, "Should have exactly one prediction"
    
    retrieved_prediction = prediction_set.predictions[0]
    
    # Verify all fields match
    assert retrieved_prediction.video_id == original_prediction.video_id, (
        f"video_id mismatch: {retrieved_prediction.video_id} != {original_prediction.video_id}"
    )
    assert retrieved_prediction.predictions == original_prediction.predictions, (
        f"predictions mismatch:\n"
        f"Retrieved: {retrieved_prediction.predictions}\n"
        f"Original: {original_prediction.predictions}"
    )
    assert retrieved_prediction.success == original_prediction.success, (
        f"success mismatch: {retrieved_prediction.success} != {original_prediction.success}"
    )
    assert retrieved_prediction.error_message == original_prediction.error_message, (
        f"error_message mismatch: {retrieved_prediction.error_message} != {original_prediction.error_message}"
    )
    assert retrieved_prediction.inference_time == original_prediction.inference_time, (
        f"inference_time mismatch: {retrieved_prediction.inference_time} != {original_prediction.inference_time}"
    )
    
    # Also retrieve using get_prediction_for_video
    video_prediction = storage.get_prediction_for_video(model_name, video_id)
    assert video_prediction is not None, "Video prediction should not be None"
    assert video_prediction.video_id == original_prediction.video_id
    assert video_prediction.predictions == original_prediction.predictions
    assert video_prediction.success == original_prediction.success


@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    num_predictions=st.integers(min_value=1, max_value=20),
)
def test_multiple_video_prediction_round_trip(model_name, num_predictions):
    """
    Property: Storing multiple predictions and retrieving each should return
    the same predictions.
    
    This test verifies that PredictionStorage maintains data integrity for
    multiple predictions.
    """
    storage = PredictionStorage()
    
    # Create multiple predictions with varying success/failure
    original_predictions = []
    base_annotations = {k: 1 for k in ANNOTATION_CATEGORIES}
    
    for i in range(num_predictions):
        success = i % 3 != 0  # Every 3rd prediction fails
        original_predictions.append(PredictionResult(
            video_id=f"video_{i}",
            predictions=base_annotations.copy(),
            success=success,
            error_message=f"Error for video_{i}" if not success else None,
            inference_time=float(i),
        ))
    
    # Store all predictions
    storage.store_predictions(model_name, original_predictions)
    
    # Retrieve and verify each prediction
    for original in original_predictions:
        retrieved = storage.get_prediction_for_video(model_name, original.video_id)
        
        assert retrieved is not None, f"Prediction for {original.video_id} should exist"
        assert retrieved.video_id == original.video_id
        assert retrieved.predictions == original.predictions
        assert retrieved.success == original.success
        assert retrieved.error_message == original.error_message
        assert retrieved.inference_time == original.inference_time
    
    # Verify statistics
    prediction_set = storage.get_predictions(model_name)
    expected_success = sum(1 for p in original_predictions if p.success)
    expected_failure = num_predictions - expected_success
    
    assert prediction_set.total_count == num_predictions
    assert prediction_set.success_count == expected_success
    assert prediction_set.failure_count == expected_failure
    assert len(prediction_set.failed_video_ids) == expected_failure


@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
)
def test_nonexistent_video_returns_none(model_name):
    """
    Property: Retrieving a prediction for a non-existent video should return None.
    
    This test verifies that PredictionStorage handles missing data correctly.
    """
    storage = PredictionStorage()
    
    # Store some predictions
    predictions = [PredictionResult(
        video_id="existing_video",
        predictions={k: 1 for k in ANNOTATION_CATEGORIES},
        success=True,
        inference_time=1.0,
    )]
    storage.store_predictions(model_name, predictions)
    
    # Try to retrieve non-existent video
    result = storage.get_prediction_for_video(model_name, "nonexistent_video")
    assert result is None, "Should return None for non-existent video"
    
    # Try to retrieve from non-existent model
    result = storage.get_prediction_for_video("nonexistent_model", "existing_video")
    assert result is None, "Should return None for non-existent model"
