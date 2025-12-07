"""
Example usage of the GroundTruthLoader class.

This example demonstrates how to:
1. Load a ground truth dataset from CSV
2. Validate the dataset
3. Sample a subset of videos
4. Access the loaded data
"""

from evaluation import GroundTruthLoader, ANNOTATION_CATEGORIES


def main():
    # Example 1: Load full dataset
    print("Example 1: Loading full dataset")
    print("-" * 50)
    
    loader = GroundTruthLoader(dataset_path="path/to/ground_truth.csv")
    
    try:
        dataset = loader.load()
        
        print(f"Total videos in file: {dataset.total_count}")
        print(f"Valid videos: {dataset.valid_count}")
        print(f"Videos in dataset: {len(dataset.videos)}")
        
        if dataset.validation_errors:
            print(f"\nValidation errors found: {len(dataset.validation_errors)}")
            for error in dataset.validation_errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        # Access first video
        if dataset.videos:
            video = dataset.videos[0]
            print(f"\nFirst video:")
            print(f"  ID: {video.video_id}")
            print(f"  URI: {video.video_uri}")
            print(f"  Script: {video.script_uri}")
            print(f"  Has sound: {video.has_sound}")
            print(f"  Annotations: {len(video.annotations)} categories")
            
            # Show some annotation values
            print(f"\n  Sample annotations:")
            for category in list(ANNOTATION_CATEGORIES)[:3]:
                value = video.annotations.get(category)
                print(f"    {category}: {value}")
    
    except FileNotFoundError:
        print("Ground truth file not found. This is just an example.")
    
    print("\n")
    
    # Example 2: Load with sampling
    print("Example 2: Loading with sampling")
    print("-" * 50)
    
    loader_sampled = GroundTruthLoader(
        dataset_path="path/to/ground_truth.csv",
        sample_size=100,  # Sample 100 videos
        random_seed=42    # For reproducibility
    )
    
    try:
        dataset_sampled = loader_sampled.load()
        
        print(f"Requested sample size: 100")
        print(f"Actual sample size: {len(dataset_sampled.videos)}")
        print(f"Valid videos: {dataset_sampled.valid_count}")
        
        # Calculate distribution in sample
        endorsed_count = 0
        conflict_count = 0
        absent_count = 0
        
        for video in dataset_sampled.videos:
            for value in video.annotations.values():
                if value in {1, 2}:
                    endorsed_count += 1
                elif value == -1:
                    conflict_count += 1
                elif value == 0:
                    absent_count += 1
        
        total = endorsed_count + conflict_count + absent_count
        print(f"\nValue distribution in sample:")
        print(f"  Endorsed (1,2): {endorsed_count} ({endorsed_count/total*100:.1f}%)")
        print(f"  Conflict (-1):  {conflict_count} ({conflict_count/total*100:.1f}%)")
        print(f"  Absent (0):     {absent_count} ({absent_count/total*100:.1f}%)")
    
    except FileNotFoundError:
        print("Ground truth file not found. This is just an example.")
    
    print("\n")
    
    # Example 3: Validation only
    print("Example 3: Validation without loading")
    print("-" * 50)
    
    # You can also validate a list of VideoAnnotation objects directly
    from evaluation import VideoAnnotation
    
    # Create a sample video
    sample_video = VideoAnnotation(
        video_id="test_video_1",
        video_uri="gs://bucket/test_video_1.mp4",
        script_uri="gs://bucket/scripts/test_video_1.txt",
        annotations={category: 1 for category in ANNOTATION_CATEGORIES},
        has_sound=True
    )
    
    loader = GroundTruthLoader(dataset_path="dummy.csv")
    validation_result = loader.validate([sample_video])
    
    print(f"Valid videos: {validation_result.valid_count}")
    print(f"Invalid videos: {validation_result.invalid_count}")
    print(f"Errors: {len(validation_result.errors)}")


if __name__ == "__main__":
    main()
