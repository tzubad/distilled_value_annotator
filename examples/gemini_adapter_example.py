"""
Example usage of GeminiAdapter for video annotation evaluation.

This example demonstrates how to:
1. Initialize a GeminiAdapter
2. Process a video annotation
3. Handle the prediction results
"""

from evaluation.adapters import GeminiAdapter
from evaluation.models import VideoAnnotation

# Example configuration for Gemini adapter
config = {
    'max_retries': 3,
    'retry_delay': 30,
    'safety_settings': {},
    'system_instructions_path': 'prompts/script_to_annotation_instructions.txt'
}

# Create adapter instance
adapter = GeminiAdapter(
    model_name='gemini-1.5-pro-002',
    config=config
)

# Initialize the adapter (loads system instructions and model)
print("Initializing Gemini adapter...")
if not adapter.initialize():
    print("Failed to initialize adapter")
    exit(1)

print(f"Adapter initialized: {adapter.get_model_name()} ({adapter.get_model_type()})")

# Create a sample video annotation
# In a real scenario, this would come from the ground truth dataset
sample_video = VideoAnnotation(
    video_id="sample_video_001",
    video_uri="gs://bucket/videos/sample_video_001.mp4",
    script_uri="gs://bucket/scripts/sample_video_001.txt",
    annotations={
        "Self_Direction_Thought": 1,
        "Self_Direction_Action": 0,
        "Stimulation": 2,
        "Hedonism": 1,
        "Achievement": 0,
        "Power_Resources": 0,
        "Power_Dominance": -1,
        "Face": 0,
        "Security_Personal": 1,
        "Security_Social": 0,
        "Conformity_Rules": 0,
        "Conformity_Interpersonal": 1,
        "Tradition": 0,
        "Humility": 0,
        "Benevolence_Dependability": 1,
        "Benevolence_Care": 2,
        "Universalism_Concern": 0,
        "Universalism_Nature": 0,
        "Universalism_Tolerance": 1,
    },
    has_sound=True,
    script_text="This is a sample script for demonstration purposes..."
)

# Generate prediction
print(f"\nGenerating prediction for video: {sample_video.video_id}")
result = adapter.predict(sample_video)

# Process the result
if result is None:
    print("Prediction returned None")
elif result.success:
    print(f"Prediction successful!")
    print(f"Inference time: {result.inference_time:.2f} seconds")
    print(f"\nPredictions:")
    for category, value in sorted(result.predictions.items()):
        value_label = {-1: "conflict", 0: "absent", 1: "present", 2: "dominant"}[value]
        print(f"  {category}: {value} ({value_label})")
else:
    print(f"Prediction failed: {result.error_message}")
    print(f"Inference time: {result.inference_time:.2f} seconds")

# Batch prediction example
print("\n" + "="*60)
print("Batch Prediction Example")
print("="*60)

# Create multiple sample videos
videos = []
for i in range(3):
    video = VideoAnnotation(
        video_id=f"video_{i:03d}",
        video_uri=f"gs://bucket/videos/video_{i:03d}.mp4",
        script_uri=f"gs://bucket/scripts/video_{i:03d}.txt",
        annotations={cat: 0 for cat in sample_video.annotations.keys()},
        has_sound=True,
        script_text=f"Sample script for video {i}..."
    )
    videos.append(video)

# Process batch
print(f"\nProcessing batch of {len(videos)} videos...")
results = adapter.batch_predict(videos)

# Summary
success_count = sum(1 for r in results if r.success)
failure_count = sum(1 for r in results if not r.success)

print(f"\nBatch processing complete:")
print(f"  Successful: {success_count}/{len(videos)}")
print(f"  Failed: {failure_count}/{len(videos)}")

for result in results:
    status = "✓" if result.success else "✗"
    print(f"  {status} {result.video_id}: {result.inference_time:.2f}s")
    if not result.success:
        print(f"    Error: {result.error_message}")
