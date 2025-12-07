"""
Example demonstrating the ModelAdapter interface and ScriptLoader utility.

This example shows how to:
1. Create a custom model adapter
2. Use the ScriptLoader to load scripts
3. Process videos through the adapter
"""

from evaluation.adapters import ModelAdapter, ScriptLoader
from evaluation.models import VideoAnnotation, PredictionResult
from typing import Optional, Dict, Any


# Define the 19 annotation categories
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


class ExampleAdapter(ModelAdapter):
    """
    Example adapter that demonstrates the ModelAdapter interface.
    
    This is a simple rule-based adapter that:
    - Loads scripts using ScriptLoader
    - Analyzes script length to make predictions
    - Returns predictions in the standard format
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self.script_loader = ScriptLoader()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the adapter."""
        self.logger.info(f"Initializing {self.model_name}")
        self.initialized = True
        return True
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        """
        Generate predictions based on script length.
        
        This is a toy example that uses script length to determine predictions:
        - Short scripts (< 100 chars): mostly absent (0)
        - Medium scripts (100-500 chars): mix of absent and endorsed
        - Long scripts (> 500 chars): mostly endorsed (1)
        """
        if not self.initialized:
            self.logger.error("Adapter not initialized")
            return None
        
        try:
            # Load the script
            script_text = self.script_loader.load_script(video.script_uri)
            
            if script_text is None:
                self.logger.warning(f"Failed to load script for {video.video_id}")
                return PredictionResult(
                    video_id=video.video_id,
                    predictions={},
                    success=False,
                    error_message="Failed to load script",
                    inference_time=0.0
                )
            
            # Simple rule-based prediction based on script length
            script_length = len(script_text)
            predictions = {}
            
            if script_length < 100:
                # Short script: mostly absent
                for category in ANNOTATION_CATEGORIES:
                    predictions[category] = 0
            elif script_length < 500:
                # Medium script: mix
                for i, category in enumerate(ANNOTATION_CATEGORIES):
                    predictions[category] = 1 if i % 2 == 0 else 0
            else:
                # Long script: mostly endorsed
                for category in ANNOTATION_CATEGORIES:
                    predictions[category] = 1
            
            self.logger.info(
                f"Predicted {video.video_id} (script length: {script_length})"
            )
            
            return PredictionResult(
                video_id=video.video_id,
                predictions=predictions,
                success=True,
                inference_time=0.01
            )
        
        except Exception as e:
            self.logger.error(f"Error predicting {video.video_id}: {str(e)}")
            return PredictionResult(
                video_id=video.video_id,
                predictions={},
                success=False,
                error_message=str(e),
                inference_time=0.0
            )
    
    def get_model_type(self) -> str:
        """Return model type."""
        return "EXAMPLE"
    
    def get_model_name(self) -> str:
        """Return model name."""
        return self.model_name


def main():
    """Demonstrate the adapter interface."""
    import logging
    import tempfile
    import os
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Model Adapter Interface Example")
    print("=" * 60)
    
    # Create a temporary script file for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test script for video annotation. " * 20)
        temp_script_path = f.name
    
    try:
        # Create an example adapter
        adapter = ExampleAdapter(
            model_name="example-rule-based-v1",
            config={"threshold": 100}
        )
        
        print(f"\n1. Created adapter: {adapter.get_model_name()}")
        print(f"   Type: {adapter.get_model_type()}")
        
        # Initialize the adapter
        success = adapter.initialize()
        print(f"\n2. Initialized adapter: {success}")
        
        # Create a test video annotation
        test_annotations = {category: 1 for category in ANNOTATION_CATEGORIES}
        video = VideoAnnotation(
            video_id="test_video_001",
            video_uri="gs://bucket/test_video_001.mp4",
            script_uri=temp_script_path,  # Use local file
            annotations=test_annotations,
            has_sound=True
        )
        
        print(f"\n3. Created test video: {video.video_id}")
        print(f"   Script URI: {video.script_uri}")
        
        # Make a prediction
        result = adapter.predict(video)
        
        if result and result.success:
            print(f"\n4. Prediction successful!")
            print(f"   Video ID: {result.video_id}")
            print(f"   Inference time: {result.inference_time:.3f}s")
            print(f"   Sample predictions:")
            for i, (category, value) in enumerate(result.predictions.items()):
                if i < 5:  # Show first 5
                    print(f"     - {category}: {value}")
            print(f"     ... ({len(result.predictions)} total categories)")
        else:
            print(f"\n4. Prediction failed!")
            if result:
                print(f"   Error: {result.error_message}")
        
        # Test batch prediction
        videos = [video] * 3  # Process same video 3 times
        print(f"\n5. Testing batch prediction with {len(videos)} videos...")
        
        results = adapter.batch_predict(videos)
        success_count = sum(1 for r in results if r.success)
        
        print(f"   Processed: {len(results)} videos")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {len(results) - success_count}")
        
        # Check script loader cache
        print(f"\n6. Script loader cache size: {adapter.script_loader.get_cache_size()}")
        print(f"   Script cached: {adapter.script_loader.is_cached(temp_script_path)}")
        
    finally:
        # Clean up temp file
        os.unlink(temp_script_path)
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
