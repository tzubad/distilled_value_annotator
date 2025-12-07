# Model Adapters

This module provides a unified interface for different model types (LLM, MLM, etc.) to enable consistent evaluation across different architectures.

## Components

### ModelAdapter (base.py)

Abstract base class that defines the interface all model adapters must implement.

**Required Methods:**
- `initialize() -> bool`: Load model and prepare for inference
- `predict(video: VideoAnnotation) -> Optional[PredictionResult]`: Generate predictions for a single video
- `get_model_type() -> str`: Return model type (e.g., 'LLM', 'MLM')
- `get_model_name() -> str`: Return specific model identifier

**Provided Methods:**
- `batch_predict(videos: List[VideoAnnotation]) -> List[PredictionResult]`: Process multiple videos with error handling

### ScriptLoader (script_loader.py)

Utility for loading video scripts from GCS or local filesystem with caching.

**Features:**
- Supports both GCS URIs (`gs://bucket/path`) and local file paths
- Caches loaded scripts to avoid repeated reads
- Handles missing files gracefully
- Supports Unicode content

**Methods:**
- `load_script(script_uri: str) -> Optional[str]`: Load script content
- `clear_cache()`: Clear the script cache
- `get_cache_size() -> int`: Get number of cached scripts
- `is_cached(script_uri: str) -> bool`: Check if script is cached

### GeminiAdapter (gemini_adapter.py)

Adapter for Google Gemini LLM models via Vertex AI.

**Features:**
- Loads system instructions from `prompts/script_to_annotation_instructions.txt`
- Supports exponential backoff retry logic for API rate limits
- Configurable safety settings
- Parses LLM JSON responses to extract 19 category annotations
- Normalizes output to standard format (-1, 0, 1, 2)

**Configuration:**
```python
config = {
    'max_retries': 3,              # Maximum retry attempts
    'retry_delay': 30,             # Initial retry delay in seconds
    'safety_settings': {},         # Safety setting overrides
    'system_instructions_path': 'prompts/script_to_annotation_instructions.txt'
}

adapter = GeminiAdapter(
    model_name='gemini-1.5-pro-002',
    config=config
)
```

**Output Format:**
The adapter converts LLM responses from the prompt format to standard category names:
- `1_Value1_Self_Direction_Thought_values` → `Self_Direction_Thought`
- Values: `"present"` → 1, `"conflict"` → -1, `None` → 0

**Error Handling:**
- Retries with exponential backoff on API failures
- Logs errors with video_id and model_name context
- Returns failed PredictionResult on errors

## Usage Example

```python
from evaluation.adapters import GeminiAdapter
from evaluation.models import VideoAnnotation

# Initialize Gemini adapter
adapter = GeminiAdapter(
    model_name='gemini-1.5-pro-002',
    config={'max_retries': 3, 'retry_delay': 30}
)

if adapter.initialize():
    # Process a single video
    result = adapter.predict(video)
    if result.success:
        print(f"Predictions: {result.predictions}")
    
    # Process multiple videos with error handling
    results = adapter.batch_predict(videos)
    success_count = sum(1 for r in results if r.success)
    print(f"Processed {success_count}/{len(videos)} videos successfully")
```

## Creating Custom Adapters

```python
from evaluation.adapters import ModelAdapter, ScriptLoader
from evaluation.models import VideoAnnotation, PredictionResult

class MyCustomAdapter(ModelAdapter):
    def __init__(self, model_name: str, config: dict):
        super().__init__(model_name, config)
        self.script_loader = ScriptLoader()
    
    def initialize(self) -> bool:
        # Load your model here
        return True
    
    def predict(self, video: VideoAnnotation) -> Optional[PredictionResult]:
        # Load script
        script = self.script_loader.load_script(video.script_uri)
        
        # Generate predictions for all 19 categories
        predictions = {}  # Your prediction logic here
        
        return PredictionResult(
            video_id=video.video_id,
            predictions=predictions,
            success=True,
            inference_time=0.1
        )
    
    def get_model_type(self) -> str:
        return "CUSTOM"
    
    def get_model_name(self) -> str:
        return self.model_name
```

## Testing

Property-based tests verify that all adapters:
- Implement the required interface methods (Property 15)
- Return correct types and handle errors gracefully
- Normalize outputs to standard format (Property 16)
- Generate complete predictions for all 19 categories (Property 6)
- Maintain independent state

Run tests:
```bash
python -m pytest tests/test_adapters.py -v
python -m pytest tests/test_script_loader.py -v
```

## Requirements Validated

- **Requirement 1.1, 1.2**: LLM model evaluation with script processing
- **Requirement 2.2**: MLM prediction completeness
- **Requirement 7.1**: Unified interface for different model types
- **Requirement 7.2, 7.3**: Model input transformation and output normalization
- **Requirement 7.4**: Consistent evaluation logic across models
- **Requirement 10.1**: Error logging with context (video_id, model_name)

