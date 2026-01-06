# Data Model: One-Step Video Annotation Pipeline

**Feature**: 001-one-step-annotation  
**Date**: 2024-12-08

## Entities

### 1. OneStepAnnotationLLMClient

**Purpose**: LLM client that processes video input directly to value annotations

**Location**: `llm/__init__.py`

**Inherits**: `BaseLLMClient`

**Attributes**:
| Attribute | Type | Description |
|-----------|------|-------------|
| model_name | str | Vertex AI model identifier |
| system_instructions | str | Loaded from `prompts/videos_to_annotations_one-step.txt` |
| safety_settings | Dict | Converted safety settings for Vertex AI |
| max_retries | int | Maximum retry attempts |
| retry_delay | int | Base delay for exponential backoff |
| model | GenerativeModel | Vertex AI model instance |

**Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_annotations_from_video` | video_uri: str | str | Generate JSON annotations from video URI |

**Relationships**:
- Inherits retry logic from `BaseLLMClient`
- Used by `VideoToAnnotationProcessor`
- Follows same pattern as `VideoScriptLLMClient`

---

### 2. VideoToAnnotationProcessor

**Purpose**: Batch processor for converting videos directly to annotation dictionaries

**Location**: `processors/__init__.py`

**Attributes**:
| Attribute | Type | Description |
|-----------|------|-------------|
| llm_client | OneStepAnnotationLLMClient | LLM client for annotation generation |
| gcs_interface | GCSInterface | GCS operations interface |
| request_delay | int | Delay between API requests |
| pipeline_logger | PipelineLogger | Structured error tracking |

**Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `_extract_json_and_text` | response: str | Dict | Parse LLM response to extract JSON and notes |
| `_process_single_video` | video_uri: str | Optional[Dict] | Process one video, return annotation or None |
| `process_videos` | video_uris: List[str] | Tuple[List[Dict], List[str]] | Batch process videos, return (annotations, failed_uris) |

**Relationships**:
- Uses `OneStepAnnotationLLMClient` for LLM calls
- Uses `GCSInterface` (for potential future features, maintains consistency)
- Outputs annotations compatible with `CSVGenerator`

---

### 3. PipelineConfig (Extended)

**Purpose**: Configuration manager with new pipeline mode support

**Location**: `config/__init__.py`

**New Properties**:
| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `pipeline_mode` | str | "two_step" | Pipeline mode: "one_step" or "two_step" |

**Validation Rules**:
- `mode` must be one of: `["one_step", "two_step"]`
- When `mode: "one_step"`, `stage` setting is ignored
- When `mode: "two_step"` or not set, existing behavior preserved

---

### 4. PipelineOrchestrator (Extended)

**Purpose**: Orchestrator with one-step pipeline support

**Location**: `orchestrator/__init__.py`

**New Attributes**:
| Attribute | Type | Condition | Description |
|-----------|------|-----------|-------------|
| one_step_client | OneStepAnnotationLLMClient | mode == "one_step" | LLM client for one-step mode |
| one_step_processor | VideoToAnnotationProcessor | mode == "one_step" | Processor for one-step mode |

**New Methods**:
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `_run_one_step_pipeline` | None | Dict[str, Any] | Execute one-step pipeline end-to-end |

**Modified Methods**:
| Method | Change |
|--------|--------|
| `__init__` | Conditionally initialize one-step components |
| `run` | Route to one-step or two-step based on `config.pipeline_mode` |

---

## Data Flow

### One-Step Pipeline Flow

```
┌─────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
│ GCS Videos  │───▶│ VideoToAnnotationProcessor│───▶│ List[Dict] annotations  │
│ (MP4 files) │    │                          │    │                         │
└─────────────┘    └──────────────────────────┘    └───────────┬─────────────┘
                              │                                 │
                              ▼                                 ▼
                   ┌──────────────────────┐         ┌──────────────────────┐
                   │ OneStepAnnotation    │         │ CSVGenerator         │
                   │ LLMClient            │         │                      │
                   │ (video → JSON)       │         │ (annotations → CSV)  │
                   └──────────────────────┘         └──────────────────────┘
                                                               │
                                                               ▼
                                                    ┌──────────────────────┐
                                                    │ GCS CSV Output       │
                                                    │ (result.csv)         │
                                                    └──────────────────────┘
```

### Annotation Dictionary Structure

**Input from LLM** (after JSON parsing):
```python
{
    "1_Value1_Self_Direction_Thought_values": "present",
    "1_Value1_Self_Direction_Action_values": None,
    "1_Value1_Stimulation_values": "conflict",
    # ... 16 more value fields ...
    "1_Value1_Universalism_Tolerance_values": "present",
    "Has_sound": True,
    "notes": "Optional text notes from LLM"
}
```

**After processing** (with video_id added):
```python
{
    "video_id": "@username_video_12345",
    "1_Value1_Self_Direction_Thought_values": "present",
    # ... all value fields ...
    "Has_sound": True,
    "notes": "Optional text notes from LLM"
}
```

**CSV Output** (after CSVGenerator normalization):
```csv
video_id,Self_Direction_Thought,Self_Direction_Action,...,Has_sound,notes
@username_video_12345,present,,conflict,...,True,"Optional text notes"
```

---

## State Transitions

### Pipeline Mode States

```
                    ┌─────────────────────────────────┐
                    │ PipelineOrchestrator.run()      │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ Check config.pipeline_mode    │
                    └───────────────────────────────┘
                         │                    │
                         ▼                    ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ mode == one_step │  │ mode == two_step │
              │                  │  │ (default)        │
              └────────┬─────────┘  └────────┬─────────┘
                       │                      │
                       ▼                      ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ _run_one_step_   │  │ Check stage      │
              │ pipeline()       │  │ (existing logic) │
              └──────────────────┘  └──────────────────┘
```

---

## Validation Rules

### Annotation Value Validation

| Field Type | Valid Values |
|------------|--------------|
| Value fields (19 total) | `"present"`, `"conflict"`, `None` |
| Has_sound | `True`, `False` |
| video_id | Non-empty string derived from filename |
| notes | Optional string |

### Configuration Validation

| Property | Valid Values | Default |
|----------|--------------|---------|
| pipeline.mode | `"one_step"`, `"two_step"` | `"two_step"` |
| pipeline.stage | `"both"`, `"video_to_script"`, `"script_to_annotation"` | `"both"` |

**Rule**: When `mode == "one_step"`, the `stage` value is ignored and has no effect.
