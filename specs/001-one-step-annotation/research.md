# Research: One-Step Video Annotation Pipeline

**Feature**: 001-one-step-annotation  
**Date**: 2024-12-08

## Research Tasks & Findings

### 1. Video Input Pattern

**Question**: How does the existing pipeline handle video input to the LLM?

**Finding**: The `VideoScriptLLMClient.generate_script_from_video()` method uses:

```python
from vertexai.generative_models import Part
contents = Part.from_uri(uri=video_uri, mime_type="video/mp4")
prompt = [Part.from_text("Video: "), contents]
response = self.model.generate_content(prompt, safety_settings=self.safety_settings)
```

**Decision**: Reuse this exact pattern in `OneStepAnnotationLLMClient.generate_annotations_from_video()`

**Rationale**: 
- Proven to work with GCS video URIs
- Handles mime type correctly
- Integrates with existing safety settings

---

### 2. JSON Response Parsing

**Question**: How does the existing pipeline parse LLM JSON responses?

**Finding**: The `ScriptToAnnotationProcessor._extract_json_and_text()` method provides robust parsing:

1. Extracts JSON from markdown code blocks (```json ... ```)
2. Falls back to parsing entire response as JSON
3. Uses regex to find JSON objects in mixed content
4. Extracts text notes from outside code blocks

**Decision**: Copy `_extract_json_and_text()` to `VideoToAnnotationProcessor` (or consider extracting to shared utility in future refactor)

**Rationale**:
- Handles all edge cases from LLM responses
- Already tested in production
- One-step prompt uses same output format

---

### 3. Output Format Alignment

**Question**: Does the one-step prompt output format match the two-step format?

**Finding**: Both prompts specify identical output format:

| Field Type | Format |
|------------|--------|
| Value fields | `1_Value1_{Name}_values: {'present', 'conflict', None}` |
| Sound field | `Has_sound: {True, False}` |

**Evidence**: End of both prompt files:
```
possible field values:
Has_sound: {True, False}
1_Value1_XXXXXXXXXXXX_values: {'present', 'conflict', None}
```

**Decision**: `CSVGenerator` can be reused without modification

**Rationale**: The normalization logic in CSVGenerator already handles the `1_Value1_*_values` format and maps to correct column names.

---

### 4. Configuration Extension Strategy

**Question**: How to add one-step mode without breaking existing configurations?

**Finding**: Current config uses `pipeline.stage` with values: `both`, `video_to_script`, `script_to_annotation`

**Decision**: Add new `pipeline.mode` property with values: `one_step`, `two_step` (default)

**Implementation**:
```python
@property
def pipeline_mode(self) -> str:
    """Get the pipeline mode ('one_step' or 'two_step')."""
    return self._config.get('pipeline', {}).get('mode', 'two_step')
```

**Rationale**:
- `mode` is semantically different from `stage` 
- Default `two_step` ensures backward compatibility
- When `mode: one_step`, the `stage` setting is ignored
- Existing configs without `mode` continue to work as before

---

### 5. Processor Design Pattern

**Question**: What pattern should `VideoToAnnotationProcessor` follow?

**Finding**: Two existing processors with different output patterns:

| Processor | Input | Output |
|-----------|-------|--------|
| `VideoToScriptProcessor` | video URIs | script texts or script URIs |
| `ScriptToAnnotationProcessor` | script sources | annotation dicts |

**Decision**: `VideoToAnnotationProcessor` follows `ScriptToAnnotationProcessor` output pattern (returns annotation dicts)

**Interface**:
```python
class VideoToAnnotationProcessor:
    def process_videos(self, video_uris: List[str]) -> Tuple[List[Dict], List[str]]:
        """Returns (annotations, failed_uris)"""
```

**Rationale**: 
- Annotations go directly to CSVGenerator
- Consistent with how orchestrator consumes ScriptToAnnotationProcessor output
- No intermediate storage needed

---

### 6. Orchestrator Routing

**Question**: How should the orchestrator route between one-step and two-step modes?

**Finding**: Current `run()` method checks `config.stage_to_run` and routes to appropriate methods.

**Decision**: Add mode check before stage check:

```python
def run(self) -> Dict[str, Any]:
    if self.config.pipeline_mode == 'one_step':
        return self._run_one_step_pipeline()
    
    # Existing two-step logic based on stage
    stage = self.config.stage_to_run
    if stage == 'both':
        return self._run_complete_pipeline()
    # ... etc
```

**Rationale**:
- Clean separation between pipeline modes
- Two-step logic unchanged
- New one-step logic isolated in dedicated method

---

## Best Practices Applied

### From Existing Codebase

1. **Logging Pattern**: Use both `logging` module and `PipelineLogger` for structured error tracking
2. **Error Handling**: Return error strings starting with "Error:" for non-exception failures
3. **Retry Logic**: Inherit from `BaseLLMClient` for exponential backoff
4. **Delay Between Requests**: Apply `request_delay` between batch processing iterations
5. **Summary Generation**: Return structured dict with stage info, counts, and paths

### Code Reuse Summary

| Component | Reuse Type | Notes |
|-----------|------------|-------|
| `BaseLLMClient` | Inheritance | Full retry logic, safety settings |
| `GCSInterface.list_videos()` | Direct call | No changes needed |
| `CSVGenerator.generate_and_save()` | Direct call | No changes needed |
| `PipelineLogger` | Instance sharing | Same logging infrastructure |
| `_extract_json_and_text()` | Method copy | From ScriptToAnnotationProcessor |
| Video Part creation | Code pattern | From VideoScriptLLMClient |

---

## Alternatives Considered

### Alternative 1: Extend VideoScriptLLMClient

**Rejected**: The purpose is fundamentally different (script generation vs annotation generation). Would violate single responsibility.

### Alternative 2: Single unified processor for all modes

**Rejected**: Would require extensive refactoring of existing processors. Risk of breaking working code.

### Alternative 3: Use `stage` values instead of new `mode` property

**Rejected**: Semantically incorrect - "one_step" is a mode, not a stage. Would confuse the mental model.

---

## Open Questions (None)

All research questions have been resolved. No blockers for implementation.
