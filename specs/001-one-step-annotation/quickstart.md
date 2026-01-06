# Quickstart: One-Step Video Annotation Pipeline

**Feature**: 001-one-step-annotation  
**Date**: 2024-12-08

## Prerequisites

- Python 3.9+
- Google Cloud SDK installed and authenticated
- Access to a GCS bucket with video files
- Vertex AI API enabled in your GCP project

## Quick Setup

### 1. Install Dependencies

```bash
cd distilled_value_annotator
pip install -r requirements.txt
```

### 2. Authenticate with GCP

```bash
gcloud auth application-default login
```

### 3. Create One-Step Configuration

Create `config_one_step.yaml`:

```yaml
# One-Step Video Annotation Configuration
gcs:
  bucket_name: "your-bucket-name"
  video_source_path: "videos/your_videos/"
  csv_output_path: "output/one_step_results.csv"

model:
  name: "gemini-1.5-pro-002"
  max_retries: 4
  retry_delay: 40
  request_delay: 3

pipeline:
  mode: "one_step"  # NEW: Enables one-step annotation

safety_settings:
  harassment: "BLOCK_NONE"
  hate_speech: "BLOCK_NONE"
  sexually_explicit: "BLOCK_NONE"
  dangerous_content: "BLOCK_NONE"
```

### 4. Run the Pipeline

```bash
python main.py --config config_one_step.yaml
```

## Expected Output

### Console Output

```
Loading configuration from: config_one_step.yaml
Configuration loaded and validated successfully

Initializing pipeline orchestrator...
Pipeline orchestrator initialized successfully

Starting pipeline execution...
Pipeline mode: one_step

============================================================
Starting one-step annotation pipeline
============================================================
Found 5 videos to process
Progress: 1/5 - Processing gs://bucket/videos/video1.mp4
Successfully generated annotations for video1
Progress: 2/5 - Processing gs://bucket/videos/video2.mp4
...

============================================================
PIPELINE EXECUTION SUMMARY
============================================================
Stage: one_step
Videos Processed: 5/5
Annotations Generated: 5
CSV Saved: Yes
CSV Location: gs://bucket/output/one_step_results.csv
============================================================
```

### CSV Output Format

```csv
video_id,Self_Direction_Thought,Self_Direction_Action,Stimulation,Hedonism,Achievement,Power_Resources,Power_Dominance,Face,Security_Personal,Security_Social,Conformity_Rules,Conformity_Interpersonal,Tradition,Humility,Benevolence_Dependability,Benevolence_Care,Universalism_Concern,Universalism_Nature,Universalism_Tolerance,Has_sound,notes
@username_video_12345,present,,conflict,present,,,present,,,,,,,,,present,,,True,"The video shows..."
```

## Running Tests

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files for this feature
pytest tests/test_llm.py -v -k "one_step"
pytest tests/test_processors.py -v -k "VideoToAnnotation"
pytest tests/test_orchestrator.py -v -k "one_step"
pytest tests/test_config.py -v -k "pipeline_mode"
```

### Integration Test

```bash
# Run with a small test dataset
pytest tests/test_integration.py -v -k "one_step"
```

## Configuration Reference

### Mode vs Stage

| Setting | Value | Description |
|---------|-------|-------------|
| `pipeline.mode` | `one_step` | Process videos directly to annotations (1 LLM call) |
| `pipeline.mode` | `two_step` | Process videos → scripts → annotations (2 LLM calls) |

When `mode: one_step`, the `stage` setting is ignored.

### Backward Compatibility

Existing configurations without the `mode` property will continue to work as `two_step` (default).

```yaml
# This config still works exactly as before
pipeline:
  stage: "both"        # Used in two_step mode
  save_scripts: true   # Used in two_step mode
```

## Comparison: One-Step vs Two-Step

| Aspect | One-Step | Two-Step |
|--------|----------|----------|
| LLM Calls per Video | 1 | 2 |
| Intermediate Files | None | Optional scripts |
| Processing Speed | Faster | Slower |
| Script Availability | No | Yes (if save_scripts: true) |
| Use Case | Final annotations only | Need scripts for analysis |

## Troubleshooting

### Error: "Prompt file not found"

Ensure `prompts/videos_to_annotations_one-step.txt` exists in the project root.

### Error: "Invalid pipeline mode"

Check that `pipeline.mode` is set to either `"one_step"` or `"two_step"`.

### Empty Annotations

- Verify videos exist at the specified GCS path
- Check LLM response in logs for parsing errors
- Ensure safety settings are not blocking content

## Development Workflow

1. **Make changes** to source files
2. **Run unit tests** to verify changes
3. **Run integration test** with small dataset
4. **Test with production config** on full dataset

```bash
# Quick validation cycle
pytest tests/test_llm.py tests/test_processors.py tests/test_orchestrator.py -v
python main.py --config config_one_step.yaml
```
