# Implementation Plan: One-Step Video Annotation Pipeline

**Branch**: `001-one-step-annotation` | **Date**: 2024-12-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-one-step-annotation/spec.md`

## Summary

Add a one-step annotation mode (`pipeline.mode: "one_step"`) that processes TikTok videos directly to Schwartz value annotations without intermediate script generation, using the existing `videos_to_annotations_one-step.txt` prompt. The implementation maximizes code reuse from the existing two-step pipeline, requiring only a new LLM client class, a new processor class, and orchestrator routing logic.

## Technical Context

**Language/Version**: Python 3.9+  
**Primary Dependencies**: vertexai, google-cloud-storage, pandas, pyyaml  
**Storage**: Google Cloud Storage (GCS) for videos and CSV output  
**Testing**: pytest (existing test infrastructure)  
**Target Platform**: Linux/Windows server with GCP authentication  
**Project Type**: Single Python project  
**Performance Goals**: Process videos faster than two-step (1 LLM call vs 2)  
**Constraints**: Must maintain backward compatibility with existing config  
**Scale/Scope**: Same as two-step pipeline (batch processing of videos from GCS bucket)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution is not customized (uses template). No specific gates or constraints to check against. Proceeding with general best practices:

- [x] **Code Reuse**: FR-004, FR-005, FR-006 explicitly require reusing GCSInterface, retry logic, and CSVGenerator
- [x] **Backward Compatibility**: FR-007 requires existing two-step pipeline to remain unchanged
- [x] **Testing**: Existing pytest infrastructure will be extended
- [x] **Simplicity**: Minimal new code - only what's necessary for the one-step path

## Project Structure

### Documentation (this feature)

```text
specs/001-one-step-annotation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # N/A - internal refactor, no new APIs
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
# Files to MODIFY
config/
└── __init__.py          # Add pipeline_mode property

llm/
└── __init__.py          # Add OneStepAnnotationLLMClient class

processors/
└── __init__.py          # Add VideoToAnnotationProcessor class

orchestrator/
└── __init__.py          # Add routing for one_step mode + _run_one_step_pipeline method

# Files REUSED (no changes needed)
gcs/
└── __init__.py          # GCSInterface - list_videos(), save_csv() ✓

utils/
├── __init__.py          # CSVGenerator ✓
└── logger.py            # PipelineLogger ✓

prompts/
└── videos_to_annotations_one-step.txt  # Already exists ✓

tests/
├── test_llm.py          # Add tests for OneStepAnnotationLLMClient
├── test_processors.py   # Add tests for VideoToAnnotationProcessor
└── test_orchestrator.py # Add tests for one_step mode routing
```

**Structure Decision**: Single project structure maintained. New classes added to existing modules following established patterns.

## Complexity Tracking

No complexity violations. The implementation adds minimal new code by reusing:
- `BaseLLMClient` for retry logic and safety settings
- `GCSInterface` for video listing
- `CSVGenerator` for output generation
- `PipelineLogger` for consistent logging
