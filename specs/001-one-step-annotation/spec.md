# Feature Specification: One-Step Video Annotation Pipeline

**Feature Branch**: `001-one-step-annotation`  
**Created**: 2024-12-08  
**Status**: Draft  
**Input**: User description: "Create a one-step video annotation pipeline that directly processes videos to CSV annotations without intermediate script generation, using an existing system prompt"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run One-Step Annotation Pipeline (Priority: P1)

As a researcher/annotator, I want to process TikTok videos directly to value annotations in a single step, so that I can skip the intermediate script generation phase and get annotations faster when I only need the final output.

**Why this priority**: This is the core functionality of the feature. Without this, the one-step pipeline doesn't exist.

**Independent Test**: Can be fully tested by running the pipeline on a single video and verifying a valid CSV output is produced with all 19 Schwartz value annotations plus the Has_sound field.

**Acceptance Scenarios**:

1. **Given** a GCS bucket with video files and a config set to `one_step` mode, **When** I run the pipeline, **Then** the system processes each video directly through the one-step prompt and produces a CSV with annotations
2. **Given** a video that has audio, **When** processed through the one-step pipeline, **Then** the Has_sound field in the output is set to `True`
3. **Given** a video where a Schwartz value is present but not dominant, **When** processed, **Then** the corresponding value field shows `present`
4. **Given** a video where content conflicts with a Schwartz value, **When** processed, **Then** the corresponding value field shows `conflict`
5. **Given** a video where a Schwartz value is not present, **When** processed, **Then** the corresponding value field is `None`

---

### User Story 2 - Choose Between One-Step and Two-Step Modes (Priority: P1)

As a researcher, I want to choose between one-step (video → annotations) and two-step (video → script → annotations) processing modes, so that I can select the approach that best fits my research needs.

**Why this priority**: Users need a clear way to switch between the existing two-step pipeline and the new one-step pipeline.

**Independent Test**: Can be tested by modifying the config file to specify the pipeline mode and verifying the correct processing path is executed.

**Acceptance Scenarios**:

1. **Given** a configuration with `pipeline.mode: "one_step"`, **When** I run the pipeline, **Then** videos are processed directly to annotations without generating intermediate scripts
2. **Given** a configuration with `pipeline.mode: "two_step"` or the existing `stage` configuration, **When** I run the pipeline, **Then** the existing two-step behavior is preserved
3. **Given** no explicit mode configuration, **When** I run the pipeline, **Then** the system defaults to the existing two-step behavior for backward compatibility

---

### User Story 3 - Use Existing System Prompt for One-Step (Priority: P2)

As a researcher, I want the one-step pipeline to use the existing `videos_to_annotations_one-step.txt` prompt, so that the annotation output follows the established Schwartz value framework and output format.

**Why this priority**: Ensures consistency with the established annotation methodology and output format.

**Independent Test**: Can be verified by checking that the LLM receives the content of `videos_to_annotations_one-step.txt` as system instructions when processing videos in one-step mode.

**Acceptance Scenarios**:

1. **Given** the one-step pipeline is running, **When** a video is processed, **Then** the system uses the prompt from `prompts/videos_to_annotations_one-step.txt` as system instructions
2. **Given** the prompt file exists, **When** the one-step LLM client initializes, **Then** it loads the prompt successfully

---

### User Story 4 - Reuse Video Processing Logic (Priority: P2)

As a developer, I want the one-step pipeline to reuse as much existing code as possible (video loading, GCS interface, retry logic, CSV generation), so that the implementation is maintainable and consistent.

**Why this priority**: Reduces development effort and ensures consistent behavior across both pipeline modes.

**Independent Test**: Can be verified by code review confirming shared components are used, and integration tests showing consistent error handling and retry behavior.

**Acceptance Scenarios**:

1. **Given** a video fails to process, **When** retry attempts are made, **Then** the same exponential backoff logic used in the two-step pipeline is applied
2. **Given** successful annotation generation, **When** the CSV is generated, **Then** the same CSVGenerator utility is used
3. **Given** videos are listed from GCS, **When** the one-step pipeline runs, **Then** the same GCSInterface methods are used

---

### User Story 5 - View Consistent Execution Summary (Priority: P3)

As a researcher, I want to see a clear execution summary showing processing results, so that I can understand what was processed and identify any failures.

**Why this priority**: Provides visibility into pipeline execution, helping users diagnose issues.

**Independent Test**: Can be tested by running the one-step pipeline and verifying the summary output includes video count, success/failure counts, and output location.

**Acceptance Scenarios**:

1. **Given** the one-step pipeline completes, **When** the summary is printed, **Then** it shows total videos processed, successful annotations, failed videos, and CSV location
2. **Given** some videos fail processing, **When** the summary is displayed, **Then** the failed videos are listed

---

### Edge Cases

- What happens when a video cannot be loaded from GCS? System should log the error and continue processing remaining videos.
- How does the system handle LLM responses that don't match the expected JSON format? System should use the existing JSON extraction logic and log parsing errors.
- What happens when the one-step prompt file is missing? System should fail initialization with a clear error message.
- How does the system handle rate limiting from the LLM API? System should use the existing retry logic with exponential backoff.
- What happens when the Has_sound field is ambiguous (no clear audio detection)? The LLM determines this based on video content; if unclear, it defaults to False.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support a `one_step` pipeline mode that processes videos directly to annotations
- **FR-002**: System MUST produce CSV output with all 19 Schwartz value columns plus Has_sound field
- **FR-003**: System MUST use the existing `prompts/videos_to_annotations_one-step.txt` as the LLM system prompt for one-step mode
- **FR-004**: System MUST reuse the existing GCSInterface for video listing and CSV output
- **FR-005**: System MUST reuse the existing retry logic with exponential backoff for LLM API calls
- **FR-006**: System MUST reuse the existing CSVGenerator for producing output files
- **FR-007**: System MUST maintain backward compatibility with the existing two-step pipeline configuration
- **FR-008**: System MUST parse LLM responses to extract JSON annotation data using existing parsing logic
- **FR-009**: System MUST log processing progress and errors consistently with the existing pipeline
- **FR-010**: System MUST produce a summary report showing processing results when execution completes
- **FR-011**: Each annotation row MUST contain a video_id derived from the video filename
- **FR-012**: Value columns MUST contain one of: `present`, `conflict`, or `None`
- **FR-013**: Has_sound column MUST contain `True` or `False`

### Key Entities

- **Video**: A TikTok video file stored in GCS (MP4 format), identified by filename (e.g., `@username_video_12345.mp4`)
- **Annotation**: A set of 19 Schwartz value assessments plus audio detection for a single video, with values indicating presence, conflict, or absence
- **Pipeline Configuration**: Settings specifying the processing mode (one-step vs two-step), GCS paths, model settings, and safety settings
- **One-Step LLM Client**: A specialized LLM client that takes video input and returns JSON annotations directly

## Assumptions

- The existing `videos_to_annotations_one-step.txt` prompt is complete and production-ready for direct video-to-annotation processing
- The Gemini model used supports video input in the same way as the existing video-to-script processing
- The output JSON format from the one-step prompt matches or can be mapped to the existing CSV column structure
- GCS video access permissions are already configured correctly (same as two-step pipeline)
- The one-step mode will use the same model (e.g., gemini-1.5-pro-002) as the two-step pipeline

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can process a batch of videos to annotations in a single pipeline run without intermediate script files
- **SC-002**: Processing time per video is reduced compared to the two-step approach (no intermediate LLM call for script-to-annotation)
- **SC-003**: Output CSV contains valid annotations for all processed videos with the correct column structure
- **SC-004**: Existing two-step pipeline functionality remains unchanged when using the existing configuration
- **SC-005**: 100% of the existing utility code (GCSInterface, CSVGenerator, retry logic) is reused without modification
- **SC-006**: Pipeline execution summary provides clear visibility into processing results (success/failure counts, output location)
