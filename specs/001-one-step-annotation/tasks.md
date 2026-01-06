# Tasks: One-Step Video Annotation Pipeline

**Input**: Design documents from `/specs/001-one-step-annotation/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓

**Tests**: Not explicitly requested in spec. Tests omitted.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Exact file paths included in descriptions

---

## Phase 1: Setup

**Purpose**: No new project setup needed - extending existing project

- [X] T001 Verify prompt file exists at `prompts/videos_to_annotations_one-step.txt`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Configuration extension that enables one-step mode routing

**⚠️ CRITICAL**: User Story implementation cannot begin until this phase is complete

- [X] T002 Add `pipeline_mode` property to `config/__init__.py` with default "two_step"
- [X] T003 Add validation for `pipeline.mode` in `config/__init__.py` validate() method

**Checkpoint**: Configuration can now distinguish between one_step and two_step modes

---

## Phase 3: User Story 1 + User Story 3 - One-Step Core (Priority: P1 + P2) 🎯 MVP

**Goal**: Process videos directly to annotations using the one-step prompt

**Independent Test**: Run `python main.py --config config_one_step.yaml` with `pipeline.mode: "one_step"` and verify CSV output with 19 value columns + Has_sound

**Why combined**: US1 (run pipeline) and US3 (use prompt) are inseparable - the pipeline needs the prompt to function

### Implementation

- [X] T004 [P] [US1] Create `OneStepAnnotationLLMClient` class in `llm/__init__.py` that loads `prompts/videos_to_annotations_one-step.txt`
- [X] T005 [P] [US1] Implement `generate_annotations_from_video(video_uri)` method in `OneStepAnnotationLLMClient` using video Part pattern from `VideoScriptLLMClient`
- [X] T006 [US1] Create `VideoToAnnotationProcessor` class in `processors/__init__.py` with constructor accepting (llm_client, gcs_interface, request_delay, pipeline_logger)
- [X] T007 [US1] Copy `_extract_json_and_text()` method from `ScriptToAnnotationProcessor` to `VideoToAnnotationProcessor` in `processors/__init__.py`
- [X] T008 [US1] Implement `_process_single_video(video_uri)` method in `VideoToAnnotationProcessor` that returns annotation dict with video_id
- [X] T009 [US1] Implement `process_videos(video_uris)` batch method in `VideoToAnnotationProcessor` returning (annotations, failed_uris)

**Checkpoint**: LLM client and processor classes are complete - ready for orchestrator integration

---

## Phase 4: User Story 2 - Mode Selection (Priority: P1)

**Goal**: Users can choose between one-step and two-step modes via configuration

**Independent Test**: Set `pipeline.mode: "one_step"` in config, run pipeline, verify one-step path is executed (no intermediate scripts)

### Implementation

- [X] T010 [US2] Initialize `one_step_client` (OneStepAnnotationLLMClient) conditionally in `orchestrator/__init__.py` `__init__` when mode is "one_step"
- [X] T011 [US2] Initialize `one_step_processor` (VideoToAnnotationProcessor) conditionally in `orchestrator/__init__.py` `__init__` when mode is "one_step"
- [X] T012 [US2] Add `_run_one_step_pipeline()` method to `PipelineOrchestrator` in `orchestrator/__init__.py` that lists videos, processes them, generates CSV
- [X] T013 [US2] Modify `run()` method in `orchestrator/__init__.py` to check `config.pipeline_mode` first and route to `_run_one_step_pipeline()` if "one_step"
- [X] T014 [US2] Ensure backward compatibility: when mode is "two_step" or not set, existing `run()` logic unchanged in `orchestrator/__init__.py`

**Checkpoint**: One-step pipeline is fully functional end-to-end

---

## Phase 5: User Story 4 - Code Reuse Verification (Priority: P2)

**Goal**: Confirm one-step pipeline uses shared components correctly

**Independent Test**: Code review + verify error handling matches two-step behavior

### Implementation

- [X] T015 [US4] Verify `VideoToAnnotationProcessor` uses same request_delay pattern as `VideoToScriptProcessor` in `processors/__init__.py`
- [X] T016 [US4] Verify `_run_one_step_pipeline()` uses existing `CSVGenerator.generate_and_save()` in `orchestrator/__init__.py`
- [X] T017 [US4] Verify `_run_one_step_pipeline()` uses existing `GCSInterface.list_videos()` in `orchestrator/__init__.py`

**Checkpoint**: Code reuse requirements (FR-004, FR-005, FR-006) confirmed

---

## Phase 6: User Story 5 - Execution Summary (Priority: P3)

**Goal**: Clear execution summary for one-step pipeline results

**Independent Test**: Run one-step pipeline, verify summary shows video count, success/failure counts, CSV location

### Implementation

- [X] T018 [US5] Add summary generation in `_run_one_step_pipeline()` returning dict with stage, total_videos, successful_annotations, failed_videos, csv_saved, csv_path in `orchestrator/__init__.py`
- [X] T019 [US5] Add summary logging in `_run_one_step_pipeline()` matching format of `_run_complete_pipeline()` in `orchestrator/__init__.py`
- [X] T020 [US5] Update `print_execution_summary()` in `main.py` to handle stage='one_step' summary format

**Checkpoint**: Pipeline provides clear visibility into processing results

---

## Phase 7: Polish & Documentation

**Purpose**: Documentation and final validation

- [X] T021 [P] Update `README.md` to document new `pipeline.mode` configuration option
- [X] T022 [P] Add example configuration section for one-step mode in `README.md`
- [X] T023 Run `quickstart.md` validation - execute one-step pipeline with test videos
- [X] T024 Verify existing two-step pipeline still works (backward compatibility test)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    │
    ▼
Phase 2: Foundational (config extension) ──── BLOCKS ALL USER STORIES
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
Phase 3: US1+US3 (Core)              Phase 4: US2 (Mode Selection)
    │                                      │
    └──────────────┬───────────────────────┘
                   │
                   ▼
             Phase 5: US4 (Code Reuse Verification)
                   │
                   ▼
             Phase 6: US5 (Execution Summary)
                   │
                   ▼
             Phase 7: Polish
```

### Task Dependencies Within Phases

**Phase 2**:
- T002 before T003 (property must exist before validation)

**Phase 3**:
- T004, T005 parallel (different methods in same class)
- T006 after T004, T005 (processor needs client class)
- T007 after T006 (method in processor class)
- T008, T009 after T007 (depend on extract method)

**Phase 4**:
- T010, T011 after Phase 3 (need client and processor classes)
- T012 after T010, T011 (method uses initialized components)
- T013 after T012 (routing calls the method)
- T014 after T013 (verification after implementation)

**Phase 5**:
- T015, T016, T017 parallel (independent verification tasks)

**Phase 6**:
- T018, T019, T020 sequential (summary generation, logging, display)

**Phase 7**:
- T021, T022 parallel (different sections)
- T023, T024 after T021, T022 (validation after documentation)

### Parallel Opportunities

```bash
# Phase 2: Sequential (dependency)
T002 → T003

# Phase 3: Models + LLM client in parallel, then processor
[T004, T005] → T006 → T007 → [T008, T009]

# Phase 5: All verification in parallel
[T015, T016, T017]

# Phase 7: Documentation in parallel, then validation
[T021, T022] → [T023, T024]
```

---

## Implementation Strategy

### MVP First (Phase 1-4)

1. Complete Phase 1: Setup verification
2. Complete Phase 2: Foundational config
3. Complete Phase 3: Core LLM client + processor
4. Complete Phase 4: Orchestrator integration
5. **STOP and VALIDATE**: Run one-step pipeline end-to-end
6. Deploy if ready

### Incremental Delivery

1. Phases 1-2: Config ready
2. Phase 3: Core components testable with mock data
3. Phase 4: Full integration → MVP complete
4. Phase 5-6: Quality improvements
5. Phase 7: Documentation and final validation

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tasks** | 24 |
| **Setup Tasks** | 1 |
| **Foundational Tasks** | 2 |
| **User Story 1+3 Tasks** | 6 |
| **User Story 2 Tasks** | 5 |
| **User Story 4 Tasks** | 3 |
| **User Story 5 Tasks** | 3 |
| **Polish Tasks** | 4 |
| **Parallel Opportunities** | 5 groups |
| **MVP Scope** | Tasks T001-T014 (Phases 1-4) |

### Independent Test Criteria

| User Story | Independent Test |
|------------|------------------|
| US1+US3 | Process single video → valid annotation dict returned |
| US2 | Config with `mode: "one_step"` → one-step path executed |
| US4 | Code review confirms shared component usage |
| US5 | Summary output includes counts and CSV path |

### Format Validation

✅ All tasks follow checklist format: `- [ ] [TaskID] [P?] [Story?] Description with file path`
