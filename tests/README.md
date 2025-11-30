# Testing Guide

This directory contains unit and integration tests for the video annotation pipeline.

## Setup

Install testing dependencies:

```bash
pip install -r tests/requirements-test.txt
```

## Running Tests

### Option 1: Run unit tests only (no GCS required)

```bash
python run_tests.py
```

Or directly with pytest:

```bash
pytest tests/ -m "not integration"
```

### Option 2: Run all tests including integration tests

```bash
python run_tests.py --all
```

Note: Integration tests require:
- GCS bucket with test videos
- GCP credentials configured
- Vertex AI API enabled

### Option 3: Run with coverage report

```bash
python run_tests.py --coverage
```

## Test Structure

- `test_config.py` - Unit tests for configuration module
- `test_gcs.py` - Unit tests for GCS interface (mocked)
- `test_llm.py` - Unit tests for LLM clients (mocked)
- `test_integration.py` - Integration tests (require GCS access)

## Uploading Test Videos to GCS

Before running integration tests, upload your test videos:

```bash
python upload_test_videos.py --bucket your-bucket-name --prefix test/videos/
```

This will upload all MP4 files from the current directory to your GCS bucket.

## Test Videos

The integration tests expect 3 sample videos:
- `@alexkay_video_6783398367490854150.mp4`
- `@alexkay_video_6807140917636648197.mp4`
- `@alexkay_video_6811970678024195334.mp4`

## Running Specific Tests

Run a specific test file:
```bash
pytest tests/test_config.py -v
```

Run a specific test:
```bash
pytest tests/test_config.py::TestPipelineConfig::test_valid_config_loading -v
```

Run tests matching a pattern:
```bash
pytest tests/ -k "config" -v
```
