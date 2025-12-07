"""
Integration tests for the video annotation pipeline.
These tests require actual GCS access and Vertex AI credentials.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from config import PipelineConfig
from orchestrator import PipelineOrchestrator


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration file using real GCS bucket."""
        config_data = {
            'gcs': {
                'bucket_name': 'videos-scripts-and-annotations',
                'video_source_path': 'videos/POC_videos',
                'script_output_path': 'saved_scripts/POC_scripts',
                'csv_output_path': 'output/POC_results/test_result.csv'
            },
            'model': {
                'name': 'gemini-2.5-pro',
                'max_retries': 4,
                'retry_delay': 40,
                'request_delay': 3
            },
            'pipeline': {
                'stage': 'both',
                'save_scripts': True
            },
            'safety_settings': {
                'harassment': 'BLOCK_NONE',
                'hate_speech': 'BLOCK_NONE',
                'sexually_explicit': 'BLOCK_NONE',
                'dangerous_content': 'BLOCK_NONE'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        yield config_path
        
        # Cleanup
        Path(config_path).unlink()
    
    def test_complete_pipeline_with_sample_videos(self, test_config):
        """
        Test the complete pipeline with sample videos.
        
        Prerequisites:
        - GCS bucket 'videos-scripts-and-annotations' must be accessible
        - Videos must exist in 'videos/POC_videos/' path
        - GCP credentials must be configured
        - Vertex AI API must be enabled
        """
        # Load configuration
        config = PipelineConfig(test_config)
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Run pipeline
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'complete'
        assert summary['total_videos'] >= 3
        assert summary['successful_videos'] > 0
        assert summary['total_annotations'] > 0
        assert summary['csv_saved'] is True
        assert summary['csv_path'] is not None
        
        # Verify CSV path format
        assert summary['csv_path'].startswith('gs://')
        assert 'results.csv' in summary['csv_path'] or 'test_result.csv' in summary['csv_path']
    
    def test_video_to_script_stage_only(self, test_config):
        """
        Test only the video-to-script stage.
        
        Prerequisites:
        - GCS bucket 'videos-scripts-and-annotations' must be accessible
        - Videos must exist in 'videos/POC_videos/' path
        - GCP credentials must be configured
        - Vertex AI API must be enabled
        """
        # Modify config for video-to-script only
        with open(test_config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data['pipeline']['stage'] = 'video_to_script'
        
        with open(test_config, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        config = PipelineConfig(test_config)
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Run pipeline
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'video_to_script'
        assert summary['total_videos'] >= 3
        assert summary['successful_videos'] > 0
        assert summary['scripts_saved'] is True
    
    def test_script_to_annotation_stage_only(self, test_config):
        """
        Test only the script-to-annotation stage.
        
        Prerequisites:
        - GCS bucket 'videos-scripts-and-annotations' must be accessible
        - Test scripts must be present in 'TikTok_Videos/Scripts/' path
        - GCP credentials must be configured
        - Vertex AI API must be enabled
        """
        # Modify config for script-to-annotation only
        with open(test_config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data['pipeline']['stage'] = 'script_to_annotation'
        
        with open(test_config, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load configuration
        config = PipelineConfig(test_config)
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Run pipeline
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'script_to_annotation'
        assert summary['total_scripts'] > 0
        assert summary['successful_annotations'] > 0
        assert summary['csv_saved'] is True
    
    def test_csv_output_format(self):
        """Test that CSV output has the correct format."""
        from utils import CSVGenerator
        from unittest.mock import Mock
        
        # Create mock GCS interface
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.save_csv.return_value = True
        
        # Create test annotations
        annotations = [
            {
                'video_id': '@user_video_123',
                'Self_Direction_Thought': 3,
                'Achievement': 5,
                'Power_Dominance': 2,
                'Has_sound': True,
                'notes': 'Test note'
            },
            {
                'video_id': '@user_video_456',
                'Self_Direction_Action': 4,
                'Hedonism': 3,
                'Has_sound': False
            }
        ]
        
        # Generate CSV
        csv_gen = CSVGenerator(mock_gcs)
        result = csv_gen.generate_and_save(annotations, 'output/test.csv')
        
        assert result is True
        mock_gcs.save_csv.assert_called_once()
        
        # Get the CSV content that was passed to save_csv
        csv_content = mock_gcs.save_csv.call_args[0][0]
        
        # Verify CSV structure
        lines = csv_content.strip().split('\n')
        assert len(lines) == 3  # Header + 2 data rows
        
        # Verify header contains expected columns
        header = lines[0]
        assert 'video_id' in header
        assert 'Self_Direction_Thought' in header
        assert 'Achievement' in header
        assert 'Has_sound' in header
        assert 'notes' in header
        
        # Verify data rows
        assert '@user_video_123' in lines[1]
        assert '@user_video_456' in lines[2]
    
    def test_error_handling_with_invalid_videos(self):
        """Test that pipeline handles invalid videos gracefully."""
        from processors import VideoToScriptProcessor
        from unittest.mock import Mock
        
        # Create mock LLM client that returns errors
        mock_llm = Mock()
        mock_llm.generate_script_from_video.return_value = "Error: Failed to process video"
        
        # Create mock GCS interface
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        
        # Create processor
        processor = VideoToScriptProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0,
            save_scripts=False,
            script_output_path=None
        )
        
        # Process videos
        video_uris = ['gs://bucket/invalid1.mp4', 'gs://bucket/invalid2.mp4']
        scripts, failed = processor.process_videos(video_uris)
        
        # Verify error handling
        assert len(scripts) == 0
        assert len(failed) == 2
        assert 'gs://bucket/invalid1.mp4' in failed
        assert 'gs://bucket/invalid2.mp4' in failed


@pytest.mark.integration
def test_upload_local_videos_to_gcs():
    """
    Helper test to upload local test videos to GCS.
    
    This test should be run manually when you want to upload
    your local test videos to GCS for integration testing.
    
    Usage:
        pytest tests/test_integration.py::test_upload_local_videos_to_gcs -v -s
    """
    from google.cloud import storage
    import os
    
    # Configuration - using your actual bucket
    bucket_name = 'videos-scripts-and-annotations'
    local_video_dir = '.'  # Current directory
    gcs_prefix = 'TikTok_Videos/Videos/'
    
    # Video files to upload
    video_files = [
        '@alexkay_video_6783398367490854150.mp4',
        '@alexkay_video_6807140917636648197.mp4',
        '@alexkay_video_6811970678024195334.mp4'
    ]
    
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Upload each video
    for video_file in video_files:
        local_path = os.path.join(local_video_dir, video_file)
        
        if not os.path.exists(local_path):
            print(f"Warning: {local_path} not found, skipping...")
            continue
        
        gcs_path = f"{gcs_prefix}{video_file}"
        blob = bucket.blob(gcs_path)
        
        print(f"Uploading {video_file} to gs://{bucket_name}/{gcs_path}...")
        blob.upload_from_filename(local_path, content_type='video/mp4')
        print(f"✓ Uploaded successfully")
    
    print(f"\nAll videos uploaded to gs://{bucket_name}/{gcs_prefix}")
    print(f"Update your config.yaml with:")
    print(f"  video_source_path: '{gcs_prefix}'")
