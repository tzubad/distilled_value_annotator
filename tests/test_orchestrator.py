"""
Unit tests for orchestrator module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator:
    """Test suite for PipelineOrchestrator class."""
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_initialization(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test PipelineOrchestrator initialization."""
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = False
        mock_config.script_output_path = None
        
        orchestrator = PipelineOrchestrator(mock_config)
        
        assert orchestrator.config == mock_config
        mock_gcs_class.assert_called_once_with(bucket_name='test-bucket')
        mock_video_llm_class.assert_called_once()
        mock_annotation_llm_class.assert_called_once()
        mock_video_processor_class.assert_called_once()
        mock_script_processor_class.assert_called_once()
        mock_csv_gen_class.assert_called_once()
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_run_video_to_script_stage(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test running video-to-script stage only."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = True
        mock_config.script_output_path = 'scripts/'
        mock_config.stage_to_run = 'video_to_script'
        mock_config.video_source_path = 'videos/'
        
        mock_gcs = Mock()
        mock_gcs.list_videos.return_value = [
            'gs://bucket/video1.mp4',
            'gs://bucket/video2.mp4'
        ]
        mock_gcs_class.return_value = mock_gcs
        
        mock_processor = Mock()
        mock_processor.process_videos.return_value = (
            ['gs://bucket/scripts/script1.txt', 'gs://bucket/scripts/script2.txt'],
            []
        )
        mock_video_processor_class.return_value = mock_processor
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(mock_config)
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'video_to_script'
        assert summary['total_videos'] == 2
        assert summary['successful_videos'] == 2
        assert len(summary['failed_videos']) == 0
        assert summary['scripts_saved'] is True
        
        mock_gcs.list_videos.assert_called_once_with(prefix='videos/')
        mock_processor.process_videos.assert_called_once()
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_run_script_to_annotation_stage(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test running script-to-annotation stage only."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = False
        mock_config.script_output_path = 'scripts/'
        mock_config.stage_to_run = 'script_to_annotation'
        mock_config.csv_output_path = 'output/results.csv'
        
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.list_scripts.return_value = [
            'gs://bucket/scripts/script1.txt',
            'gs://bucket/scripts/script2.txt'
        ]
        mock_gcs_class.return_value = mock_gcs
        
        mock_processor = Mock()
        mock_processor.process_scripts.return_value = (
            [
                {'video_id': 'video1', 'Achievement': 5},
                {'video_id': 'video2', 'Hedonism': 3}
            ],
            []
        )
        mock_script_processor_class.return_value = mock_processor
        
        mock_csv_gen = Mock()
        mock_csv_gen.generate_and_save.return_value = True
        mock_csv_gen_class.return_value = mock_csv_gen
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(mock_config)
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'script_to_annotation'
        assert summary['total_scripts'] == 2
        assert summary['successful_annotations'] == 2
        assert len(summary['failed_scripts']) == 0
        assert summary['csv_saved'] is True
        assert summary['csv_path'] == 'gs://test-bucket/output/results.csv'
        
        mock_gcs.list_scripts.assert_called_once_with(prefix='scripts/')
        mock_processor.process_scripts.assert_called_once()
        mock_csv_gen.generate_and_save.assert_called_once()
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_run_complete_pipeline(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test running complete pipeline (both stages)."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = False
        mock_config.script_output_path = None
        mock_config.stage_to_run = 'both'
        mock_config.video_source_path = 'videos/'
        mock_config.csv_output_path = 'output/results.csv'
        
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.list_videos.return_value = [
            'gs://bucket/video1.mp4',
            'gs://bucket/video2.mp4'
        ]
        mock_gcs_class.return_value = mock_gcs
        
        mock_video_processor = Mock()
        mock_video_processor.process_videos.return_value = (
            ['Script 1 content', 'Script 2 content'],  # In-memory scripts
            []
        )
        mock_video_processor_class.return_value = mock_video_processor
        
        mock_script_processor = Mock()
        mock_script_processor.process_scripts.return_value = (
            [
                {'video_id': 'video1', 'Achievement': 5},
                {'video_id': 'video2', 'Hedonism': 3}
            ],
            []
        )
        mock_script_processor_class.return_value = mock_script_processor
        
        mock_csv_gen = Mock()
        mock_csv_gen.generate_and_save.return_value = True
        mock_csv_gen_class.return_value = mock_csv_gen
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(mock_config)
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'complete'
        assert summary['total_videos'] == 2
        assert summary['successful_videos'] == 2
        assert summary['total_annotations'] == 2
        assert summary['csv_saved'] is True
        assert summary['csv_path'] == 'gs://test-bucket/output/results.csv'
        
        mock_gcs.list_videos.assert_called_once()
        mock_video_processor.process_videos.assert_called_once()
        mock_script_processor.process_scripts.assert_called_once()
        mock_csv_gen.generate_and_save.assert_called_once()
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_run_with_failures(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test pipeline execution with some failures."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = False
        mock_config.script_output_path = None
        mock_config.stage_to_run = 'both'
        mock_config.video_source_path = 'videos/'
        mock_config.csv_output_path = 'output/results.csv'
        
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.list_videos.return_value = [
            'gs://bucket/video1.mp4',
            'gs://bucket/video2.mp4',
            'gs://bucket/video3.mp4'
        ]
        mock_gcs_class.return_value = mock_gcs
        
        mock_video_processor = Mock()
        mock_video_processor.process_videos.return_value = (
            ['Script 1 content', 'Script 2 content'],  # 2 successful
            ['gs://bucket/video3.mp4']  # 1 failed
        )
        mock_video_processor_class.return_value = mock_video_processor
        
        mock_script_processor = Mock()
        mock_script_processor.process_scripts.return_value = (
            [{'video_id': 'video1', 'Achievement': 5}],  # 1 successful
            ['Script 2 content']  # 1 failed
        )
        mock_script_processor_class.return_value = mock_script_processor
        
        mock_csv_gen = Mock()
        mock_csv_gen.generate_and_save.return_value = True
        mock_csv_gen_class.return_value = mock_csv_gen
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(mock_config)
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'complete'
        assert summary['total_videos'] == 3
        assert summary['successful_videos'] == 2
        assert len(summary['failed_videos']) == 1
        assert summary['total_annotations'] == 1
        assert len(summary['failed_scripts']) == 1
        assert summary['csv_saved'] is True
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_run_no_videos_found(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test pipeline when no videos are found."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = False
        mock_config.script_output_path = None
        mock_config.stage_to_run = 'both'
        mock_config.video_source_path = 'videos/'
        mock_config.csv_output_path = 'output/results.csv'
        
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.list_videos.return_value = []  # No videos
        mock_gcs_class.return_value = mock_gcs
        
        mock_video_processor = Mock()
        mock_video_processor_class.return_value = mock_video_processor
        
        mock_script_processor = Mock()
        mock_script_processor_class.return_value = mock_script_processor
        
        mock_csv_gen = Mock()
        mock_csv_gen_class.return_value = mock_csv_gen
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(mock_config)
        summary = orchestrator.run()
        
        # Verify results
        assert summary['stage'] == 'complete'
        assert summary['total_videos'] == 0
        assert summary['successful_videos'] == 0
        assert summary['total_annotations'] == 0
        assert summary['csv_saved'] is False
        
        mock_video_processor.process_videos.assert_not_called()
        mock_script_processor.process_scripts.assert_not_called()
        mock_csv_gen.generate_and_save.assert_not_called()
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_run_invalid_stage(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test error handling for invalid stage configuration."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = False
        mock_config.script_output_path = None
        mock_config.stage_to_run = 'invalid_stage'
        
        mock_gcs_class.return_value = Mock()
        mock_video_llm_class.return_value = Mock()
        mock_annotation_llm_class.return_value = Mock()
        mock_video_processor_class.return_value = Mock()
        mock_script_processor_class.return_value = Mock()
        mock_csv_gen_class.return_value = Mock()
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(mock_config)
        
        with pytest.raises(ValueError, match="Invalid stage specified"):
            orchestrator.run()
    
    @patch('orchestrator.CSVGenerator')
    @patch('orchestrator.ScriptToAnnotationProcessor')
    @patch('orchestrator.VideoToScriptProcessor')
    @patch('orchestrator.AnnotationLLMClient')
    @patch('orchestrator.VideoScriptLLMClient')
    @patch('orchestrator.GCSInterface')
    def test_csv_generation_failure(
        self,
        mock_gcs_class,
        mock_video_llm_class,
        mock_annotation_llm_class,
        mock_video_processor_class,
        mock_script_processor_class,
        mock_csv_gen_class
    ):
        """Test handling of CSV generation failure."""
        # Setup mocks
        mock_config = Mock()
        mock_config.gcs_bucket_name = 'test-bucket'
        mock_config.model_name = 'test-model'
        mock_config.safety_settings = {}
        mock_config.max_retries = 3
        mock_config.retry_delay = 10
        mock_config.request_delay = 2
        mock_config.save_scripts = False
        mock_config.script_output_path = None
        mock_config.stage_to_run = 'both'
        mock_config.video_source_path = 'videos/'
        mock_config.csv_output_path = 'output/results.csv'
        
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.list_videos.return_value = ['gs://bucket/video1.mp4']
        mock_gcs_class.return_value = mock_gcs
        
        mock_video_processor = Mock()
        mock_video_processor.process_videos.return_value = (['Script content'], [])
        mock_video_processor_class.return_value = mock_video_processor
        
        mock_script_processor = Mock()
        mock_script_processor.process_scripts.return_value = (
            [{'video_id': 'video1', 'Achievement': 5}],
            []
        )
        mock_script_processor_class.return_value = mock_script_processor
        
        mock_csv_gen = Mock()
        mock_csv_gen.generate_and_save.return_value = False  # CSV save fails
        mock_csv_gen_class.return_value = mock_csv_gen
        
        # Run orchestrator
        orchestrator = PipelineOrchestrator(mock_config)
        summary = orchestrator.run()
        
        # Verify results
        assert summary['csv_saved'] is False
        assert summary['csv_path'] is None
        assert summary['total_annotations'] == 1  # Annotations were generated
