"""
Unit tests for utility modules.
"""

import pytest
from unittest.mock import Mock, patch
from utils import CSVGenerator
from utils.logger import PipelineLogger


class TestCSVGenerator:
    """Test suite for CSVGenerator class."""
    
    def test_initialization(self):
        """Test CSVGenerator initialization."""
        mock_gcs = Mock()
        
        csv_gen = CSVGenerator(mock_gcs)
        
        assert csv_gen.gcs_interface == mock_gcs
    
    def test_generate_and_save_success(self):
        """Test successful CSV generation and upload."""
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.save_csv.return_value = True
        
        csv_gen = CSVGenerator(mock_gcs)
        
        annotations = [
            {
                'video_id': '@user_video_123',
                'Self_Direction_Thought': 3,
                'Achievement': 5,
                'Has_sound': True,
                'notes': 'Test note'
            },
            {
                'video_id': '@user_video_456',
                'Hedonism': 4,
                'Has_sound': False
            }
        ]
        
        result = csv_gen.generate_and_save(annotations, 'output/results.csv')
        
        assert result is True
        mock_gcs.save_csv.assert_called_once()
        
        # Verify CSV content structure
        csv_content = mock_gcs.save_csv.call_args[0][0]
        assert 'video_id' in csv_content
        assert '@user_video_123' in csv_content
        assert '@user_video_456' in csv_content
    
    def test_generate_and_save_empty_annotations(self):
        """Test handling of empty annotations list."""
        mock_gcs = Mock()
        
        csv_gen = CSVGenerator(mock_gcs)
        
        result = csv_gen.generate_and_save([], 'output/results.csv')
        
        assert result is False
        mock_gcs.save_csv.assert_not_called()
    
    def test_generate_and_save_column_ordering(self):
        """Test that CSV has correct column ordering."""
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.save_csv.return_value = True
        
        csv_gen = CSVGenerator(mock_gcs)
        
        annotations = [
            {
                'video_id': '@user_video_123',
                'Achievement': 5,
                'Power_Dominance': 3,
                'Has_sound': True
            }
        ]
        
        result = csv_gen.generate_and_save(annotations, 'output/results.csv')
        
        assert result is True
        
        # Get CSV content
        csv_content = mock_gcs.save_csv.call_args[0][0]
        lines = csv_content.strip().split('\n')
        header = lines[0]
        
        # Verify column order
        columns = [col.strip() for col in header.split(',')]  # Strip whitespace and line endings
        assert columns[0] == 'video_id'
        assert 'Self_Direction_Thought' in columns
        assert 'Achievement' in columns
        assert 'Has_sound' in columns
        assert 'notes' in columns
        
        # Verify video_id is first and Has_sound/notes are last
        assert columns.index('video_id') == 0
        assert columns.index('Has_sound') == len(columns) - 2
        assert columns.index('notes') == len(columns) - 1
    
    def test_generate_and_save_missing_columns(self):
        """Test that missing columns are filled with None."""
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.save_csv.return_value = True
        
        csv_gen = CSVGenerator(mock_gcs)
        
        # Annotation with only a few fields
        annotations = [
            {
                'video_id': '@user_video_123',
                'Achievement': 5
            }
        ]
        
        result = csv_gen.generate_and_save(annotations, 'output/results.csv')
        
        assert result is True
        
        # Verify all expected columns are present
        csv_content = mock_gcs.save_csv.call_args[0][0]
        lines = csv_content.strip().split('\n')
        header = lines[0]
        
        # Check that all 22 columns are present (video_id + 19 values + Has_sound + notes)
        columns = [col.strip() for col in header.split(',')]  # Strip whitespace and line endings
        assert len(columns) == 22
    
    def test_generate_and_save_upload_failure(self):
        """Test handling of GCS upload failure."""
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.save_csv.return_value = False
        
        csv_gen = CSVGenerator(mock_gcs)
        
        annotations = [{'video_id': '@user_video_123', 'Achievement': 5}]
        
        result = csv_gen.generate_and_save(annotations, 'output/results.csv')
        
        assert result is False
    
    def test_generate_and_save_exception(self):
        """Test handling of exceptions during CSV generation."""
        mock_gcs = Mock()
        mock_gcs.save_csv.side_effect = Exception('Upload error')
        
        csv_gen = CSVGenerator(mock_gcs)
        
        annotations = [{'video_id': '@user_video_123'}]
        
        result = csv_gen.generate_and_save(annotations, 'output/results.csv')
        
        assert result is False


class TestPipelineLogger:
    """Test suite for PipelineLogger class."""
    
    def test_initialization(self):
        """Test PipelineLogger initialization."""
        logger = PipelineLogger('TestLogger')
        
        assert logger.logger.name == 'TestLogger'
        assert logger.errors == {}
        assert logger.warnings == []
        assert logger.info_messages == []
    
    def test_log_error(self):
        """Test logging errors."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('video_to_script', 'gs://bucket/video.mp4', 'API error')
        
        assert 'video_to_script' in logger.errors
        assert len(logger.errors['video_to_script']) == 1
        
        error = logger.errors['video_to_script'][0]
        assert error['item'] == 'gs://bucket/video.mp4'
        assert error['error'] == 'API error'
        assert 'timestamp' in error
    
    def test_log_multiple_errors_same_stage(self):
        """Test logging multiple errors in the same stage."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('video_to_script', 'video1.mp4', 'Error 1')
        logger.log_error('video_to_script', 'video2.mp4', 'Error 2')
        
        assert len(logger.errors['video_to_script']) == 2
    
    def test_log_errors_different_stages(self):
        """Test logging errors in different stages."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('video_to_script', 'video.mp4', 'Error 1')
        logger.log_error('script_to_annotation', 'script.txt', 'Error 2')
        
        assert 'video_to_script' in logger.errors
        assert 'script_to_annotation' in logger.errors
        assert len(logger.errors['video_to_script']) == 1
        assert len(logger.errors['script_to_annotation']) == 1
    
    def test_log_warning(self):
        """Test logging warnings."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_warning('This is a warning')
        
        assert len(logger.warnings) == 1
        assert 'This is a warning' in logger.warnings[0]
    
    def test_log_info(self):
        """Test logging info messages."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_info('This is info')
        
        assert len(logger.info_messages) == 1
        assert 'This is info' in logger.info_messages[0]
    
    def test_get_failure_summary(self):
        """Test getting failure summary."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('stage1', 'item1', 'error1')
        logger.log_error('stage1', 'item2', 'error2')
        logger.log_error('stage2', 'item3', 'error3')
        
        summary = logger.get_failure_summary()
        
        assert 'stage1' in summary
        assert 'stage2' in summary
        assert len(summary['stage1']) == 2
        assert len(summary['stage2']) == 1
    
    def test_get_error_count(self):
        """Test getting total error count."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('stage1', 'item1', 'error1')
        logger.log_error('stage1', 'item2', 'error2')
        logger.log_error('stage2', 'item3', 'error3')
        
        count = logger.get_error_count()
        
        assert count == 3
    
    def test_get_errors_by_stage(self):
        """Test getting errors for a specific stage."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('stage1', 'item1', 'error1')
        logger.log_error('stage2', 'item2', 'error2')
        
        stage1_errors = logger.get_errors_by_stage('stage1')
        stage2_errors = logger.get_errors_by_stage('stage2')
        stage3_errors = logger.get_errors_by_stage('stage3')
        
        assert len(stage1_errors) == 1
        assert len(stage2_errors) == 1
        assert len(stage3_errors) == 0
    
    def test_has_errors(self):
        """Test checking if errors exist."""
        logger = PipelineLogger('TestLogger')
        
        assert logger.has_errors() is False
        
        logger.log_error('stage1', 'item1', 'error1')
        
        assert logger.has_errors() is True
    
    def test_clear_errors(self):
        """Test clearing all errors."""
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('stage1', 'item1', 'error1')
        logger.log_error('stage2', 'item2', 'error2')
        
        assert logger.has_errors() is True
        
        logger.clear_errors()
        
        assert logger.has_errors() is False
        assert logger.errors == {}
    
    def test_print_summary_no_errors(self, caplog):
        """Test printing summary when no errors exist."""
        import logging
        caplog.set_level(logging.INFO)
        
        logger = PipelineLogger('TestLogger')
        
        logger.print_summary()
        
        assert 'No errors logged' in caplog.text
    
    def test_print_summary_with_errors(self, caplog):
        """Test printing summary with errors."""
        import logging
        caplog.set_level(logging.INFO)
        
        logger = PipelineLogger('TestLogger')
        
        logger.log_error('video_to_script', 'video.mp4', 'API error')
        logger.log_error('script_to_annotation', 'script.txt', 'Parse error')
        
        logger.print_summary()
        
        assert 'ERROR SUMMARY' in caplog.text
        assert 'video_to_script' in caplog.text
        assert 'script_to_annotation' in caplog.text
        assert 'video.mp4' in caplog.text
        assert 'script.txt' in caplog.text
