"""
Unit tests for processor modules.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from processors import VideoToScriptProcessor, ScriptToAnnotationProcessor


class TestVideoToScriptProcessor:
    """Test suite for VideoToScriptProcessor class."""
    
    def test_initialization(self):
        """Test VideoToScriptProcessor initialization."""
        mock_llm = Mock()
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        
        processor = VideoToScriptProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=2,
            save_scripts=False,
            script_output_path=None
        )
        
        assert processor.llm_client == mock_llm
        assert processor.gcs_interface == mock_gcs
        assert processor.request_delay == 2
        assert processor.save_scripts is False
        assert processor.script_output_path is None
    
    def test_initialization_requires_output_path_when_saving(self):
        """Test that script_output_path is required when save_scripts is True."""
        mock_llm = Mock()
        mock_gcs = Mock()
        
        with pytest.raises(ValueError, match="script_output_path is required"):
            VideoToScriptProcessor(
                llm_client=mock_llm,
                gcs_interface=mock_gcs,
                request_delay=2,
                save_scripts=True,
                script_output_path=None
            )
    
    def test_process_single_video_success(self):
        """Test successful processing of a single video."""
        mock_llm = Mock()
        mock_llm.generate_script_from_video.return_value = 'Generated script content'
        mock_gcs = Mock()
        
        processor = VideoToScriptProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0,
            save_scripts=False,
            script_output_path=None
        )
        
        result = processor._process_single_video('gs://bucket/video.mp4')
        
        assert result == 'Generated script content'
        mock_llm.generate_script_from_video.assert_called_once_with('gs://bucket/video.mp4')
    
    def test_process_single_video_error(self):
        """Test error handling when processing a single video fails."""
        mock_llm = Mock()
        mock_llm.generate_script_from_video.return_value = 'Error: Failed to generate'
        mock_gcs = Mock()
        
        processor = VideoToScriptProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0,
            save_scripts=False,
            script_output_path=None
        )
        
        result = processor._process_single_video('gs://bucket/video.mp4')
        
        assert result is None
    
    @patch('processors.time.sleep')
    def test_process_videos_in_memory(self, mock_sleep):
        """Test batch processing of videos without saving scripts."""
        mock_llm = Mock()
        mock_llm.generate_script_from_video.side_effect = [
            'Script 1',
            'Script 2',
            'Error: Failed'
        ]
        mock_gcs = Mock()
        
        processor = VideoToScriptProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=1,
            save_scripts=False,
            script_output_path=None
        )
        
        video_uris = [
            'gs://bucket/video1.mp4',
            'gs://bucket/video2.mp4',
            'gs://bucket/video3.mp4'
        ]
        
        scripts, failed = processor.process_videos(video_uris)
        
        assert len(scripts) == 2
        assert 'Script 1' in scripts
        assert 'Script 2' in scripts
        assert len(failed) == 1
        assert 'gs://bucket/video3.mp4' in failed
        assert mock_sleep.call_count == 2  # Delay between videos
    
    @patch('processors.time.sleep')
    def test_process_videos_with_saving(self, mock_sleep):
        """Test batch processing of videos with script saving."""
        mock_llm = Mock()
        mock_llm.generate_script_from_video.side_effect = ['Script 1', 'Script 2']
        
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.save_script.return_value = True
        
        processor = VideoToScriptProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0,
            save_scripts=True,
            script_output_path='scripts/'
        )
        
        video_uris = [
            'gs://bucket/videos/@user_video_123.mp4',
            'gs://bucket/videos/@user_video_456.mp4'
        ]
        
        script_uris, failed = processor.process_videos(video_uris)
        
        assert len(script_uris) == 2
        assert 'gs://test-bucket/scripts/@user_video_123.txt' in script_uris
        assert 'gs://test-bucket/scripts/@user_video_456.txt' in script_uris
        assert len(failed) == 0
        assert mock_gcs.save_script.call_count == 2
    
    @patch('processors.time.sleep')
    def test_process_videos_save_failure(self, mock_sleep):
        """Test handling of GCS save failures."""
        mock_llm = Mock()
        mock_llm.generate_script_from_video.return_value = 'Script content'
        
        mock_gcs = Mock()
        mock_gcs.bucket_name = 'test-bucket'
        mock_gcs.save_script.return_value = False  # Save fails
        
        processor = VideoToScriptProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0,
            save_scripts=True,
            script_output_path='scripts/'
        )
        
        video_uris = ['gs://bucket/video.mp4']
        script_uris, failed = processor.process_videos(video_uris)
        
        assert len(script_uris) == 0
        assert len(failed) == 1
        assert 'gs://bucket/video.mp4' in failed


class TestScriptToAnnotationProcessor:
    """Test suite for ScriptToAnnotationProcessor class."""
    
    def test_initialization(self):
        """Test ScriptToAnnotationProcessor initialization."""
        mock_llm = Mock()
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=2
        )
        
        assert processor.llm_client == mock_llm
        assert processor.gcs_interface == mock_gcs
        assert processor.request_delay == 2
    
    def test_extract_json_from_markdown(self):
        """Test JSON extraction from markdown code blocks."""
        mock_llm = Mock()
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        response = '''```json
{
    "Achievement": 5,
    "Power_Dominance": 3,
    "Has_sound": true
}
```'''
        
        result = processor._extract_json_and_text(response)
        
        assert result['Achievement'] == 5
        assert result['Power_Dominance'] == 3
        assert result['Has_sound'] is True
    
    def test_extract_json_plain(self):
        """Test JSON extraction from plain JSON response."""
        mock_llm = Mock()
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        response = '{"Achievement": 4, "Hedonism": 2}'
        
        result = processor._extract_json_and_text(response)
        
        assert result['Achievement'] == 4
        assert result['Hedonism'] == 2
    
    def test_extract_json_with_notes(self):
        """Test JSON extraction with additional text notes."""
        mock_llm = Mock()
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        response = '''Some notes here

```json
{"Achievement": 5}
```

More notes'''
        
        result = processor._extract_json_and_text(response)
        
        assert result['Achievement'] == 5
        assert 'notes' in result
    
    def test_extract_json_invalid(self):
        """Test handling of invalid JSON response."""
        mock_llm = Mock()
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        response = 'This is not JSON at all'
        
        result = processor._extract_json_and_text(response)
        
        assert result == {}
    
    def test_process_single_script_from_gcs(self):
        """Test processing a script from GCS URI."""
        mock_llm = Mock()
        mock_llm.generate_annotations_from_script.return_value = '{"Achievement": 5}'
        
        mock_gcs = Mock()
        mock_gcs.read_script.return_value = 'Script content from GCS'
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        result = processor._process_single_script('gs://bucket/scripts/@user_video_123.txt')
        
        assert result is not None
        assert result['video_id'] == '@user_video_123'
        assert result['Achievement'] == 5
        mock_gcs.read_script.assert_called_once_with('gs://bucket/scripts/@user_video_123.txt')
    
    def test_process_single_script_in_memory(self):
        """Test processing an in-memory script."""
        mock_llm = Mock()
        mock_llm.generate_annotations_from_script.return_value = '{"Hedonism": 3}'
        
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        result = processor._process_single_script('This is script content')
        
        assert result is not None
        assert result['video_id'] == 'unknown'
        assert result['Hedonism'] == 3
        mock_gcs.read_script.assert_not_called()
    
    def test_process_single_script_error(self):
        """Test error handling when processing script fails."""
        mock_llm = Mock()
        mock_llm.generate_annotations_from_script.return_value = 'Error: Failed to generate'
        
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        result = processor._process_single_script('Script content')
        
        assert result is None
    
    @patch('processors.time.sleep')
    def test_process_scripts_batch(self, mock_sleep):
        """Test batch processing of scripts."""
        mock_llm = Mock()
        mock_llm.generate_annotations_from_script.side_effect = [
            '{"Achievement": 5}',
            '{"Hedonism": 3}',
            'Error: Failed'
        ]
        
        mock_gcs = Mock()
        mock_gcs.read_script.side_effect = ['Script 1', 'Script 2', 'Script 3']
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=1
        )
        
        script_uris = [
            'gs://bucket/scripts/@user_video_1.txt',
            'gs://bucket/scripts/@user_video_2.txt',
            'gs://bucket/scripts/@user_video_3.txt'
        ]
        
        annotations, failed = processor.process_scripts(script_uris)
        
        assert len(annotations) == 2
        assert annotations[0]['video_id'] == '@user_video_1'
        assert annotations[0]['Achievement'] == 5
        assert annotations[1]['video_id'] == '@user_video_2'
        assert annotations[1]['Hedonism'] == 3
        assert len(failed) == 1
        assert 'gs://bucket/scripts/@user_video_3.txt' in failed
        assert mock_sleep.call_count == 2
    
    @patch('processors.time.sleep')
    def test_process_scripts_empty_list(self, mock_sleep):
        """Test processing empty script list."""
        mock_llm = Mock()
        mock_gcs = Mock()
        
        processor = ScriptToAnnotationProcessor(
            llm_client=mock_llm,
            gcs_interface=mock_gcs,
            request_delay=0
        )
        
        annotations, failed = processor.process_scripts([])
        
        assert len(annotations) == 0
        assert len(failed) == 0
        mock_llm.generate_annotations_from_script.assert_not_called()
