"""
Unit tests for LLM client module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from llm import BaseLLMClient, VideoScriptLLMClient, AnnotationLLMClient


class TestBaseLLMClient:
    """Test suite for BaseLLMClient class."""
    
    @patch('llm.GenerativeModel')
    def test_initialization(self, mock_model_class):
        """Test BaseLLMClient initialization."""
        safety_settings = {
            'harassment': 'BLOCK_NONE',
            'hate_speech': 'BLOCK_NONE'
        }
        
        client = BaseLLMClient(
            model_name='test-model',
            system_instructions='Test instructions',
            safety_settings=safety_settings,
            max_retries=3,
            retry_delay=10
        )
        
        assert client.model_name == 'test-model'
        assert client.system_instructions == 'Test instructions'
        assert client.max_retries == 3
        assert client.retry_delay == 10
        mock_model_class.assert_called_once()
    
    @patch('llm.GenerativeModel')
    def test_convert_safety_settings(self, mock_model_class):
        """Test conversion of safety settings to Vertex AI format."""
        from vertexai.generative_models import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            'harassment': 'BLOCK_NONE',
            'hate_speech': 'BLOCK_ONLY_HIGH',
            'sexually_explicit': 'BLOCK_MEDIUM_AND_ABOVE',
            'dangerous_content': 'BLOCK_LOW_AND_ABOVE'
        }
        
        client = BaseLLMClient(
            model_name='test-model',
            system_instructions='Test',
            safety_settings=safety_settings,
            max_retries=3,
            retry_delay=10
        )
        
        converted = client.safety_settings
        
        assert converted[HarmCategory.HARM_CATEGORY_HARASSMENT] == HarmBlockThreshold.BLOCK_NONE
        assert converted[HarmCategory.HARM_CATEGORY_HATE_SPEECH] == HarmBlockThreshold.BLOCK_ONLY_HIGH
        assert converted[HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT] == HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        assert converted[HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT] == HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    
    @patch('llm.GenerativeModel')
    @patch('llm.time.sleep')
    def test_retry_with_backoff_success(self, mock_sleep, mock_model_class):
        """Test retry logic with successful execution."""
        client = BaseLLMClient(
            model_name='test-model',
            system_instructions='Test',
            safety_settings={},
            max_retries=3,
            retry_delay=10
        )
        
        mock_func = Mock(return_value='success')
        result = client._retry_with_backoff(mock_func, 'arg1', kwarg1='value1')
        
        assert result == 'success'
        mock_func.assert_called_once_with('arg1', kwarg1='value1')
        mock_sleep.assert_not_called()
    
    @patch('llm.GenerativeModel')
    @patch('llm.time.sleep')
    def test_retry_with_backoff_eventual_success(self, mock_sleep, mock_model_class):
        """Test retry logic with eventual success after failures."""
        client = BaseLLMClient(
            model_name='test-model',
            system_instructions='Test',
            safety_settings={},
            max_retries=3,
            retry_delay=10
        )
        
        mock_func = Mock(side_effect=[Exception('Error 1'), Exception('Error 2'), 'success'])
        result = client._retry_with_backoff(mock_func)
        
        assert result == 'success'
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2  # Sleep after first two failures
    
    @patch('llm.GenerativeModel')
    @patch('llm.time.sleep')
    def test_retry_with_backoff_all_failures(self, mock_sleep, mock_model_class):
        """Test retry logic when all attempts fail."""
        client = BaseLLMClient(
            model_name='test-model',
            system_instructions='Test',
            safety_settings={},
            max_retries=3,
            retry_delay=10
        )
        
        mock_func = Mock(side_effect=Exception('Persistent error'))
        result = client._retry_with_backoff(mock_func)
        
        assert 'Error: Failed after 3 retries' in result
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2  # No sleep after last failure


class TestVideoScriptLLMClient:
    """Test suite for VideoScriptLLMClient class."""
    
    @patch('llm.GenerativeModel')
    @patch('builtins.open', new_callable=mock_open, read_data='Video to script instructions')
    def test_initialization(self, mock_file, mock_model_class):
        """Test VideoScriptLLMClient initialization."""
        safety_settings = {'harassment': 'BLOCK_NONE'}
        
        client = VideoScriptLLMClient(
            model_name='test-model',
            safety_settings=safety_settings,
            max_retries=3,
            retry_delay=10
        )
        
        assert client.model_name == 'test-model'
        assert client.max_retries == 3
        mock_file.assert_called_once()
    
    @patch('llm.GenerativeModel')
    @patch('builtins.open', new_callable=mock_open, read_data='Instructions')
    @patch('vertexai.generative_models.Part')
    def test_generate_script_success(self, mock_part, mock_file, mock_model_class):
        """Test successful script generation from video."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = 'Generated script content'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        client = VideoScriptLLMClient(
            model_name='test-model',
            safety_settings={},
            max_retries=3,
            retry_delay=10
        )
        
        result = client.generate_script_from_video('gs://bucket/video.mp4')
        
        assert result == 'Generated script content'
        mock_model.generate_content.assert_called_once()
    
    @patch('llm.GenerativeModel')
    @patch('builtins.open', new_callable=mock_open, read_data='Instructions')
    @patch('vertexai.generative_models.Part')
    def test_generate_script_blocked_content(self, mock_part, mock_file, mock_model_class):
        """Test handling of blocked content response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = None
        mock_response.prompt_feedback = 'Content blocked'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        client = VideoScriptLLMClient(
            model_name='test-model',
            safety_settings={},
            max_retries=3,
            retry_delay=10
        )
        
        result = client.generate_script_from_video('gs://bucket/video.mp4')
        
        assert 'Error: Could not generate script' in result
        assert 'Content blocked' in result


class TestAnnotationLLMClient:
    """Test suite for AnnotationLLMClient class."""
    
    @patch('llm.GenerativeModel')
    @patch('builtins.open', new_callable=mock_open, read_data='Script to annotation instructions')
    def test_initialization(self, mock_file, mock_model_class):
        """Test AnnotationLLMClient initialization."""
        safety_settings = {'harassment': 'BLOCK_NONE'}
        
        client = AnnotationLLMClient(
            model_name='test-model',
            safety_settings=safety_settings,
            max_retries=3,
            retry_delay=10
        )
        
        assert client.model_name == 'test-model'
        assert client.max_retries == 3
        mock_file.assert_called_once()
    
    @patch('llm.GenerativeModel')
    @patch('builtins.open', new_callable=mock_open, read_data='Instructions')
    @patch('vertexai.generative_models.Part')
    def test_generate_annotations_success(self, mock_part, mock_file, mock_model_class):
        """Test successful annotation generation from script."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = '{"Achievement": 5, "Power_Dominance": 3}'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        client = AnnotationLLMClient(
            model_name='test-model',
            safety_settings={},
            max_retries=3,
            retry_delay=10
        )
        
        result = client.generate_annotations_from_script('Script text here')
        
        assert '{"Achievement": 5, "Power_Dominance": 3}' in result
        mock_model.generate_content.assert_called_once()
    
    @patch('llm.GenerativeModel')
    @patch('builtins.open', new_callable=mock_open, read_data='Instructions')
    @patch('vertexai.generative_models.Part')
    def test_generate_annotations_blocked_content(self, mock_part, mock_file, mock_model_class):
        """Test handling of blocked content response."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = None
        mock_response.prompt_feedback = 'Content blocked'
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        client = AnnotationLLMClient(
            model_name='test-model',
            safety_settings={},
            max_retries=3,
            retry_delay=10
        )
        
        result = client.generate_annotations_from_script('Script text')
        
        assert 'Error: Could not generate annotations' in result
        assert 'Content blocked' in result
