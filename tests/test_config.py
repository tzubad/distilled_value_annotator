"""
Unit tests for configuration module.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from config import PipelineConfig


class TestPipelineConfig:
    """Test suite for PipelineConfig class."""
    
    def test_valid_config_loading(self):
        """Test loading a valid configuration file."""
        config_data = {
            'gcs': {
                'bucket_name': 'test-bucket',
                'video_source_path': 'videos/',
                'csv_output_path': 'output/results.csv'
            },
            'model': {
                'name': 'gemini-1.5-pro-002',
                'max_retries': 3,
                'retry_delay': 30,
                'request_delay': 2
            },
            'pipeline': {
                'stage': 'both',
                'save_scripts': False
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
        
        try:
            config = PipelineConfig(config_path)
            
            assert config.gcs_bucket_name == 'test-bucket'
            assert config.video_source_path == 'videos/'
            assert config.csv_output_path == 'output/results.csv'
            assert config.model_name == 'gemini-1.5-pro-002'
            assert config.max_retries == 3
            assert config.retry_delay == 30
            assert config.request_delay == 2
            assert config.stage_to_run == 'both'
            assert config.save_scripts is False
        finally:
            Path(config_path).unlink()
    
    def test_missing_config_file(self):
        """Test error handling for missing configuration file."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig('nonexistent_config.yaml')
    
    def test_empty_config_file(self):
        """Test error handling for empty configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Configuration file is empty"):
                PipelineConfig(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_missing_required_gcs_fields(self):
        """Test validation of missing required GCS fields."""
        config_data = {
            'gcs': {
                'bucket_name': ''  # Missing required field
            },
            'model': {
                'name': 'gemini-1.5-pro-002'
            },
            'pipeline': {
                'stage': 'both'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="bucket_name is required"):
                PipelineConfig(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_invalid_stage_value(self):
        """Test validation of invalid pipeline stage."""
        config_data = {
            'gcs': {
                'bucket_name': 'test-bucket',
                'video_source_path': 'videos/',
                'csv_output_path': 'output/results.csv'
            },
            'model': {
                'name': 'gemini-1.5-pro-002'
            },
            'pipeline': {
                'stage': 'invalid_stage'  # Invalid value
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Pipeline stage must be one of"):
                PipelineConfig(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_save_scripts_without_output_path(self):
        """Test validation when save_scripts is True but script_output_path is missing."""
        config_data = {
            'gcs': {
                'bucket_name': 'test-bucket',
                'video_source_path': 'videos/',
                'csv_output_path': 'output/results.csv'
            },
            'model': {
                'name': 'gemini-1.5-pro-002'
            },
            'pipeline': {
                'stage': 'both',
                'save_scripts': True  # True but no script_output_path
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="script_output_path is required when save_scripts is True"):
                PipelineConfig(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_default_values(self):
        """Test that default values are applied correctly."""
        config_data = {
            'gcs': {
                'bucket_name': 'test-bucket',
                'video_source_path': 'videos/',
                'csv_output_path': 'output/results.csv'
            }
            # Missing model and pipeline sections - should use defaults
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config = PipelineConfig(config_path)
            
            # Check default values
            assert config.model_name == 'gemini-1.5-pro-002'
            assert config.max_retries == 4
            assert config.retry_delay == 40
            assert config.request_delay == 3
            assert config.stage_to_run == 'both'
            assert config.save_scripts is False
            assert config.safety_settings == {
                'harassment': 'BLOCK_NONE',
                'hate_speech': 'BLOCK_NONE',
                'sexually_explicit': 'BLOCK_NONE',
                'dangerous_content': 'BLOCK_NONE'
            }
        finally:
            Path(config_path).unlink()
    
    def test_invalid_safety_settings(self):
        """Test validation of invalid safety settings."""
        config_data = {
            'gcs': {
                'bucket_name': 'test-bucket',
                'video_source_path': 'videos/',
                'csv_output_path': 'output/results.csv'
            },
            'model': {
                'name': 'gemini-1.5-pro-002'
            },
            'pipeline': {
                'stage': 'both'
            },
            'safety_settings': {
                'harassment': 'INVALID_VALUE'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Safety setting"):
                PipelineConfig(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_negative_retry_values(self):
        """Test validation of negative retry values."""
        config_data = {
            'gcs': {
                'bucket_name': 'test-bucket',
                'video_source_path': 'videos/',
                'csv_output_path': 'output/results.csv'
            },
            'model': {
                'name': 'gemini-1.5-pro-002',
                'max_retries': -1  # Invalid negative value
            },
            'pipeline': {
                'stage': 'both'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
                PipelineConfig(config_path)
        finally:
            Path(config_path).unlink()
