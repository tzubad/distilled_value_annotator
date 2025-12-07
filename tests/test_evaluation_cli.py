"""
Tests for the Model Evaluation CLI (run_evaluation.py).

This module tests the command-line interface for the evaluation module.
"""

import pytest
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_evaluation import (
    parse_arguments,
    setup_logging,
    register_default_adapters,
    run_evaluation,
    main,
)
from evaluation import EvaluationOrchestrator


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a sample YAML configuration file."""
    config_content = """
ground_truth_path: "{gt_path}"
scripts_path: "{scripts_path}"
output_dir: "{output_dir}"
models:
  - model_type: llm
    model_name: test_gemini
    adapter_class: GeminiAdapter
    config:
      model_id: gemini-1.5-flash
"""
    gt_path = tmp_path / "ground_truth.csv"
    scripts_path = tmp_path / "scripts"
    output_dir = tmp_path / "output"
    
    # Create directories
    scripts_path.mkdir()
    output_dir.mkdir()
    
    # Create minimal ground truth
    gt_path.write_text("video_id,Achievement,Benevolence,Conformity,Hedonism,Power,Security,Self_direction,Stimulation,Tradition,Universalism,Face,Humility,Societal_security,Personal_security,Routine,Rules,Caring,Dependability,Concern_for_others\nvideo_001,1,0,-1,2,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0\n")
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content.format(
        gt_path=str(gt_path).replace("\\", "/"),
        scripts_path=str(scripts_path).replace("\\", "/"),
        output_dir=str(output_dir).replace("\\", "/"),
    ))
    
    return config_path


@pytest.fixture
def sample_config_json(tmp_path):
    """Create a sample JSON configuration file."""
    gt_path = tmp_path / "ground_truth.csv"
    scripts_path = tmp_path / "scripts"
    output_dir = tmp_path / "output"
    
    # Create directories
    scripts_path.mkdir()
    output_dir.mkdir()
    
    # Create minimal ground truth
    gt_path.write_text("video_id,Achievement,Benevolence,Conformity,Hedonism,Power,Security,Self_direction,Stimulation,Tradition,Universalism,Face,Humility,Societal_security,Personal_security,Routine,Rules,Caring,Dependability,Concern_for_others\nvideo_001,1,0,-1,2,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0\n")
    
    config = {
        "ground_truth_path": str(gt_path).replace("\\", "/"),
        "scripts_path": str(scripts_path).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "models": [
            {
                "model_type": "llm",
                "model_name": "test_model",
                "adapter_class": "GeminiAdapter",
                "config": {}
            }
        ]
    }
    
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    
    return config_path


# ==============================================================================
# Test Class: Argument Parsing
# ==============================================================================

class TestArgumentParsing:
    """Tests for CLI argument parsing."""
    
    def test_required_config_argument(self):
        """Test that --config is required."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run_evaluation.py']):
                parse_arguments()
    
    def test_config_argument_short_form(self):
        """Test -c short form for config."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml']):
            args = parse_arguments()
            assert args.config == 'config.yaml'
    
    def test_config_argument_long_form(self):
        """Test --config long form."""
        with patch('sys.argv', ['run_evaluation.py', '--config', 'config.yaml']):
            args = parse_arguments()
            assert args.config == 'config.yaml'
    
    def test_verbose_flag(self):
        """Test --verbose flag."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml', '--verbose']):
            args = parse_arguments()
            assert args.verbose is True
    
    def test_quiet_flag(self):
        """Test --quiet flag."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml', '--quiet']):
            args = parse_arguments()
            assert args.quiet is True
    
    def test_dry_run_flag(self):
        """Test --dry-run flag."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml', '--dry-run']):
            args = parse_arguments()
            assert args.dry_run is True
    
    def test_output_dir_override(self):
        """Test --output-dir override."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml', '-o', '/custom/output']):
            args = parse_arguments()
            assert args.output_dir == '/custom/output'
    
    def test_skip_reports_flag(self):
        """Test --skip-reports flag."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml', '--skip-reports']):
            args = parse_arguments()
            assert args.skip_reports is True
    
    def test_models_filter(self):
        """Test --models filter."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml', '--models', 'model1', 'model2']):
            args = parse_arguments()
            assert args.models == ['model1', 'model2']
    
    def test_default_values(self):
        """Test default argument values."""
        with patch('sys.argv', ['run_evaluation.py', '-c', 'config.yaml']):
            args = parse_arguments()
            assert args.verbose is False
            assert args.quiet is False
            assert args.dry_run is False
            assert args.output_dir is None
            assert args.skip_reports is False
            assert args.models is None


# ==============================================================================
# Test Class: Logging Setup
# ==============================================================================

class TestLoggingSetup:
    """Tests for logging configuration."""
    
    def test_default_logging_level(self):
        """Test default logging level is INFO."""
        import logging
        # Reset root logger
        logging.root.setLevel(logging.NOTSET)
        logging.root.handlers = []
        
        logger = setup_logging()
        # Root logger should be set to INFO
        assert logging.getLogger().level == logging.INFO
    
    def test_verbose_logging_level(self):
        """Test verbose mode sets DEBUG level."""
        import logging
        # Reset root logger
        logging.root.setLevel(logging.NOTSET)
        logging.root.handlers = []
        
        logger = setup_logging(verbose=True)
        # Root logger should be set to DEBUG
        assert logging.getLogger().level == logging.DEBUG
    
    def test_quiet_logging_level(self):
        """Test quiet mode sets WARNING level."""
        import logging
        # Reset root logger
        logging.root.setLevel(logging.NOTSET)
        logging.root.handlers = []
        
        logger = setup_logging(quiet=True)
        assert logging.getLogger().level == logging.WARNING


# ==============================================================================
# Test Class: Adapter Registration
# ==============================================================================

class TestAdapterRegistration:
    """Tests for default adapter registration."""
    
    def test_register_default_adapters(self):
        """Test that default adapters are registered."""
        # Clear any existing registrations
        EvaluationOrchestrator._adapter_registry = {}
        
        register_default_adapters()
        
        assert "GeminiAdapter" in EvaluationOrchestrator._adapter_registry
        assert "MLMAdapter" in EvaluationOrchestrator._adapter_registry


# ==============================================================================
# Test Class: Dry Run Mode
# ==============================================================================

class TestDryRunMode:
    """Tests for dry run functionality."""
    
    def test_dry_run_validates_config(self, sample_config_yaml):
        """Test dry run validates configuration without running evaluation."""
        with patch('sys.argv', ['run_evaluation.py', '-c', str(sample_config_yaml), '--dry-run']):
            args = parse_arguments()
            logger = setup_logging()
            
            result = run_evaluation(args, logger)
            
            assert result == 0  # Success
    
    def test_dry_run_fails_invalid_config(self, tmp_path):
        """Test dry run fails with invalid configuration."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: content:")
        
        with patch('sys.argv', ['run_evaluation.py', '-c', str(config_path), '--dry-run']):
            args = parse_arguments()
            logger = setup_logging()
            
            result = run_evaluation(args, logger)
            
            assert result != 0  # Failure


# ==============================================================================
# Test Class: Error Handling
# ==============================================================================

class TestErrorHandling:
    """Tests for CLI error handling."""
    
    def test_missing_config_file(self, tmp_path):
        """Test error when config file doesn't exist."""
        missing_path = tmp_path / "nonexistent.yaml"
        
        with patch('sys.argv', ['run_evaluation.py', '-c', str(missing_path)]):
            args = parse_arguments()
            logger = setup_logging()
            
            result = run_evaluation(args, logger)
            
            assert result == 1  # Failure
    
    def test_invalid_config_format(self, tmp_path):
        """Test error with invalid config format."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("not: valid: yaml: {{")
        
        with patch('sys.argv', ['run_evaluation.py', '-c', str(config_path)]):
            args = parse_arguments()
            logger = setup_logging()
            
            result = run_evaluation(args, logger)
            
            assert result == 1  # Failure


# ==============================================================================
# Test Class: Main Function
# ==============================================================================

class TestMainFunction:
    """Tests for main entry point."""
    
    def test_keyboard_interrupt_handling(self, sample_config_yaml):
        """Test graceful handling of keyboard interrupt."""
        with patch('sys.argv', ['run_evaluation.py', '-c', str(sample_config_yaml)]):
            with patch('run_evaluation.run_evaluation', side_effect=KeyboardInterrupt):
                result = main()
                assert result == 130  # Standard exit code for SIGINT
    
    def test_main_with_dry_run(self, sample_config_yaml):
        """Test main function with dry run."""
        with patch('sys.argv', ['run_evaluation.py', '-c', str(sample_config_yaml), '--dry-run', '-q']):
            result = main()
            assert result == 0


# ==============================================================================
# Test Class: Output Directory Override
# ==============================================================================

class TestOutputDirOverride:
    """Tests for output directory override functionality."""
    
    def test_output_dir_override_applied(self, sample_config_yaml, tmp_path):
        """Test that --output-dir overrides config setting."""
        custom_output = tmp_path / "custom_output"
        custom_output.mkdir()
        
        with patch('sys.argv', [
            'run_evaluation.py', 
            '-c', str(sample_config_yaml), 
            '-o', str(custom_output),
            '--dry-run'
        ]):
            args = parse_arguments()
            logger = setup_logging()
            
            # Capture the config after loading
            from evaluation import EvaluationConfigLoader
            loader = EvaluationConfigLoader()
            config = loader.load(str(sample_config_yaml))
            
            # Apply override
            if args.output_dir:
                config.output_dir = args.output_dir
            
            assert config.output_dir == str(custom_output)
