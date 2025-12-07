# Tests for EvaluationConfigLoader

import pytest
import json
import yaml
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, assume, settings

from evaluation.config_loader import EvaluationConfigLoader, ConfigValidationError
from evaluation.models import EvaluationConfig, ModelConfig


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def loader():
    """Create a fresh EvaluationConfigLoader instance."""
    return EvaluationConfigLoader()


@pytest.fixture
def valid_config_dict():
    """Create a valid configuration dictionary."""
    return {
        'ground_truth_path': '/path/to/ground_truth.csv',
        'scripts_path': '/path/to/scripts',
        'output_dir': '/path/to/output',
        'models': [
            {
                'model_type': 'gemini',
                'model_name': 'gemini-1.5-pro',
                'adapter_class': 'GeminiAdapter',
                'config': {'temperature': 0.0}
            }
        ]
    }


@pytest.fixture
def valid_yaml_file(valid_config_dict, tmp_path):
    """Create a valid YAML configuration file."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(valid_config_dict, f)
    return config_file


@pytest.fixture
def valid_json_file(valid_config_dict, tmp_path):
    """Create a valid JSON configuration file."""
    config_file = tmp_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(valid_config_dict, f)
    return config_file


# ==============================================================================
# Hypothesis Strategies
# ==============================================================================

def valid_model_config_strategy():
    """Strategy for generating valid model configurations."""
    return st.fixed_dictionaries({
        'model_type': st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
        'model_name': st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        'adapter_class': st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        'config': st.fixed_dictionaries({})
    })


def valid_config_strategy():
    """Strategy for generating valid configurations."""
    return st.fixed_dictionaries({
        'ground_truth_path': st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        'scripts_path': st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        'output_dir': st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        'models': st.lists(valid_model_config_strategy(), min_size=1, max_size=5),
        'sample_size': st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
        'random_seed': st.integers(min_value=0, max_value=2**31 - 1),
        'min_frequency_threshold': st.floats(min_value=0.0, max_value=1.0),
        'parallel_execution': st.booleans(),
        'max_workers': st.integers(min_value=1, max_value=32),
    })


def invalid_config_strategy():
    """Strategy for generating invalid configurations (missing required fields)."""
    # Generate configs missing one or more required fields
    required_fields = ['ground_truth_path', 'scripts_path', 'output_dir', 'models']
    
    return st.fixed_dictionaries({
        # Randomly omit some required fields
    }).flatmap(lambda _: st.sampled_from([
        # Missing ground_truth_path
        {
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}]
        },
        # Missing scripts_path
        {
            'ground_truth_path': '/path/to/gt.csv',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}]
        },
        # Missing output_dir
        {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}]
        },
        # Missing models
        {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
        },
        # Empty models list
        {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': []
        },
        # Invalid model (missing model_type)
        {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_name': 'test', 'adapter_class': 'Test'}]
        },
        # Invalid threshold (> 1.0)
        {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}],
            'min_frequency_threshold': 1.5
        },
        # Invalid sample_size (negative)
        {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}],
            'sample_size': -5
        },
    ]))


# ==============================================================================
# Test Class: Basic Loading
# ==============================================================================

class TestConfigLoaderBasic:
    """Tests for basic configuration loading functionality."""
    
    def test_load_yaml_file(self, loader, valid_yaml_file, valid_config_dict):
        """Test loading a valid YAML configuration file."""
        config = loader.load(valid_yaml_file)
        
        assert isinstance(config, EvaluationConfig)
        assert config.ground_truth_path == valid_config_dict['ground_truth_path']
        assert config.scripts_path == valid_config_dict['scripts_path']
        assert config.output_dir == valid_config_dict['output_dir']
        assert len(config.models) == 1
    
    def test_load_json_file(self, loader, valid_json_file, valid_config_dict):
        """Test loading a valid JSON configuration file."""
        config = loader.load(valid_json_file)
        
        assert isinstance(config, EvaluationConfig)
        assert config.ground_truth_path == valid_config_dict['ground_truth_path']
    
    def test_load_from_dict(self, loader, valid_config_dict):
        """Test loading configuration from a dictionary."""
        config = loader.load_from_dict(valid_config_dict)
        
        assert isinstance(config, EvaluationConfig)
        assert config.ground_truth_path == valid_config_dict['ground_truth_path']
    
    def test_file_not_found(self, loader):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            loader.load('/nonexistent/config.yaml')
    
    def test_unsupported_format(self, loader, tmp_path):
        """Test that ValueError is raised for unsupported file formats."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")
        
        with pytest.raises(ValueError, match="Unsupported configuration format"):
            loader.load(config_file)
    
    def test_load_yml_extension(self, loader, valid_config_dict, tmp_path):
        """Test loading a file with .yml extension."""
        config_file = tmp_path / "config.yml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_config_dict, f)
        
        config = loader.load(config_file)
        assert isinstance(config, EvaluationConfig)


# ==============================================================================
# Test Class: Validation
# ==============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_validate_valid_config(self, loader, valid_yaml_file):
        """Test that valid configuration passes validation."""
        errors = loader.validate(valid_yaml_file)
        assert errors == []
    
    def test_validate_missing_required_field(self, loader):
        """Test that missing required fields are detected."""
        config = {
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}]
        }
        
        errors = loader.validate_dict(config)
        assert any("ground_truth_path" in e for e in errors)
    
    def test_validate_empty_models_list(self, loader):
        """Test that empty models list is rejected."""
        config = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': []
        }
        
        errors = loader.validate_dict(config)
        assert any("empty" in e.lower() for e in errors)
    
    def test_validate_invalid_model_config(self, loader):
        """Test that invalid model configuration is detected."""
        config = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_name': 'test'}]  # Missing model_type and adapter_class
        }
        
        errors = loader.validate_dict(config)
        assert any("model_type" in e for e in errors)
        assert any("adapter_class" in e for e in errors)
    
    def test_validate_invalid_threshold(self, loader):
        """Test that invalid min_frequency_threshold is detected."""
        config = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}],
            'min_frequency_threshold': 1.5  # Must be between 0.0 and 1.0
        }
        
        errors = loader.validate_dict(config)
        assert any("min_frequency_threshold" in e for e in errors)
    
    def test_validate_invalid_sample_size(self, loader):
        """Test that invalid sample_size is detected."""
        config = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}],
            'sample_size': 0  # Must be positive
        }
        
        errors = loader.validate_dict(config)
        assert any("sample_size" in e for e in errors)
    
    def test_validate_invalid_max_workers(self, loader):
        """Test that invalid max_workers is detected."""
        config = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}],
            'max_workers': -1
        }
        
        errors = loader.validate_dict(config)
        assert any("max_workers" in e for e in errors)
    
    def test_validate_wrong_type_for_path(self, loader):
        """Test that wrong types for path fields are detected."""
        config = {
            'ground_truth_path': 123,  # Should be string
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'test', 'model_name': 'test', 'adapter_class': 'Test'}]
        }
        
        errors = loader.validate_dict(config)
        assert any("must be a string" in e for e in errors)
    
    def test_load_invalid_config_raises(self, loader):
        """Test that loading invalid config raises ConfigValidationError."""
        config = {
            'scripts_path': '/path/to/scripts',
            # Missing ground_truth_path, output_dir, models
        }
        
        with pytest.raises(ConfigValidationError) as exc_info:
            loader.load_from_dict(config)
        
        assert len(exc_info.value.errors) > 0


# ==============================================================================
# Test Class: Default Values
# ==============================================================================

class TestDefaultValues:
    """Tests for default value application."""
    
    def test_default_sample_size(self, loader, valid_config_dict):
        """Test that sample_size defaults to None."""
        config = loader.load_from_dict(valid_config_dict)
        assert config.sample_size is None
    
    def test_default_random_seed(self, loader, valid_config_dict):
        """Test that random_seed defaults to 42."""
        config = loader.load_from_dict(valid_config_dict)
        assert config.random_seed == 42
    
    def test_default_min_frequency_threshold(self, loader, valid_config_dict):
        """Test that min_frequency_threshold defaults to 0.05."""
        config = loader.load_from_dict(valid_config_dict)
        assert config.min_frequency_threshold == 0.05
    
    def test_default_parallel_execution(self, loader, valid_config_dict):
        """Test that parallel_execution defaults to False."""
        config = loader.load_from_dict(valid_config_dict)
        assert config.parallel_execution is False
    
    def test_default_max_workers(self, loader, valid_config_dict):
        """Test that max_workers defaults to 4."""
        config = loader.load_from_dict(valid_config_dict)
        assert config.max_workers == 4
    
    def test_default_model_config(self, loader):
        """Test that model config defaults to empty dict."""
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [
                {
                    'model_type': 'gemini',
                    'model_name': 'gemini-1.5-pro',
                    'adapter_class': 'GeminiAdapter',
                    # No 'config' key
                }
            ]
        }
        
        config = loader.load_from_dict(config_dict)
        assert config.models[0].config == {}
    
    def test_explicit_values_override_defaults(self, loader, valid_config_dict):
        """Test that explicit values override defaults."""
        valid_config_dict['random_seed'] = 123
        valid_config_dict['min_frequency_threshold'] = 0.1
        valid_config_dict['parallel_execution'] = True
        valid_config_dict['max_workers'] = 8
        valid_config_dict['sample_size'] = 100
        
        config = loader.load_from_dict(valid_config_dict)
        
        assert config.random_seed == 123
        assert config.min_frequency_threshold == 0.1
        assert config.parallel_execution is True
        assert config.max_workers == 8
        assert config.sample_size == 100
    
    def test_get_defaults(self, loader):
        """Test that get_defaults returns all default values."""
        defaults = loader.get_defaults()
        
        assert 'sample_size' in defaults
        assert 'random_seed' in defaults
        assert 'min_frequency_threshold' in defaults
        assert 'parallel_execution' in defaults
        assert 'max_workers' in defaults


# ==============================================================================
# Test Class: Model Configuration
# ==============================================================================

class TestModelConfiguration:
    """Tests for model configuration parsing."""
    
    def test_model_config_created(self, loader, valid_config_dict):
        """Test that ModelConfig objects are created correctly."""
        config = loader.load_from_dict(valid_config_dict)
        
        assert len(config.models) == 1
        model = config.models[0]
        
        assert isinstance(model, ModelConfig)
        assert model.model_type == 'gemini'
        assert model.model_name == 'gemini-1.5-pro'
        assert model.adapter_class == 'GeminiAdapter'
        assert model.config == {'temperature': 0.0}
    
    def test_multiple_models(self, loader):
        """Test loading configuration with multiple models."""
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [
                {
                    'model_type': 'gemini',
                    'model_name': 'gemini-1.5-pro',
                    'adapter_class': 'GeminiAdapter'
                },
                {
                    'model_type': 'mlm',
                    'model_name': 'roberta-base',
                    'adapter_class': 'RoBERTaAdapter',
                    'config': {'max_length': 512}
                },
                {
                    'model_type': 'mlm',
                    'model_name': 'deberta-v3-base',
                    'adapter_class': 'DeBERTaAdapter',
                    'config': {'max_length': 768}
                }
            ]
        }
        
        config = loader.load_from_dict(config_dict)
        
        assert len(config.models) == 3
        assert config.models[0].model_name == 'gemini-1.5-pro'
        assert config.models[1].model_name == 'roberta-base'
        assert config.models[2].model_name == 'deberta-v3-base'
    
    def test_model_specific_config_preserved(self, loader):
        """Test that model-specific config parameters are preserved."""
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [
                {
                    'model_type': 'gemini',
                    'model_name': 'gemini-1.5-pro',
                    'adapter_class': 'GeminiAdapter',
                    'config': {
                        'temperature': 0.0,
                        'max_output_tokens': 4096,
                        'top_p': 0.95,
                        'custom_param': 'custom_value'
                    }
                }
            ]
        }
        
        config = loader.load_from_dict(config_dict)
        model_config = config.models[0].config
        
        assert model_config['temperature'] == 0.0
        assert model_config['max_output_tokens'] == 4096
        assert model_config['top_p'] == 0.95
        assert model_config['custom_param'] == 'custom_value'


# ==============================================================================
# Test Class: Serialization
# ==============================================================================

class TestSerialization:
    """Tests for configuration serialization."""
    
    def test_to_dict(self, loader, valid_config_dict):
        """Test converting EvaluationConfig to dictionary."""
        config = loader.load_from_dict(valid_config_dict)
        result = EvaluationConfigLoader.to_dict(config)
        
        assert isinstance(result, dict)
        assert result['ground_truth_path'] == valid_config_dict['ground_truth_path']
        assert result['scripts_path'] == valid_config_dict['scripts_path']
        assert result['output_dir'] == valid_config_dict['output_dir']
    
    def test_to_yaml(self, loader, valid_config_dict):
        """Test converting EvaluationConfig to YAML string."""
        config = loader.load_from_dict(valid_config_dict)
        yaml_str = EvaluationConfigLoader.to_yaml(config)
        
        assert isinstance(yaml_str, str)
        assert 'ground_truth_path' in yaml_str
        
        # Verify it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert parsed['ground_truth_path'] == valid_config_dict['ground_truth_path']
    
    def test_to_json(self, loader, valid_config_dict):
        """Test converting EvaluationConfig to JSON string."""
        config = loader.load_from_dict(valid_config_dict)
        json_str = EvaluationConfigLoader.to_json(config)
        
        assert isinstance(json_str, str)
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed['ground_truth_path'] == valid_config_dict['ground_truth_path']
    
    def test_round_trip_yaml(self, loader, valid_config_dict, tmp_path):
        """Test that config can be saved to YAML and loaded back."""
        # Load original
        config1 = loader.load_from_dict(valid_config_dict)
        
        # Save to YAML
        yaml_str = EvaluationConfigLoader.to_yaml(config1)
        yaml_file = tmp_path / "roundtrip.yaml"
        yaml_file.write_text(yaml_str)
        
        # Load back
        config2 = loader.load(yaml_file)
        
        assert config1.ground_truth_path == config2.ground_truth_path
        assert config1.scripts_path == config2.scripts_path
        assert config1.random_seed == config2.random_seed
        assert len(config1.models) == len(config2.models)
    
    def test_round_trip_json(self, loader, valid_config_dict, tmp_path):
        """Test that config can be saved to JSON and loaded back."""
        # Load original
        config1 = loader.load_from_dict(valid_config_dict)
        
        # Save to JSON
        json_str = EvaluationConfigLoader.to_json(config1)
        json_file = tmp_path / "roundtrip.json"
        json_file.write_text(json_str)
        
        # Load back
        config2 = loader.load(json_file)
        
        assert config1.ground_truth_path == config2.ground_truth_path
        assert config1.scripts_path == config2.scripts_path


# ==============================================================================
# Property Tests
# ==============================================================================

class TestConfigValidationProperty:
    """
    Property 19: Configuration Validation
    
    Validates: Requirements 9.4
    Verify: Invalid configs rejected before evaluation
    """
    
    @given(invalid_config_strategy())
    @settings(max_examples=50)
    def test_invalid_configs_always_rejected(self, invalid_config):
        """
        Property 19: Any configuration that violates the schema 
        should be rejected with appropriate error messages.
        """
        loader = EvaluationConfigLoader()
        errors = loader.validate_dict(invalid_config)
        
        # Invalid configs should always have at least one error
        assert len(errors) > 0, f"Expected errors for invalid config: {invalid_config}"
    
    def test_all_required_fields_validated(self):
        """Test that all required fields are validated."""
        loader = EvaluationConfigLoader()
        
        # Test each required field individually
        required_fields = ['ground_truth_path', 'scripts_path', 'output_dir', 'models']
        
        for field in required_fields:
            # Create config missing just this field
            config = {
                'ground_truth_path': '/path/to/gt.csv',
                'scripts_path': '/path/to/scripts',
                'output_dir': '/path/to/output',
                'models': [{'model_type': 't', 'model_name': 't', 'adapter_class': 'T'}]
            }
            del config[field]
            
            errors = loader.validate_dict(config)
            assert any(field in e for e in errors), f"Expected error for missing {field}"
    
    def test_invalid_types_rejected(self):
        """Test that invalid types for all fields are rejected."""
        loader = EvaluationConfigLoader()
        
        # Test invalid type for each optional numeric field
        type_tests = [
            ('sample_size', 'not_an_int'),
            ('random_seed', 'not_an_int'),
            ('min_frequency_threshold', 'not_a_float'),
            ('parallel_execution', 'not_a_bool'),
            ('max_workers', 'not_an_int'),
        ]
        
        for field, invalid_value in type_tests:
            config = {
                'ground_truth_path': '/path/to/gt.csv',
                'scripts_path': '/path/to/scripts',
                'output_dir': '/path/to/output',
                'models': [{'model_type': 't', 'model_name': 't', 'adapter_class': 'T'}],
                field: invalid_value
            }
            
            errors = loader.validate_dict(config)
            assert any(field in e for e in errors), f"Expected error for invalid type in {field}"


class TestDefaultValueApplicationProperty:
    """
    Property 20: Default Value Application
    
    Validates: Requirements 9.5
    Verify: Missing optional params get defaults
    """
    
    @given(st.sampled_from(['sample_size', 'random_seed', 'min_frequency_threshold', 
                            'parallel_execution', 'max_workers']))
    def test_missing_optional_params_get_defaults(self, field_to_omit):
        """
        Property 20: When an optional parameter is not specified,
        it should receive its documented default value.
        """
        loader = EvaluationConfigLoader()
        defaults = loader.get_defaults()
        
        # Create minimal valid config
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 'gemini', 'model_name': 'test', 'adapter_class': 'Test'}]
        }
        
        # Load config (without the optional field)
        config = loader.load_from_dict(config_dict)
        
        # Verify the default was applied
        actual_value = getattr(config, field_to_omit)
        expected_default = defaults[field_to_omit]
        
        assert actual_value == expected_default, \
            f"Expected {field_to_omit} to default to {expected_default}, got {actual_value}"
    
    def test_all_defaults_documented(self):
        """Test that all optional parameters have documented defaults."""
        loader = EvaluationConfigLoader()
        defaults = loader.get_defaults()
        
        # These are all the optional parameters
        optional_params = [
            'sample_size',
            'random_seed', 
            'min_frequency_threshold',
            'parallel_execution',
            'max_workers'
        ]
        
        for param in optional_params:
            assert param in defaults, f"Missing default for {param}"
    
    def test_model_config_default_applied(self):
        """Test that model config defaults to empty dict when not specified."""
        loader = EvaluationConfigLoader()
        
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [
                {'model_type': 'gemini', 'model_name': 'test', 'adapter_class': 'Test'}
                # No 'config' key
            ]
        }
        
        config = loader.load_from_dict(config_dict)
        
        assert config.models[0].config == {}, \
            "Expected model config to default to empty dict"


class TestParameterPropagationProperty:
    """
    Property 18: Configuration Parameter Propagation
    
    Validates: Requirements 9.3
    Verify: Adapter receives config params
    """
    
    def test_model_config_propagated_to_model_config_object(self):
        """
        Property 18: Model-specific configuration parameters should be
        propagated to the ModelConfig object for use by adapters.
        """
        loader = EvaluationConfigLoader()
        
        # Define model-specific parameters
        model_params = {
            'temperature': 0.5,
            'max_output_tokens': 2048,
            'top_p': 0.9,
            'custom_param': 'custom_value',
            'nested': {'key': 'value'}
        }
        
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [
                {
                    'model_type': 'gemini',
                    'model_name': 'gemini-1.5-pro',
                    'adapter_class': 'GeminiAdapter',
                    'config': model_params
                }
            ]
        }
        
        config = loader.load_from_dict(config_dict)
        
        # Verify all parameters are propagated
        model_config = config.models[0]
        
        for key, value in model_params.items():
            assert key in model_config.config, \
                f"Expected {key} to be propagated to ModelConfig"
            assert model_config.config[key] == value, \
                f"Expected {key}={value}, got {model_config.config[key]}"
    
    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
        values=st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.text(max_size=50)
        ),
        min_size=1,
        max_size=10
    ))
    @settings(max_examples=30)
    def test_arbitrary_config_params_propagated(self, arbitrary_params):
        """
        Property 18: Any arbitrary configuration parameters should be
        propagated through to the ModelConfig object unchanged.
        """
        loader = EvaluationConfigLoader()
        
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [
                {
                    'model_type': 'test',
                    'model_name': 'test-model',
                    'adapter_class': 'TestAdapter',
                    'config': arbitrary_params
                }
            ]
        }
        
        config = loader.load_from_dict(config_dict)
        
        # Verify all arbitrary params are present
        for key, value in arbitrary_params.items():
            assert key in config.models[0].config
            assert config.models[0].config[key] == value
    
    def test_multiple_models_have_independent_configs(self):
        """Test that multiple models maintain independent configurations."""
        loader = EvaluationConfigLoader()
        
        config_dict = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [
                {
                    'model_type': 'gemini',
                    'model_name': 'model1',
                    'adapter_class': 'Adapter1',
                    'config': {'param1': 'value1'}
                },
                {
                    'model_type': 'mlm',
                    'model_name': 'model2',
                    'adapter_class': 'Adapter2',
                    'config': {'param2': 'value2'}
                }
            ]
        }
        
        config = loader.load_from_dict(config_dict)
        
        # Verify configs are independent
        assert config.models[0].config == {'param1': 'value1'}
        assert config.models[1].config == {'param2': 'value2'}
        
        # Verify changing one doesn't affect the other
        config.models[0].config['new_param'] = 'new_value'
        assert 'new_param' not in config.models[1].config


# ==============================================================================
# Test Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_config_file(self, loader, tmp_path):
        """Test handling of empty configuration file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        
        errors = loader.validate(config_file)
        assert len(errors) > 0
    
    def test_null_config_file(self, loader, tmp_path):
        """Test handling of null configuration file."""
        config_file = tmp_path / "null.yaml"
        config_file.write_text("null")
        
        errors = loader.validate(config_file)
        assert len(errors) > 0
    
    def test_empty_string_paths(self, loader):
        """Test that empty string paths are rejected."""
        config_dict = {
            'ground_truth_path': '',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 't', 'model_name': 't', 'adapter_class': 'T'}]
        }
        
        errors = loader.validate_dict(config_dict)
        assert any("empty" in e.lower() or "ground_truth_path" in e for e in errors)
    
    def test_whitespace_only_paths(self, loader):
        """Test that whitespace-only paths are rejected."""
        config_dict = {
            'ground_truth_path': '   ',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 't', 'model_name': 't', 'adapter_class': 'T'}]
        }
        
        errors = loader.validate_dict(config_dict)
        assert len(errors) > 0
    
    def test_boundary_threshold_values(self, loader):
        """Test boundary values for min_frequency_threshold."""
        base_config = {
            'ground_truth_path': '/path/to/gt.csv',
            'scripts_path': '/path/to/scripts',
            'output_dir': '/path/to/output',
            'models': [{'model_type': 't', 'model_name': 't', 'adapter_class': 'T'}]
        }
        
        # Test 0.0 - should be valid
        config = dict(base_config, min_frequency_threshold=0.0)
        errors = loader.validate_dict(config)
        assert not any("min_frequency_threshold" in e for e in errors)
        
        # Test 1.0 - should be valid
        config = dict(base_config, min_frequency_threshold=1.0)
        errors = loader.validate_dict(config)
        assert not any("min_frequency_threshold" in e for e in errors)
        
        # Test just below 0.0 - should be invalid
        config = dict(base_config, min_frequency_threshold=-0.001)
        errors = loader.validate_dict(config)
        assert any("min_frequency_threshold" in e for e in errors)
        
        # Test just above 1.0 - should be invalid
        config = dict(base_config, min_frequency_threshold=1.001)
        errors = loader.validate_dict(config)
        assert any("min_frequency_threshold" in e for e in errors)
    
    def test_config_with_path_object(self, loader, valid_yaml_file):
        """Test loading with Path object instead of string."""
        from pathlib import Path
        config = loader.load(Path(valid_yaml_file))
        assert isinstance(config, EvaluationConfig)
