# Configuration loader for the evaluation module

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict

from evaluation.models import EvaluationConfig, ModelConfig


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


class EvaluationConfigLoader:
    """
    Loader for evaluation configuration files.
    
    Supports both YAML and JSON configuration formats. Validates configuration,
    applies default values, and returns EvaluationConfig objects.
    
    Default values:
        - sample_size: None (use all videos)
        - random_seed: 42
        - min_frequency_threshold: 0.05
        - parallel_execution: False
        - max_workers: 4
    """
    
    # Default values for optional configuration parameters
    DEFAULTS = {
        'sample_size': None,
        'random_seed': 42,
        'min_frequency_threshold': 0.05,
        'parallel_execution': False,
        'max_workers': 4,
    }
    
    # Default values for model configuration
    MODEL_DEFAULTS = {
        'config': {},
    }
    
    # Required top-level fields
    REQUIRED_FIELDS = ['ground_truth_path', 'scripts_path', 'output_dir', 'models']
    
    # Required model fields
    REQUIRED_MODEL_FIELDS = ['model_type', 'model_name', 'adapter_class']
    
    def __init__(self):
        """Initialize the configuration loader."""
        self._validation_errors: List[str] = []
    
    def load(self, config_path: Union[str, Path]) -> EvaluationConfig:
        """
        Load and validate configuration from a YAML or JSON file.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
            
        Returns:
            Validated EvaluationConfig object
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ConfigValidationError: If the configuration is invalid
            ValueError: If the file format is not supported
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load raw configuration
        raw_config = self._load_file(config_path)
        
        # Validate and apply defaults
        validated_config = self._validate_and_apply_defaults(raw_config)
        
        # Create EvaluationConfig object
        return self._create_config(validated_config)
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> EvaluationConfig:
        """
        Load and validate configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Validated EvaluationConfig object
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        validated_config = self._validate_and_apply_defaults(config_dict)
        return self._create_config(validated_config)
    
    def validate(self, config_path: Union[str, Path]) -> List[str]:
        """
        Validate a configuration file without loading it.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            List of validation error messages (empty if valid)
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            return [f"Configuration file not found: {config_path}"]
        
        try:
            raw_config = self._load_file(config_path)
        except Exception as e:
            return [f"Failed to parse configuration file: {e}"]
        
        return self._get_validation_errors(raw_config)
    
    def validate_dict(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            List of validation error messages (empty if valid)
        """
        return self._get_validation_errors(config_dict)
    
    def _load_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ValueError: If the file format is not supported
        """
        suffix = config_path.suffix.lower()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix in ('.yaml', '.yml'):
                config = yaml.safe_load(f)
            elif suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration format: {suffix}. "
                    f"Supported formats: .yaml, .yml, .json"
                )
        
        if config is None:
            return {}
        
        return config
    
    def _validate_and_apply_defaults(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and apply default values.
        
        Args:
            raw_config: Raw configuration dictionary
            
        Returns:
            Configuration dictionary with defaults applied
            
        Raises:
            ConfigValidationError: If validation fails
        """
        errors = self._get_validation_errors(raw_config)
        
        if errors:
            raise ConfigValidationError(errors)
        
        # Apply defaults
        config = self._apply_defaults(raw_config)
        
        return config
    
    def _get_validation_errors(self, raw_config: Dict[str, Any]) -> List[str]:
        """
        Get all validation errors for a configuration.
        
        Args:
            raw_config: Raw configuration dictionary
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check if config is a dictionary
        if not isinstance(raw_config, dict):
            return ["Configuration must be a dictionary/object"]
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in raw_config:
                errors.append(f"Missing required field: '{field}'")
        
        # Skip further validation if required fields are missing
        if errors:
            return errors
        
        # Validate paths (must be strings)
        for path_field in ['ground_truth_path', 'scripts_path', 'output_dir']:
            value = raw_config.get(path_field)
            if not isinstance(value, str):
                errors.append(f"'{path_field}' must be a string, got {type(value).__name__}")
            elif not value.strip():
                errors.append(f"'{path_field}' cannot be empty")
        
        # Validate models list
        models = raw_config.get('models', [])
        if not isinstance(models, list):
            errors.append(f"'models' must be a list, got {type(models).__name__}")
        elif len(models) == 0:
            errors.append("'models' list cannot be empty")
        else:
            # Validate each model configuration
            for i, model in enumerate(models):
                model_errors = self._validate_model_config(model, i)
                errors.extend(model_errors)
        
        # Validate optional numeric fields
        if 'sample_size' in raw_config and raw_config['sample_size'] is not None:
            sample_size = raw_config['sample_size']
            if not isinstance(sample_size, int):
                errors.append(f"'sample_size' must be an integer, got {type(sample_size).__name__}")
            elif sample_size <= 0:
                errors.append(f"'sample_size' must be positive, got {sample_size}")
        
        if 'random_seed' in raw_config:
            random_seed = raw_config['random_seed']
            if not isinstance(random_seed, int):
                errors.append(f"'random_seed' must be an integer, got {type(random_seed).__name__}")
        
        if 'min_frequency_threshold' in raw_config:
            threshold = raw_config['min_frequency_threshold']
            if not isinstance(threshold, (int, float)):
                errors.append(f"'min_frequency_threshold' must be a number, got {type(threshold).__name__}")
            elif threshold < 0.0 or threshold > 1.0:
                errors.append(f"'min_frequency_threshold' must be between 0.0 and 1.0, got {threshold}")
        
        if 'parallel_execution' in raw_config:
            parallel = raw_config['parallel_execution']
            if not isinstance(parallel, bool):
                errors.append(f"'parallel_execution' must be a boolean, got {type(parallel).__name__}")
        
        if 'max_workers' in raw_config:
            max_workers = raw_config['max_workers']
            if not isinstance(max_workers, int):
                errors.append(f"'max_workers' must be an integer, got {type(max_workers).__name__}")
            elif max_workers <= 0:
                errors.append(f"'max_workers' must be positive, got {max_workers}")
        
        return errors
    
    def _validate_model_config(self, model: Any, index: int) -> List[str]:
        """
        Validate a single model configuration.
        
        Args:
            model: Model configuration dictionary
            index: Index of the model in the list (for error messages)
            
        Returns:
            List of validation error messages
        """
        errors = []
        prefix = f"models[{index}]"
        
        if not isinstance(model, dict):
            return [f"{prefix}: must be a dictionary/object, got {type(model).__name__}"]
        
        # Check required model fields
        for field in self.REQUIRED_MODEL_FIELDS:
            if field not in model:
                errors.append(f"{prefix}: missing required field '{field}'")
        
        # Validate field types
        for field in ['model_type', 'model_name', 'adapter_class']:
            if field in model:
                value = model[field]
                if not isinstance(value, str):
                    errors.append(f"{prefix}.{field}: must be a string, got {type(value).__name__}")
                elif not value.strip():
                    errors.append(f"{prefix}.{field}: cannot be empty")
        
        # Validate optional config field
        if 'config' in model:
            config = model['config']
            if not isinstance(config, dict):
                errors.append(f"{prefix}.config: must be a dictionary/object, got {type(config).__name__}")
        
        return errors
    
    def _apply_defaults(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values to a configuration.
        
        Args:
            raw_config: Raw configuration dictionary
            
        Returns:
            Configuration dictionary with defaults applied
        """
        config = dict(raw_config)
        
        # Apply top-level defaults
        for key, default_value in self.DEFAULTS.items():
            if key not in config:
                config[key] = default_value
        
        # Apply model-level defaults
        models = []
        for model in config.get('models', []):
            model_config = dict(model)
            for key, default_value in self.MODEL_DEFAULTS.items():
                if key not in model_config:
                    model_config[key] = default_value
            models.append(model_config)
        config['models'] = models
        
        return config
    
    def _create_config(self, validated_config: Dict[str, Any]) -> EvaluationConfig:
        """
        Create an EvaluationConfig object from a validated configuration.
        
        Args:
            validated_config: Validated configuration dictionary
            
        Returns:
            EvaluationConfig object
        """
        # Create ModelConfig objects
        model_configs = []
        for model_dict in validated_config['models']:
            model_config = ModelConfig(
                model_type=model_dict['model_type'],
                model_name=model_dict['model_name'],
                adapter_class=model_dict['adapter_class'],
                config=model_dict.get('config', {}),
            )
            model_configs.append(model_config)
        
        # Create EvaluationConfig
        return EvaluationConfig(
            ground_truth_path=validated_config['ground_truth_path'],
            scripts_path=validated_config['scripts_path'],
            output_dir=validated_config['output_dir'],
            models=model_configs,
            sample_size=validated_config.get('sample_size'),
            random_seed=validated_config.get('random_seed', self.DEFAULTS['random_seed']),
            min_frequency_threshold=validated_config.get('min_frequency_threshold', self.DEFAULTS['min_frequency_threshold']),
            parallel_execution=validated_config.get('parallel_execution', self.DEFAULTS['parallel_execution']),
            max_workers=validated_config.get('max_workers', self.DEFAULTS['max_workers']),
        )
    
    def get_defaults(self) -> Dict[str, Any]:
        """
        Get a copy of the default values.
        
        Returns:
            Dictionary of default values
        """
        return dict(self.DEFAULTS)
    
    def get_model_defaults(self) -> Dict[str, Any]:
        """
        Get a copy of the model default values.
        
        Returns:
            Dictionary of model default values
        """
        return dict(self.MODEL_DEFAULTS)
    
    @staticmethod
    def to_dict(config: EvaluationConfig) -> Dict[str, Any]:
        """
        Convert an EvaluationConfig to a dictionary.
        
        Args:
            config: EvaluationConfig object
            
        Returns:
            Configuration dictionary
        """
        return asdict(config)
    
    @staticmethod
    def to_yaml(config: EvaluationConfig) -> str:
        """
        Convert an EvaluationConfig to YAML string.
        
        Args:
            config: EvaluationConfig object
            
        Returns:
            YAML string representation
        """
        return yaml.dump(asdict(config), default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def to_json(config: EvaluationConfig, indent: int = 2) -> str:
        """
        Convert an EvaluationConfig to JSON string.
        
        Args:
            config: EvaluationConfig object
            indent: JSON indentation level
            
        Returns:
            JSON string representation
        """
        return json.dumps(asdict(config), indent=indent)
