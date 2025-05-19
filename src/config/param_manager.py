import os
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

class ParamType(Enum):
    CATEGORICAL = "categorical"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"

@dataclass
class ParamConfig:
    enabled: bool
    type: ParamType
    description: str
    default: Any
    choices: Optional[List[Any]] = None
    range: Optional[Dict[str, Any]] = None
    derived_from: Optional[str] = None
    derivation: Optional[str] = None

class TunableParamManager:
    def __init__(self, config_path: str):
        """Initialize the parameter manager with a configuration file."""
        self.config_path = config_path
        self.config = self._load_config()
        self.parameters = self._parse_parameters()

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _parse_parameters(self) -> Dict[str, ParamConfig]:
        """Parse the parameters section of the configuration."""
        params = {}
        for name, config in self.config['parameters'].items():
            param_type = ParamType(config['type'])
            
            # Extract range or choices based on type
            choices = None
            range_config = None
            if param_type == ParamType.CATEGORICAL:
                choices = config.get('choices')
            elif param_type in (ParamType.INT, ParamType.FLOAT):
                range_config = config.get('range')

            params[name] = ParamConfig(
                enabled=config['enabled'],
                type=param_type,
                description=config['description'],
                default=config['default'],
                choices=choices,
                range=range_config,
                derived_from=config.get('derived_from'),
                derivation=config.get('derivation')
            )
        return params

    def get_tunable_params(self) -> Dict[str, ParamConfig]:
        """Get all parameters that are enabled for tuning."""
        return {name: param for name, param in self.parameters.items() if param.enabled}

    def get_param_config(self, name: str) -> Optional[ParamConfig]:
        """Get configuration for a specific parameter."""
        return self.parameters.get(name)

    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {name: param.default for name, param in self.parameters.items()}

    def get_derived_value(self, param_name: str, base_value: Any) -> Any:
        """Calculate derived value for a parameter based on its derivation rule."""
        param = self.parameters[param_name]
        if not param.derived_from or not param.derivation:
            return None

        if param.derivation == "next_power_of_2":
            return 2 ** (base_value.bit_length())
        
        raise ValueError(f"Unknown derivation rule: {param.derivation}")

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get the optimization configuration."""
        return self.config['optimization']

    def get_model_config(self) -> Dict[str, Any]:
        """Get the model configuration."""
        return self.config['model']

    def get_test_config(self) -> Dict[str, Any]:
        """Get the test configuration."""
        return self.config['test']

    def format_storage_name(self) -> str:
        """Format the storage name with the study name."""
        storage = self.config['optimization']['storage']
        study_name = self.config['optimization']['study_name']
        return storage.format(study_name=study_name) 