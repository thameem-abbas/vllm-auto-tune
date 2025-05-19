import optuna
import torch
import datetime
import logging
import sys
from typing import Dict, Any, Optional
from src.config.param_manager import TunableParamManager

def create_study(param_manager: TunableParamManager) -> optuna.Study:
    """Create and configure an Optuna study based on parameter manager configuration."""
    opt_config = param_manager.get_optimization_config()
    
    # Create study with appropriate sampler
    if opt_config['sampler'] == 'Grid':
        if not opt_config['grid_sampler_specs']:
            raise ValueError("Grid sampler specs path required when using grid sampler")
        study = optuna.create_study(
            directions=opt_config['directions'],
            study_name=opt_config['study_name'],
            storage=param_manager.format_storage_name(),
            load_if_exists=True,
            sampler=optuna.samplers.GridSampler(opt_config['grid_sampler_specs'])
        )
    else:
        study = optuna.create_study(
            directions=opt_config['directions'],
            study_name=opt_config['study_name'],
            storage=param_manager.format_storage_name(),
            load_if_exists=True,
            sampler=optuna.samplers.NSGAIISampler()
        )

    # Set study attributes
    study.set_user_attr("study_start_time", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    study.set_user_attr("log_folder_path", "/tmp/vllm-tune/logs")
    study.set_user_attr("artifacts_dir", "/tmp/vllm-tune/artifacts")

    # Set GPU properties
    gpu_properties = torch.cuda.get_device_properties(0)
    study.set_user_attr("gpu_name", gpu_properties.name)
    study.set_user_attr("gpu_memory", gpu_properties.total_memory)

    return study

def suggest_trial_params(trial: optuna.Trial, param_manager: TunableParamManager) -> Dict[str, Any]:
    """Suggest parameters for a trial based on the parameter manager configuration."""
    params = {}
    tunable_params = param_manager.get_tunable_params()

    for name, param_config in tunable_params.items():
        if param_config.type == 'categorical':
            params[name] = trial.suggest_categorical(name, param_config.choices)
        elif param_config.type == 'int':
            range_config = param_config.range
            params[name] = trial.suggest_int(
                name,
                low=range_config['low'],
                high=range_config['high'],
                step=range_config.get('step', 1)
            )
        elif param_config.type == 'float':
            range_config = param_config.range
            params[name] = trial.suggest_float(
                name,
                low=range_config['low'],
                high=range_config['high'],
                step=range_config.get('step')
            )
        elif param_config.type == 'bool':
            params[name] = trial.suggest_categorical(name, [True, False])

    # Add derived parameters
    for name, param_config in param_manager.parameters.items():
        if param_config.derived_from and param_config.derived_from in params:
            params[name] = param_manager.get_derived_value(
                name, params[param_config.derived_from]
            )

    # Add non-tunable parameters with default values
    for name, param_config in param_manager.parameters.items():
        if not param_config.enabled and name not in params:
            params[name] = param_config.default

    return params

def setup_logging():
    """Configure logging for the optimization process."""
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

def save_study_results(study: optuna.Study, output_path: str):
    """Save study results to CSV."""
    study.trials_dataframe().to_csv(output_path)

def print_best_trials(study: optuna.Study):
    """Print information about the best trials."""
    trials = sorted(study.best_trials, key=lambda trial: trial.values)
    print("Best Trials : ")
    for trial in trials:
        print("Trial : ", trial.number)
        print("Params : ", trial.params)
        print("Value : ", trial.values)
        print("Attributes : ", trial.user_attrs)
        print("Intermediate Values : ", trial.intermediate_values)
        print("Datetime : ", trial.datetime_start) 