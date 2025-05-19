import signal
import time

import optuna
import os
import subprocess
from subprocess import STDOUT, check_output
import json
import yaml
import requests
import warnings
import datetime
import logging

# LLM-load-test is currently not a package and will have to be invoked via the command line
import sys
LLM_LOAD_TEST_HOME = os.environ("LLM_LOAD_TEST_HOME")
if LLM_LOAD_TEST_HOME is None:
    raise RuntimeError("Set LLM_LOAD_TEST_HOME !!!")
sys.path.append(LLM_LOAD_TEST_HOME)

BASE_CONFIG = os.environ("BASE_CONFIG_PATH")
if not os.path.exists(BASE_CONFIG):
    raise RuntimeError("Base config missing")

MAX_SINGLE_TEST_DURATION = 500
MAX_TRAILING_REQUEST_ALLOWANCE = 0.2
ITL_MEDIAN_CEILING = 25
MAX_NUM_TRIALS = 30

def read_llm_load_test_summary(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError("Result file not found")
    try:
        load_test_summary = json.load(open(filepath)).get("summary")
    except json.JSONDecodeError as e:
        raise RuntimeError("JSON Invalid : " + e.msg)
    if load_test_summary is None:
        raise RuntimeError("Could not find test summary in : " + filepath)
    
    return load_test_summary
    
def make_file_name(prefix : str, params : dict, suffix : str = ""):
    return prefix + "_".join([str(param[0]) + "-" + str(param[1]) for param in params.items()]) + suffix

def check_vllm_health():
    try:
        if (requests.get("http://0.0.0.0:8000/health").status_code == 200):
            return True
    except requests.exceptions.ConnectionError as ce:
        return False
    except Exception as e:
        print("Error : e : " , type(e))
        return False

def tail(f, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end"""
    # place holder for the lines found
    lines_found = []

    # block counter will be multiplied by buffer
    # to get the block size from the end
    block_counter = -1

    # loop until we find X lines
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            f.seek(0)
            lines_found = f.readlines()
            break

        lines_found = f.readlines()

        # we found enough lines, get out
        # Removed this line because it was redundant the while will catch
        # it, I left it for history
        # if len(lines_found) > lines:
        #    break

        # decrement the block counter to get the
        # next X bytes
        block_counter -= 1

    return lines_found[-lines:]

# Potentially long running operation if the log file is large. 
# Limiting the search space to only the last 1000 lines
def check_log_for_errors(log_file_path):
    with open(log_file_path, "r") as f:
        log_lines = tail(f, lines=1000)
    
    for line in log_lines:
        if "ERROR" in line:
            return True
        
    return False

# TODO: Investigate the ideal properties of the scoring function
# Can we have multipart scoring functions ?
# Does the function need to be continuous ?
# Does the function need to be differentiable ?
def score_metric(throughput, itl_median):
    if itl_median > ITL_MEDIAN_CEILING:
        # TODO: Investigate if returning zero is better or if returning a negative value is better
        # TODO: Investigate scoring as a function of 
        return throughput * -0.5 
    return throughput

def score_metric_v2(throughput, itl_median):
    # FLAW: This can incentivize lower ITL over higher throughput disproportionately
    return (throughput/1000) * (1 - itl_median / ITL_MEDIAN_CEILING)

# Need to figure out a way to design better objective scoring functions
def score_metric_v3(throughput, itl_median):
    # abs to try and keep it as close to the ITL_MEDIAN_CEILING as possible while still keeping it just below
    # The 0.95 is to try and ensure that the ITL is just below the ceiling
    return (throughput/1000) * abs(0.95 - itl_median / ITL_MEDIAN_CEILING)
    # This also negatively influences when the ITL is too low even if the throughput is higher which is what we really want. 

def score_metric_v4(throughput, itl_median):
    penalty = max(0, (itl_median - ITL_MEDIAN_CEILING) / ITL_MEDIAN_CEILING)
    return (throughput / 1000) * (1 - penalty)

# Just letting it run with the values directly
# This is a multi-objective scoring function
def multi_objective_score_v1(throughput, itl_median):
    return throughput, itl_median

def multi_objective_score_v2(throughput, itl_median):
    # Trying to bring them to the same scale
    # ITL penalty is a quadratic function to penalize more as it moves further away from the ITL_MEDIAN_CEILING
    itl_penalty = max(0, ((itl_median - ITL_MEDIAN_CEILING) / ITL_MEDIAN_CEILING)**2)
    return throughput / 1000, itl_penalty

# The next power of 2 greater than the current concurrency
def get_next_max_concurrency_limit(concurrency):
    return 2 ** (concurrency.bit_length())

from src.config.settings import (
    STUDY_NAME, STORAGE_NAME, MAX_NUM_TRIALS,
    LOG_FOLDER_PATH, ARTIFACTS_DIR
)
from src.core.objective import multi_objective_score_v2, get_next_max_concurrency_limit
from src.core.optimization import (
    create_study, setup_logging, save_study_results, print_best_trials, suggest_trial_params
)
from src.serving.server import VLLMServer
from src.serving.load_test import LoadTester
from src.utils.file_utils import make_file_name
from src.config.param_manager import TunableParamManager

def objective(trial: optuna.Trial, param_manager: TunableParamManager):
    """Optuna objective function for optimizing vLLM parameters."""
    # Get study attributes
    study_start_time = trial.study.user_attrs["study_start_time"]
    log_folder_path = trial.study.user_attrs["log_folder_path"]
    artifacts_dir = trial.study.user_attrs["artifacts_dir"]

    # Get parameters for this trial
    params = suggest_trial_params(trial, param_manager)

    # Setup paths
    trial_log_dir = os.path.join(log_folder_path, trial.study.study_name, study_start_time)
    os.makedirs(trial_log_dir, exist_ok=True)
    log_file_path = os.path.join(trial_log_dir, make_file_name("trial_", params, ".log"))

    # Initialize server and load tester
    server = VLLMServer(log_file_path, param_manager)
    load_tester = LoadTester(artifacts_dir, trial.study.study_name, study_start_time)

    try:
        # Start vLLM server
        if not server.start(params):
            return 1

        # Prepare and run load test
        _, config_file_path = load_tester.prepare_config(params, params['concurrency'])
        load_test_results = load_tester.run_load_test(config_file_path, log_file_path)

        if load_test_results is None:
            return 1

        # Get metrics
        throughput = load_test_results["throughput"]
        itl_median = load_test_results["itl"]["median"]

        # Set trial attributes
        trial.set_user_attr("throughput", throughput)
        trial.set_user_attr("itl_median", itl_median)
        trial.set_user_attr("config_file_path", config_file_path)

        return multi_objective_score_v2(throughput, itl_median)

    finally:
        # Cleanup
        load_tester.terminate()
        server.terminate()

def main():
    """Main entry point for the optimization process."""
    # Initialize parameter manager
    param_manager = TunableParamManager("src/config/tunable_params.yaml")
    
    # Setup logging
    setup_logging()

    # Create study
    study = create_study(param_manager)

    # Get optimization config
    opt_config = param_manager.get_optimization_config()

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, param_manager),
        show_progress_bar=True,
        n_trials=opt_config['max_trials']
    )

    # Save and print results
    save_study_results(study, f"/tmp/vllm-tune/{opt_config['study_name']}.csv")
    print_best_trials(study)

if __name__ == "__main__":
    main()