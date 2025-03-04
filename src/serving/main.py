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
LLM_LOAD_TEST_HOME = "/root/thibrahi/vllm-auto-tune/llm-load-test"
sys.path.append(LLM_LOAD_TEST_HOME)

MAX_SINGLE_TEST_DURATION = 500
MAX_TRAILING_REQUEST_ALLOWANCE = 0.2
ITL_MEDIAN_CEILING = 25

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


def objective(trial : optuna.Trial):

    # User Attributes
    study_start_time = trial.study.user_attrs["study_start_time"]
    log_folder_path = trial.study.user_attrs["log_folder_path"] if "log_folder_path" in trial.study.user_attrs else "/tmp/vllm-tune/logs"
    artifacts_dir = trial.study.user_attrs["artifacts_dir"] if "artifacts_dir" in trial.study.user_attrs else "/tmp/vllm-tune/artifacts"

    # Parameter Suggestions

    # enable_chunked_prefill = trial.suggest_categorical(
        # "chunked_prefill",
        # choices=[True, False]
    # )

    num_scheduler_steps = trial.suggest_int("num-multi-steps", low=1, high=10, step=1)

    block_size = trial.suggest_categorical(
        "block_size",
        # choices=[8,16,32] # Opt250m doesn't support 64 and 128
        choices=[8,16,32,64,128]
        # choices=[8]
        )

    # Concurrency
    concurrency = trial.suggest_int("concurrency", low=32, high=256, step=4) # Step size of 4 to reduce the number of possible trials
    max_seq_num = get_next_max_concurrency_limit(concurrency)

    log_folder_path = os.path.join(log_folder_path, trial.study.study_name, study_start_time) 
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    log_file_path = os.path.join(log_folder_path, make_file_name("trial_", trial.params, ".log"))

    # Launch vLLM as a subprocess
    with open(log_file_path, "w+") as f:

        # Kinda invasive and can break anytime that vLLM chooses to update their 
        # entrypoints
        # args = {}
        # if not vllm.entrypoints.openai.cli_args.validate_parsed_serve_args(args):
        #     raise RuntimeError("Invalid args")

        # vllm.entrypoints.openai.api_server.run_server(
        #     args=args
        # )
        
        vllm_proc = subprocess.Popen(
                args=[
                    "vllm",
                    "serve",
                    # "/mnt/data/models/Llama-3.2-3B",
                    # "/mnt/data/models/opt250m",
                    "/home/ubuntu/thibrahi/vllm-auto-tune/llama-32-3b-instruct",

                    "--max-model-len",
                    "2048",
                    
                    "--block-size",
                    str(block_size),
                    
                    # "--enable_chunked_prefill",
                    # str(enable_chunked_prefill).lower(),
                    # "false",

                    # "--enable-prefix-caching",
                    # "--no-enable-prefix-caching",

                    "--max-num-seqs",
                    str(max_seq_num),

         
                    "--num-scheduler-steps",
                    str(num_scheduler_steps),

                    "--served-model-name",
                    "trial",
                    # "--dtype", # Not needed on newer GPUs
                    # "half"
                    ],
                # executable="vllm",
                env= dict(os.environ).update({
                    "VLLM_ATTENTION_BACKEND":"FLASH_ATTN" # Possibly a way to also compare across Attention backends
                }),
                stdout=f,
                stderr=f,
                stdin=subprocess.PIPE,
                text=True
            )
        print("Done Starting process")

        # Wait until the health check becomes available
        LAUNCH_WAIT_TIME = 60
        start_time = time.time()
        while True:
            if check_vllm_health():
                break
            if time.time() - start_time > LAUNCH_WAIT_TIME:
                warnings.warn("vLLM did not start in time")
                vllm_proc.terminate()
                return 1
            if check_log_for_errors(log_file_path):
                warnings.warn("vLLM did not start successfully. Check logs for errors : " + os.path.join(log_folder_path, make_file_name("trial_", trial.params, ".log")))
                vllm_proc.terminate()
                return 1
            time.sleep(1)

        print("Health Check Passed")

        # llm-load-test
        # Using a base config file
        load_test_config = yaml.load(open("base_config.yaml"), Loader=yaml.FullLoader)
        # Parameters to be overwritten
        # Output dir
        # Load Options - Duration
        # Load Options - Concurrencies

        # Might make it flat with the output dir being the same and the output file name being different per trial
        load_test_config["output"]["dir"] = os.path.join(artifacts_dir, trial.study.study_name, study_start_time, make_file_name("output_", trial.params))
        if not os.path.exists(load_test_config["output"]["dir"]):
            os.makedirs(load_test_config["output"]["dir"])
        load_test_config["load_options"]["duration"] = 60 # TODO: Need a way to identify steady state
        load_test_config["load_options"]["concurrency"] = concurrency

        # Write the config file to tmp with a random ID
        config_file_path = os.path.join(artifacts_dir, trial.study.study_name, study_start_time, make_file_name("config_", trial.params, ".yaml"))
        with open(config_file_path, "w+") as config_file:
            yaml.dump(load_test_config, config_file)

        try:

            with open(log_file_path + "_load-test", "a") as log_file:
                log_file.write("Starting Load Test with Config : " + config_file_path + "\n")
                # Run the load test
                load_test_proc = subprocess.Popen(

                    args=" ".join([
                        # Ensure that the python executable is correct. If using a llm-load-test ghcr container, 
                        # the default python executable should have all the dependencies installed
                        "ulimit -n 8192 &&",
                        os.path.join(LLM_LOAD_TEST_HOME, "venv", "bin", "python"),
                        "load_test.py",
                        "-c",
                        config_file_path
                    ]),
                    cwd=LLM_LOAD_TEST_HOME,
                    stdout=log_file,
                    stderr=log_file,
                    shell=True # TODO: Move at least llm-load-test to containers to avoid the too many files open error for failed tests
                )
                # Wait for the load test to finish - 1.2x the duration to allow for the trailing requests to finish
                # Removing the timeout for now since llm-load-test cannot exit in time when there are too many requests trailing
                # This can be partly resolved by preemption detection but that is not implemented yet
                # TODO: Consider the condition of what happens when the performance is just too low and there are too many trailing requests without any preemption on the server side
                load_test_proc.wait(timeout=load_test_config["load_options"]["duration"] * (1 + MAX_TRAILING_REQUEST_ALLOWANCE) + 20)
                print("Load Test Finished")
                # Kill the load test process if not finished
                if load_test_proc.poll() is None:
                    load_test_proc.kill()
                    print("Killed Load Test")

            # Read the output file
            load_test_summary = read_llm_load_test_summary(os.path.join(load_test_config["output"]["dir"], f"output-{str(load_test_config["load_options"]["concurrency"]).zfill(3)}.json"))

            # Get throughput and latency
            throughput = load_test_summary["throughput"]
            itl_median = load_test_summary["itl"]["median"]

            # Score the metric
            # score = score_metric(throughput, itl_median)
            # score = score_metric_v4(throughput, itl_median)

            # Waiting for vLLM to timeout or finish
            try:
                print("Trying to comm 1")
                # Max Trailing Request Allowance
                out, errs = vllm_proc.communicate(timeout=load_test_config["load_options"]["duration"] * 0.1)
                # proc.wait(MAX_SINGLE_TEST_DURATION)
                print("commed 1")
                print("Kill sent")
            except subprocess.TimeoutExpired:
                # out, errs = proc.communicate()
                vllm_proc.send_signal(sig=signal.SIGINT)
        finally:
            load_test_proc.send_signal(sig=signal.SIGKILL)
            vllm_proc.send_signal(sig=signal.SIGINT)
            
            print("Communicated")
        print("Try Except Out")

        trial.set_user_attr("throughput", throughput)
        trial.set_user_attr("itl_median", itl_median)
        trial.set_user_attr("config_file_path", config_file_path)

        # f.write(out)

    return multi_objective_score_v2(throughput, itl_median)

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "vllm-tune-multi-objective-v2" # Will need to be made more discrete. Possibly identify code changes to ensure we're comparing apples to apples
storage_name = f"sqlite:////tmp/vllm-tune/{study_name}.db" # Save more information to the RDB to be accessed later, resume if needed

# Grid Sampler Specs
GRID_SAMPLER = False

if GRID_SAMPLER:
    # Load grid sampler_specs from yaml file
    grid_sampler_specs = yaml.load(open("grid_sampler_specs.yaml"), Loader=yaml.FullLoader)
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.GridSampler(grid_sampler_specs)
    )
else:
    study = optuna.create_study(
        directions=['maximize', 'minimize'],
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler()
    )

study.set_user_attr("study_start_time",datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
study.set_user_attr("log_folder_path", "/tmp/vllm-tune/logs")
study.set_user_attr("artifacts_dir", "/tmp/vllm-tune/artifacts")

# Get GPU Properties
# TODO: Extend to Multi-GPU Scenario
import torch
gpu_properties = torch.cuda.get_device_properties(0)
study.set_user_attr("gpu_name", gpu_properties.name)
study.set_user_attr("gpu_memory", gpu_properties.total_memory)
        

# Cannot run more than one job at this time
# TODO: GPU Indexing for parallelizing the trials
# TODO: CLI Output parsing to remove dependency on setting up yaml files
# TODO: Multi objective Optimization
# TODO: Constrained Optimization
# TODO: Logging and Monitoring
# TODO: Central Storage for experiments
# TODO: Central Storage for logs and artifacts
# TODO: Custom callback to monitor and stop the study if the optimizations are not making any further improvements
# TODO: Figure out a way to organize the whole thing. Not a code task
# TODO: A way to rapidly reach steady state ? - How to do this ?
# TODO: Add explanation of how to add good starting points for the optimization - Suggested Trials
# TODO: We don't Add KV Cache usage for the optimization. Rather we prune the trials where preemption happens. 
    # This can potentially be handled inside of this automation alone. 
    # This should avoid any bias towards higher KV Cache space util and will focus purely on throughput

# TODO: Investigate if vLLM telemetry can be used to optimize the model - How to do this ? (Not talking about the prometheus metrics)
    # Check if Preemption events are transmitted - Yes. It's available in the prometheus metrics

# TODO: Identify if any of the tunables have affinities to certain values. Eg: Is there a performance benefit to 

# Why not use the existing Kruize org repos for this ? - They only seem to run the experiments and want manual intervention for the optimization (Choosing if the experiment was good or not)
# This means it's easier to do subjective optimization but harder to do automated optimization

study.optimize(objective, show_progress_bar=True, n_trials=50)

# study.trials_dataframe().to_csv("/tmp/vllm-tune/study.csv")
trials = sorted(study.best_trials, key = lambda trial: trial.values)
print("Best Trials : ")
for trial in trials:
    print("Trial : ", trial.number)
    print("Params : ", trial.params)
    print("Value : ", trial.values)
    print("Attributes : ", trial.user_attrs)
    print("Intermediate Values : ", trial.intermediate_values)
    print("Datetime : ", trial.datetime_start)
study.trials_dataframe().to_csv("/tmp/vllm-tune/multi-objective-study.csv")