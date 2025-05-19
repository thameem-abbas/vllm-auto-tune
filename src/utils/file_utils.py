import os
import json
import yaml

def read_llm_load_test_summary(filepath):
    """Read and parse the LLM load test summary from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError("Result file not found")
    try:
        load_test_summary = json.load(open(filepath)).get("summary")
    except json.JSONDecodeError as e:
        raise RuntimeError("JSON Invalid : " + e.msg)
    if load_test_summary is None:
        raise RuntimeError("Could not find test summary in : " + filepath)
    
    return load_test_summary

def make_file_name(prefix: str, params: dict, suffix: str = ""):
    """Create a filename from parameters."""
    return prefix + "_".join([str(param[0]) + "-" + str(param[1]) for param in params.items()]) + suffix

def tail(f, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end."""
    lines_found = []
    block_counter = -1

    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            f.seek(0)
            lines_found = f.readlines()
            break

        lines_found = f.readlines()
        block_counter -= 1

    return lines_found[-lines:]

def check_log_for_errors(log_file_path):
    """Check the last 1000 lines of a log file for errors."""
    with open(log_file_path, "r") as f:
        log_lines = tail(f, lines=1000)
    
    for line in log_lines:
        if "ERROR" in line:
            return True
        
    return False

def save_yaml_config(config, filepath):
    """Save a configuration dictionary to a YAML file."""
    with open(filepath, "w+") as config_file:
        yaml.dump(config, config_file)

def load_yaml_config(filepath):
    """Load a configuration from a YAML file."""
    with open(filepath, "r") as config_file:
        return yaml.load(config_file, Loader=yaml.FullLoader) 