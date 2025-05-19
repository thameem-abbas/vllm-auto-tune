import subprocess
import os
from src.config.settings import (
    LLM_LOAD_TEST_HOME, BASE_CONFIG, LOAD_TEST_DURATION,
    MAX_TRAILING_REQUEST_ALLOWANCE
)
from src.utils.file_utils import (
    load_yaml_config, save_yaml_config, read_llm_load_test_summary,
    make_file_name
)

class LoadTester:
    def __init__(self, artifacts_dir, study_name, study_start_time):
        self.artifacts_dir = artifacts_dir
        self.study_name = study_name
        self.study_start_time = study_start_time
        self.process = None

    def prepare_config(self, trial_params, concurrency):
        """Prepare the load test configuration."""
        # Load base config
        load_test_config = load_yaml_config(BASE_CONFIG)

        # Update config with trial-specific parameters
        output_dir = os.path.join(
            self.artifacts_dir,
            self.study_name,
            self.study_start_time,
            make_file_name("output_", trial_params)
        )
        os.makedirs(output_dir, exist_ok=True)

        load_test_config["output"]["dir"] = output_dir
        load_test_config["load_options"]["duration"] = LOAD_TEST_DURATION
        load_test_config["load_options"]["concurrency"] = concurrency

        # Save config
        config_file_path = os.path.join(
            self.artifacts_dir,
            self.study_name,
            self.study_start_time,
            make_file_name("config_", trial_params, ".yaml")
        )
        save_yaml_config(load_test_config, config_file_path)

        return load_test_config, config_file_path

    def run_load_test(self, config_file_path, log_file_path):
        """Run the load test with the given configuration."""
        with open(log_file_path + "_load-test", "a") as log_file:
            log_file.write(f"Starting Load Test with Config: {config_file_path}\n")
            
            self.process = subprocess.Popen(
                args=" ".join([
                    "ulimit -n 8192 &&",
                    os.path.join(LLM_LOAD_TEST_HOME, "venv", "bin", "python"),
                    "load_test.py",
                    "-c",
                    config_file_path
                ]),
                cwd=LLM_LOAD_TEST_HOME,
                stdout=log_file,
                stderr=log_file,
                shell=True
            )

            # Wait for completion with allowance for trailing requests
            timeout = LOAD_TEST_DURATION * (1 + MAX_TRAILING_REQUEST_ALLOWANCE) + 20
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.terminate()
                return None

        return self._get_test_results(config_file_path)

    def terminate(self):
        """Terminate the load test process."""
        if self.process:
            self.process.kill()
            self.process = None

    def _get_test_results(self, config_file_path):
        """Get the results from the load test."""
        config = load_yaml_config(config_file_path)
        output_dir = config["output"]["dir"]
        concurrency = config["load_options"]["concurrency"]
        
        result_file = os.path.join(
            output_dir,
            f"output-{str(concurrency).zfill(3)}.json"
        )
        
        return read_llm_load_test_summary(result_file) 