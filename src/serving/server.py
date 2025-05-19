import subprocess
import time
import warnings
import signal
from typing import Dict, Any, Optional
from src.config.param_manager import TunableParamManager
from src.utils.health import check_vllm_health
from src.utils.file_utils import check_log_for_errors

class VLLMServer:
    def __init__(self, log_file_path: str, param_manager: TunableParamManager):
        self.log_file_path = log_file_path
        self.param_manager = param_manager
        self.process = None

    def _build_server_args(self, params: Dict[str, Any]) -> list:
        """Build the vLLM server command line arguments from parameters."""
        model_config = self.param_manager.get_model_config()
        args = [
            "vllm",
            "serve",
            model_config['path'],
        ]

        # Add all parameters that have values
        for name, value in params.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:  # Only add flag if True
                        args.append(f"--{name.replace('_', '-')}")
                else:
                    args.extend([f"--{name.replace('_', '-')}", str(value)])

        # Add model name
        args.extend(["--served-model-name", model_config['served_model_name']])

        return args

    def start(self, params: Dict[str, Any]) -> bool:
        """Start the vLLM server with given parameters."""
        with open(self.log_file_path, "w+") as f:
            args = self._build_server_args(params)
            env = dict(os.environ)
            env["VLLM_ATTENTION_BACKEND"] = self.param_manager.get_model_config()['attention_backend']

            self.process = subprocess.Popen(
                args=args,
                env=env,
                stdout=f,
                stderr=f,
                stdin=subprocess.PIPE,
                text=True
            )

        # Wait for server to start
        test_config = self.param_manager.get_test_config()
        start_time = time.time()
        while True:
            if check_vllm_health():
                return True
            if time.time() - start_time > test_config['launch_wait_time']:
                warnings.warn("vLLM did not start in time")
                self.terminate()
                return False
            if check_log_for_errors(self.log_file_path):
                warnings.warn(f"vLLM did not start successfully. Check logs for errors: {self.log_file_path}")
                self.terminate()
                return False
            time.sleep(1)

    def terminate(self):
        """Terminate the vLLM server process."""
        if self.process:
            self.process.terminate()
            self.process = None

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for the server process to complete."""
        if self.process:
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.send_signal(signal.SIGINT)
                return False
        return True 