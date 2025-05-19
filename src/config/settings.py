import os

# Environment Variables
LLM_LOAD_TEST_HOME = os.environ.get("LLM_LOAD_TEST_HOME")
if LLM_LOAD_TEST_HOME is None:
    raise RuntimeError("Set LLM_LOAD_TEST_HOME !!!")

BASE_CONFIG = os.environ.get("BASE_CONFIG_PATH")
if not os.path.exists(BASE_CONFIG):
    raise RuntimeError("Base config missing")

# Constants
MAX_SINGLE_TEST_DURATION = 500
MAX_TRAILING_REQUEST_ALLOWANCE = 0.2
ITL_MEDIAN_CEILING = 25
MAX_NUM_TRIALS = 30

# Study Configuration
STUDY_NAME = "vllm-tune-multi-objective-non-grid-sampler-v2"
STORAGE_NAME = f"sqlite:////tmp/vllm-tune/{STUDY_NAME}.db"

# Paths
LOG_FOLDER_PATH = "/tmp/vllm-tune/logs"
ARTIFACTS_DIR = "/tmp/vllm-tune/artifacts"

# vLLM Configuration
VLLM_ATTENTION_BACKEND = "FLASH_ATTN"
VLLM_HEALTH_CHECK_URL = "http://0.0.0.0:8000/health"
VLLM_LAUNCH_WAIT_TIME = 60

# Load Test Configuration
LOAD_TEST_DURATION = 60  # Duration in seconds for each load test 