import requests
from src.config.settings import VLLM_HEALTH_CHECK_URL

def check_vllm_health():
    """Check if the vLLM server is healthy."""
    try:
        if (requests.get(VLLM_HEALTH_CHECK_URL).status_code == 200):
            return True
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        print("Error checking vLLM health:", type(e))
        return False
    return False 