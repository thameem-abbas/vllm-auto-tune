# vLLM Auto-Tune

An automated parameter tuning system for vLLM using Optuna for multi-objective optimization. This tool helps find optimal vLLM server configurations by automatically testing different parameter combinations and optimizing for throughput and latency.

## Features

- Multi-objective optimization of vLLM parameters
- Configurable parameter space via YAML
- Support for different parameter types (categorical, integer, float, boolean)
- Derived parameters (e.g., max_num_seqs from concurrency)
- Automatic load testing with configurable duration and concurrency
- Comprehensive logging and result tracking
- Support for both NSGAII and Grid sampling strategies

## Prerequisites

- Python 3.8+
- vLLM
- Optuna
- PyTorch
- llm-load-test (for load testing)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vllm-auto-tune.git
cd vllm-auto-tune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export LLM_LOAD_TEST_HOME=/path/to/llm-load-test
export BASE_CONFIG_PATH=/path/to/base/config.yaml
```

## Configuration

The system is configured through a YAML file (`src/config/tunable_params.yaml`). Here's an example configuration:

```yaml
parameters:
  block_size:
    enabled: true
    type: categorical
    choices: [8, 16, 32, 64, 128]
    description: "Block size for KV cache"
    default: 16

  concurrency:
    enabled: true
    type: int
    range:
      low: 32
      high: 256
      step: 4
    description: "Number of concurrent requests"
    default: 64

optimization:
  study_name: "vllm-tune-multi-objective"
  storage: "sqlite:////tmp/vllm-tune/{study_name}.db"
  max_trials: 30
  directions:
    - maximize  # throughput
    - minimize  # itl_penalty
  sampler: "NSGAII"
```

### Parameter Types

1. **Categorical Parameters**:
   ```yaml
   parameter_name:
     enabled: true
     type: categorical
     choices: [value1, value2, ...]
     default: value1
   ```

2. **Integer Parameters**:
   ```yaml
   parameter_name:
     enabled: true
     type: int
     range:
       low: 1
       high: 100
       step: 1
     default: 50
   ```

3. **Boolean Parameters**:
   ```yaml
   parameter_name:
     enabled: true
     type: bool
     default: false
   ```

4. **Derived Parameters**:
   ```yaml
   parameter_name:
     enabled: false
     type: int
     derived_from: "base_parameter"
     derivation: "next_power_of_2"
   ```

### Optimization Settings

- `study_name`: Name of the optimization study
- `storage`: SQLite database path for storing results
- `max_trials`: Maximum number of trials to run
- `directions`: Optimization objectives (maximize/minimize)
- `sampler`: Optimization algorithm (NSGAII/Grid)

### Model Settings

- `path`: Path to the model
- `attention_backend`: Attention backend to use
- `served_model_name`: Name of the served model

### Test Settings

- `duration`: Duration of each load test in seconds
- `max_trailing_request_allowance`: Allowance for trailing requests
- `itl_median_ceiling`: Maximum acceptable median ITL
- `health_check_url`: URL for health checks
- `launch_wait_time`: Maximum time to wait for server launch

## Usage

1. Configure the parameters in `src/config/tunable_params.yaml`

2. Run the optimization:
```bash
python -m src.serving.main
```

3. Monitor the optimization:
   - Logs are stored in `/tmp/vllm-tune/logs`
   - Results are stored in `/tmp/vllm-tune/artifacts`
   - Study database is stored in `/tmp/vllm-tune/{study_name}.db`

4. View results:
```bash
python -c "import optuna; study = optuna.load_study(study_name='your_study_name', storage='sqlite:////tmp/vllm-tune/your_study_name.db'); print(study.best_trials)"
```

## Project Structure

```
src/
├── config/
│   ├── tunable_params.yaml  # Parameter configuration
│   └── param_manager.py     # Parameter management
├── core/
│   ├── objective.py         # Scoring functions
│   └── optimization.py      # Optimization logic
├── serving/
│   ├── main.py             # Main entry point
│   ├── server.py           # vLLM server management
│   └── load_test.py        # Load testing
└── utils/
    ├── file_utils.py       # File operations
    └── health.py           # Health check utilities
```

## Adding New Parameters

To add a new tunable parameter:

1. Add the parameter definition to `tunable_params.yaml`:
```yaml
parameters:
  new_parameter:
    enabled: true
    type: int  # or categorical, float, bool
    range:
      low: 1
      high: 100
      step: 1
    description: "Description of the parameter"
    default: 50
```

2. The parameter will automatically be included in the optimization if `enabled: true`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Acknowledgments

- vLLM team for the excellent inference engine
- Optuna team for the optimization framework
- llm-load-test for the load testing capabilities 