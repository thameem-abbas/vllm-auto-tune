from src.config.settings import ITL_MEDIAN_CEILING

def score_metric(throughput, itl_median):
    """Basic scoring metric that penalizes high ITL."""
    if itl_median > ITL_MEDIAN_CEILING:
        return throughput * -0.5 
    return throughput

def score_metric_v2(throughput, itl_median):
    """Scoring metric that balances throughput and ITL."""
    return (throughput/1000) * (1 - itl_median / ITL_MEDIAN_CEILING)

def score_metric_v3(throughput, itl_median):
    """Scoring metric that tries to keep ITL just below ceiling."""
    return (throughput/1000) * abs(0.95 - itl_median / ITL_MEDIAN_CEILING)

def score_metric_v4(throughput, itl_median):
    """Scoring metric with linear penalty for exceeding ITL ceiling."""
    penalty = max(0, (itl_median - ITL_MEDIAN_CEILING) / ITL_MEDIAN_CEILING)
    return (throughput / 1000) * (1 - penalty)

def multi_objective_score_v1(throughput, itl_median):
    """Multi-objective scoring that returns raw values."""
    return throughput, itl_median

def multi_objective_score_v2(throughput, itl_median):
    """Multi-objective scoring with quadratic ITL penalty."""
    itl_penalty = max(0, ((itl_median - ITL_MEDIAN_CEILING) / ITL_MEDIAN_CEILING)**2)
    return throughput / 1000, itl_penalty

def get_next_max_concurrency_limit(concurrency):
    """Get the next power of 2 greater than the current concurrency."""
    return 2 ** (concurrency.bit_length()) 