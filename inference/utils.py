import gc

import torch
from tabulate import tabulate


def reset_memory(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device):
    memory = torch.cuda.memory_allocated(device) / 1024**3
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory=:.3f}")
    print(f"{max_memory=:.3f}")
    print(f"{max_reserved=:.3f}")


def pretty_print_results(results, precision: int = 6):
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.{precision}f}"
        return value

    filtered_table = {k: format_value(v) for k, v in results.items()}
    print(tabulate([filtered_table], headers="keys", tablefmt="pipe", stralign="center"))
