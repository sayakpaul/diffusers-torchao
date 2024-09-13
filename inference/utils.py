import gc

import torch
import os
import pwd
import shutil
import torch.utils.benchmark as benchmark
from tabulate import tabulate


def bytes_to_giga_bytes(bytes):
    return f"{(bytes / 1024 / 1024 / 1024):.3f}"


def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"


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


def get_current_user():
    """Get the current user's username."""
    return pwd.getpwuid(os.getuid()).pw_name


def is_file_owned_by_current_user(filepath):
    """Check if a file is owned by the current user."""
    file_stat = os.stat(filepath)
    return file_stat.st_uid == os.getuid()


def cleanup_tmp_directory():
    """Remove files in /tmp owned by the current user."""
    tmp_dir = "/tmp"
    current_user = get_current_user()

    for filename in os.listdir(tmp_dir):
        filepath = os.path.join(tmp_dir, filename)

        try:
            if is_file_owned_by_current_user(filepath):
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    print(f"Removed file: {filepath}")
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                    print(f"Removed directory: {filepath}")
            else:
                print(f"Skipping {filepath}, not owned by {current_user}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
