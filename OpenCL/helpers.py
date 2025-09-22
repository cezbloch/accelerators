"""Utility functions for GPU profiling and plotting."""

from time import time

import numpy as np
import matplotlib.pyplot as plt


def profile_gpu(function, n, queue, global_size, local_size, *args):
    """
    Profiles the execution time of a GPU kernel function over multiple runs.

    Args:
        function: The GPU kernel function to profile.
            Should return an event with profiling info.
        n (int): Number of times to run the function for profiling.
        queue: The OpenCL command queue.
        global_size: Global work size for the kernel.
        local_size: Local work size for the kernel.
        *args: Additional arguments to pass to the kernel function.

    Returns:
        float: The average execution time in milliseconds.
    """
    times = np.zeros(n)
    function(queue, global_size, local_size, *args).wait()
    function(queue, global_size, local_size, *args).wait()

    for i in range(n):
        e = function(queue, global_size, local_size, *args)
        e.wait()
        elapsed = (e.profile.end - e.profile.start) * 1e-6
        times[i] = elapsed

    avg_ms = np.mean(times)
    median_ms = np.median(times)
    variance = np.var(times)
    std = np.std(times)
    min_ms = np.min(times)

    print(
        f"{function.function_name} took minimum = {min_ms:.4f} ms, "
        f"on average {avg_ms:.4f} ms, "
        f"with median {median_ms:.4f} ms, "
        f"variance {variance:.4f} ms, "
        f"standard deviation {std:.4f} ms."
    )

    return median_ms


def plot(x, y, x_label, y_label):
    """
    Plots data points using matplotlib.

    Args:
        x: X-axis data points.
        y: Y-axis data points.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def overlay_plots(x, y_prime, y_prime_gpu, x_label='x', y_label='Derivative'):
    """
    Overlays two plots: y_prime (red, underneath) and y_prime_gpu (blue, above).

    Args:
        x: X-axis data points.
        y_prime: Ground truth derivative values (plotted in red).
        y_prime_gpu: Derivative values computed on GPU (plotted in blue).
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
    """
    plt.figure(figsize=(12, 7))
    plt.plot(x, y_prime, color='red', label='Ground Truth (-sin(x))', zorder=1)
    plt.plot(x, y_prime_gpu, color='blue', label='GPU Result', zorder=2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def profile_cpu(function, n, *args):
    """
    This function profiles other functions:
        function - any function to be profiled
        n - number of times a function will be rerun - the more times you run it the more stable results you get,
            but the more time it will take to profile
        args - variable list of arguments that will be passed to a profiled function
    """
    times = np.zeros(n)
    value = 0
    for i in range(n):
        start = time()
        value = function(*args)
        end = time()
        elapsed = (end - start) * 1e3
        times[i] = elapsed

    avg_ms = np.mean(times)
    median_ms = np.median(times)
    variance = np.var(times)
    std = np.std(times)

    print(f"{function.__name__} took on average {avg_ms:.4f} ms, with median {median_ms:.4f} ms, variance {variance:.4f} ms, standard deviation {std:.4f} ms.")        
    return value, median_ms
