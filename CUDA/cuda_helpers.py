import numpy as np
import pycuda.driver as cuda

def profile_gpu(func, n_warmup, n_iters, *kernel_launch_args, **kernel_launch_kwargs):
    """
    Profile the execution time of a CUDA GPU kernel (or any callable that launches CUDA work)
    over multiple iterations using CUDA Events for high‑resolution timing.
    The function first performs a number of warm-up launches (not timed) to allow for
    JIT compilation (e.g. Numba) and GPU frequency stabilization. It then records
    per-iteration execution times (in milliseconds) for the specified number of timed runs.
    Parameters
    ----------
    func : Callable
        A callable that launches GPU work. It must accept the arguments passed via
        kernel_launch_args. The callable should return None (its return value is ignored).
    n_warmup : int
        Number of warm-up launches to perform before timing (not included in results).
        Use at least 1–3 for JIT-compiled kernels or when measuring very short kernels.
    n_iters : int
        Number of timed iterations. Higher values improve statistical stability but
        increase total profiling time.
    *kernel_launch_args :
        Positional arguments forwarded directly to `func`. These typically include
        kernel configuration (if wrapped) and kernel parameters.
    Returns
    -------
    numpy.ndarray
        A 1D array of shape (n_iters,) and dtype float64 containing the elapsed
        time in milliseconds for each timed iteration.
    Side Effects
    ------------
    Prints a summary line reporting mean, median, and standard deviation of the
    measured times. No logging is performed beyond this print.
    Timing Method
    -------------
    Uses `cuda.Event()` (start/end) with `record()` and `synchronize()` to measure
    elapsed GPU time. This avoids host-side synchronization costs and provides
    sub-millisecond resolution. Each iteration records:
        elapsed_ms = start.time_till(end)
    Important Notes
    ---------------
    - The callable `func` should launch work asynchronously and must not itself
      perform a blocking device synchronize (other than what is required for a
      correct kernel launch), or the measured times may include extra overhead.
    - If `func` enqueues multiple kernels or includes device-to-host copies, the
      reported time covers all operations submitted between the start and end events.
    - Ensure that any required CUDA context is initialized before calling this
      profiler (importing and first use may trigger one-time overhead).
    - For extremely short kernels (< ~5 µs), event timing noise and clock granularity
      may dominate; consider batching work inside the kernel or increasing iterations.
    Error Handling
    --------------
    This function assumes successful kernel launches. If a launch fails
    (asynchronously), the error may surface on `end.synchronize()`. Wrap the call
    site in appropriate try/except blocks if needed.
    Examples
    --------
    # Basic usage with a kernel-launch wrapper (Numba example):
    # profile_gpu(my_kernel[grid_dim, block_dim], 3, 50, arg1, arg2)
    # With a custom callable:
    # def launch():
    #     kernel[grid, block](d_out, d_in)
    # times = profile_gpu(launch, n_warmup=5, n_iters=100)
    Statistical Guidance
    --------------------
    Use median for robust central tendency if outliers are present (e.g., due to
    occasional context scheduling). The standard deviation helps assess run-to-run
    variability. Consider discarding the first few timed iterations if you still
    observe warm-up effects.
    Performance Tips
    ----------------
    - Pin host memory for transfers performed inside `func` to reduce variability.
    - Avoid printing or Python-side allocations inside `func` during profiling.
    - Run on an otherwise idle GPU to minimize interference.
    """
    
    
    # Warm-up launches (not timed)
    for _ in range(n_warmup):
        func(*kernel_launch_args, **kernel_launch_kwargs)
    times = np.zeros(n_iters, dtype=np.float64)
    for i in range(n_iters):
        start = cuda.Event(); end = cuda.Event()
        start.record()
        func(*kernel_launch_args, **kernel_launch_kwargs)
        end.record()
        end.synchronize()
        times[i] = start.time_till(end)  # ms
        
    avg_ms = np.mean(times)
    median_ms = np.median(times)
    variance = np.var(times)
    std = np.std(times)
    min_ms = np.min(times)

    print(
        f"Function took minimum = {min_ms:.4f} ms, "
        f"on average {avg_ms:.4f} ms, "
        f"with median {median_ms:.4f} ms, "
        f"variance {variance:.4f} ms, "
        f"standard deviation {std:.4f} ms."
    )

    return times        
