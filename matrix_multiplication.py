import argparse

import numpy as np
import pyopencl as cl
import pyopencl.characterize.performance as perf
from time import time


def compute(N, threads):
    print("Initialization - creating context and queue with profiling enabled..")

    platform = cl.get_platforms()[0]

    context = cl.Context(
        dev_type=cl.device_type.ALL, 
        properties=[(cl.context_properties.PLATFORM, platform)])    
    
    queue = cl.CommandQueue(context, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)
 
    matrix_a_cpu = np.full(N*N, fill_value = 1).astype(np.float32)
    matrix_b_cpu = np.full(N*N, fill_value = 2).astype(np.float32)
    
    #matrix_a_cpu = np.arange(0, N, 1/N).astype(np.float32)
    #matrix_b_cpu = np.arange(0, N, 1/N).astype(np.float32)

    # Your code goes here
    
    
    
    res_cpu = np.zeros(N*N).reshape(N, N)
    # Verify results

    numpy_start = time()
    check = np.dot(matrix_a_cpu.reshape(N, N), matrix_b_cpu.reshape(N, N))
    numpy_end = time()
    numpy_time_ms = (numpy_end - numpy_start) * 1e3
    print(f"matrix_multiplication with numpy took: CPU time = {numpy_time_ms:.4f} ms.")    
    
    print(f"computed in cpu = {check}")
    print(f"computed in GPU = {res_cpu}")
    are_equal = np.allclose(res_cpu, check)

    print(f"Matrix {res_cpu.shape} elements multiplied correctly: {are_equal}")


if __name__ == '__main__':
    import sys
    cmdLineParser = argparse.ArgumentParser()
    cmdLineParser.add_argument("--n", type=int, required=True, help='number of elements in a vector')
    cmdLineParser.add_argument("--threads", type=int, required=True, help='vector multiplier value')

    if len(sys.argv[1:]) < 2:
        cmdLineParser.print_help()
        sys.exit(0)

    argsRead = cmdLineParser.parse_args(sys.argv[1:])
    compute(argsRead.n, argsRead.threads)