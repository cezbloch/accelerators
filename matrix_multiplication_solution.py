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

    flags = cl.mem_flags
    matrix_a_gpu = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=matrix_a_cpu)
    matrix_b_gpu = cl.Buffer(context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=matrix_b_cpu)
    size_in_bytes = matrix_b_cpu.nbytes
    res_gpu = cl.Buffer(context, flags.WRITE_ONLY, size_in_bytes)
    
    matrix_dim = np.int32(N) # convert to numpy variable - required by PyOpenCL

    kernel_string = """
    __kernel void matrix_multiplication(__global float *a, __global float *b, __global float *res, int matrix_dim)
    {
      const uint2 gid = (uint2)(get_global_id(0), get_global_id(1));
      const uint index = gid.x + gid.y * matrix_dim;
      uint result = 0;
      for (uint i = 0; i < matrix_dim; i++)
      {
          result += a[gid.y * matrix_dim + i] * b[gid.x + i * matrix_dim];
      }
      res[index] = result;
    }
    """ 

    print("Compiling kernel..")

    program = cl.Program(context, kernel_string).build()

    local_work_size = (threads, threads)
    global_work_size = (N, N)

    print(f"Scheduling kernel execution with global_work_size = {global_work_size}, local_work_size = {local_work_size}")
    kernel_start = time()
    event = program.matrix_multiplication(queue, 
                                        global_work_size, 
                                        local_work_size,
                                        matrix_a_gpu, 
                                        matrix_b_gpu,
                                        res_gpu,
                                        matrix_dim)
    
    event.wait()
    kernel_finish = time()
    execution_time_ms = (event.profile.end - event.profile.start) * 1e-6
    cpu_kernel_time_ms = (kernel_finish - kernel_start) * 1e3
    print(f"matrix_multiplication kernel took: GPU time = {execution_time_ms:.4f} ms. CPU time = {cpu_kernel_time_ms:.4f} ms.")

    res_cpu = np.zeros(N * N).astype(np.float32)
    e = cl.enqueue_copy(queue, res_cpu, res_gpu)
    
    e.wait()
    e_ms = (e.profile.end - e.profile.start) * 1e-6
    print(f"copy memory from device to host took: {e_ms:.4f} ms.")
    
    print("Verifying results..")

    numpy_start = time()
    check = np.dot(matrix_a_cpu.reshape(N, N), matrix_b_cpu.reshape(N, N))
    numpy_end = time()
    numpy_time_ms = (numpy_end - numpy_start) * 1e3
    print(f"matrix_multiplication with numpy took: CPU time = {numpy_time_ms:.4f} ms.")
    
    res_cpu = res_cpu.reshape(N, N)
#     print(f"computed in cpu = {check}")
#     print(f"computed in GPU = {res_cpu}")
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