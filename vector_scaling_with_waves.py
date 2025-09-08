import argparse

import numpy as np
import pyopencl as cl
import pyopencl.characterize.performance as perf


def compute(N, multiplier):
    print("Initialization - creating context and queue with profiling enabled..")

    platform = cl.get_platforms()[0]

    context = cl.Context(
        dev_type=cl.device_type.ALL, 
        properties=[(cl.context_properties.PLATFORM, platform)])    
    
    queue = cl.CommandQueue(context, 
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    buffer_cpu = np.arange(0, N).astype(np.float32)

    flags = cl.mem_flags
    buffer_gpu = cl.Buffer(context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=buffer_cpu)

    kernel_string = """
    __kernel void multiply_vector(__global float *buffer_gpu, int multiplier)
    {
      int gid = get_global_id(0);
      buffer_gpu[gid] = buffer_gpu[gid] * multiplier;
    }
    """

    print("Compiling kernel..")

    program = cl.Program(context, kernel_string).build()

    local_work_size = (1,)
    global_work_size = (N,)

    multiplier = np.int32(multiplier) # convert to numpy variable - required by PyOpenCL

    print("Scheduling kernel execution..")

    event = program.multiply_vector(queue, 
                                    global_work_size, 
                                    local_work_size,
                                    buffer_gpu, 
                                    multiplier)
    
    event.wait()
    execution_time_ms = (event.profile.end - event.profile.start) * 1e-6
    print(f"mutiply_vector kernel took: {execution_time_ms} ms.")

    res_cpu = np.zeros(N).astype(np.float32)
    e = cl.enqueue_copy(queue, res_cpu, buffer_gpu)
    
    e.wait()    
    e_ms = (e.profile.end - e.profile.start) * 1e-6
    print(f"copy memory from device to host took: {e_ms} ms.")
    
    print("Verifying results..")

    check = buffer_cpu * multiplier
    # print(f"computed in cpu = {check}")
    # print(f"computed in GPU = {res_cpu}")
    are_equal = np.allclose(res_cpu, check)

    print(f"{len(res_cpu)} elements multiplied correctly: {are_equal}")


if __name__ == '__main__':
    import sys
    cmdLineParser = argparse.ArgumentParser()
    cmdLineParser.add_argument("--n", type=int, required=True, help='number of elements in a vector')
    cmdLineParser.add_argument("--multiplier", type=int, required=True, help='vector multiplier value')

    if len(sys.argv[1:]) < 2:
        cmdLineParser.print_help()
        sys.exit(0)

    argsRead = cmdLineParser.parse_args(sys.argv[1:])
    compute(argsRead.n, argsRead.multiplier)