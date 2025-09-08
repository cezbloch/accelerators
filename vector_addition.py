import pyopencl as cl
import numpy as np

print("Hello")

N = 32 * 1024 * 1024

a_cpu = np.full(N, fill_value = 1).astype(np.float32)
b_cpu = np.full(N, fill_value = 2).astype(np.float32)
