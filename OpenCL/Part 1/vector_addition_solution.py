flags = cl.mem_flags

d_a = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(ctx, flags.WRITE_ONLY, h_a.nbytes)

---------------------------
%%cl_kernel -o "-cl-fast-relaxed-math"

__kernel void add_vectors(__global const int *a, __global const int *b, __global int *c)
{
    int gid = get_global_id(0);
    c[gid] = 2 * a[gid] + b[gid];
}
----------------------------
local_work_size = (64,)
global_work_size = (N,)

profile_gpu(add_vectors, 20, 
            queue, 
            global_work_size, 
            local_work_size,
            d_a,
            d_b, 
            d_c)