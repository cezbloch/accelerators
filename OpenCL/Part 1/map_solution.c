__kernel void compute_linear_equations_gpu(const __global int *a, 
                                           const __global int *b, 
                                           __global int *res, 
                                           int number_of_elements)
{
    const uint gid = get_global_id(0);
    const int grid_stride = get_global_size(0);

    for (int i = gid; i < number_of_elements; i += grid_stride) 
    {
        res[i] = 2 * a[i] + b[i];
    }
}

---------- host ------------

d_a = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(ctx, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=h_b)
d_c = cl.Buffer(ctx, flags.WRITE_ONLY, h_b.nbytes)



local_work_size = (64, )
global_work_size = (local_work_size[0] * 256, )



_ = cl.enqueue_copy(queue, h_res, d_c)