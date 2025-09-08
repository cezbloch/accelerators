%%cl_kernel

// This function is deliberately made inefficient to demonstrate some complex computations
float f(float x)
{
    float y = 0;
    // make sure the result is just cos(x) but slow the execution down
    for (int i = -100; i < 1; i++)
    {
        y = cos(x + i);
    }
    return y;
}


__kernel void compute_derivative_slm(const __global float *x, 
                                 __global float *y_prime, 
                                 int nr_elements)
{
    __local float y_local[64];
    
    const int lid = get_local_id(0);
    const int local_size = get_local_size(0) - 1;
    const int num_groups = get_num_groups(0);
    const int group_id = get_group_id(0);
    
    const int grid_stride = num_groups * local_size;
    const int thread_id = lid  + group_id * local_size;

    for(int i = thread_id; i < nr_elements; i+=grid_stride) {
        float x_0 = x[i];
        float y_0 = f(x_0);
        
        y_local[lid] = y_0;
        
        barrier(CLK_LOCAL_MEM_FENCE); // -----------------------------

        float y_1 = y_local[lid + 1];
        
        if (lid < local_size) 
        {
            float x_1 = x[i + 1];
            y_prime[i] = (y_1 - y_0) / (x_1 - x_0);
        }
    }    
}