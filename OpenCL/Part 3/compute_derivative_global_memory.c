%%cl_kernel

float f(float x)
{
    return cos(x);
}

__kernel void compute_derivative(const __global float *x,
                                 __global float *y_prime,
                                 int nr_elements)
{
    int thread_id = get_global_id(0);
    int grid_stride = get_global_size(0);
    
    for(int i = thread_id; i < nr_elements - 1; i+=grid_stride) {
        float x_0 = x[i];
        float x_1 = x[i + 1];
                
        float y_0 = f(x_0);
        float y_1 = f(x_1);
        
        y_prime[i] = (y_1 - y_0) / (x_1 - x_0);
    }    
}