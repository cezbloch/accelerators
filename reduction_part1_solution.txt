 %%cl_kernel -o "-cl-fast-relaxed-math"

// Task 2 - consider all available memory types
__kernel void reduce(const __global int *in_buf, __global int *temp, __global int *result)
{
    const uint gid = get_global_id(0);
    const uint group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int local_id = get_local_id(0);
    
    // Task 2 - what ids should be used to store and later retrieve intermediate results?
    temp[gid] = in_buf[gid];

    // Task 3 - Is this looping most convenient for calculations?
    for (int step = 1; step < local_size; step *= 2)
    {   
        // Task 1 - fix synchronization
        barrier(CLK_GLOBAL_MEM_FENCE);
        if (gid % (step * 2) == 0) // Task 3 - think which threads perform the actual addition
        {
            temp[gid] = temp[gid] + temp[gid + step];
        }        
    }
    
    if (local_id == 0) {
        result[group_id] = temp[gid];
    }
}