__kernel void reduce(const __global int *in_buf, __local int *temp, __global int *result)
{
    const uint gid = 2 * get_global_id(0);
    const uint group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int lid = get_local_id(0);
    
    temp[lid] = in_buf[gid]  + in_buf[gid + 1];   

    for (int active_threads = local_size/2; active_threads > 0; active_threads /= 2)
    {           
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < active_threads) 
        {
            temp[lid] = temp[lid] + temp[lid + active_threads];
        }
    }
    
    if (lid == 0) {
        result[group_id] = temp[0];
    }
}


//------ host side code ------

d_intermediate_buffer = cl.LocalMemory(size=local_work_size[0] * 4)
