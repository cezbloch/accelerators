__kernel void reduce(const __global int *in_buf, __local int *temp, __global int *result)
{
    const uint gid = get_global_id(0);
    const uint group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int lid = get_local_id(0);
    
    temp[lid] = in_buf[gid];

    for (int step = 1; step < local_size; step *= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (gid % (step * 2) == 0)
        {
            temp[lid] = temp[lid] + temp[lid + step];
        }
    }
    
    if (lid == 0) {
        result[group_id] = temp[lid];
    }
}


//------ host code ------

d_intermediate_buffer = cl.LocalMemory(size=local_work_size[0] * 4)
