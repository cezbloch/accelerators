%%cl_kernel -o "-cl-fast-relaxed-math"

__kernel void reduce_local(const __global int *in_buf, __global int *result)
{
    const uint gid = get_global_id(0);
    const uint group_id = get_group_id(0);
    const int local_id = get_local_id(0);
    
    int sum = work_group_reduce_add(in_buf[gid]);
    if (local_id == 0) {
        result[group_id] = sum;
    }
}