#define THREAD_INDEX (get_local_id(1) * get_local_size(0) + get_local_id(0))

__kernel void shared_mem_kernel(__global float *A,
                     __global float *B,
                     __global float *C, 
                    __local float* slm_A, 
                    __local float* slm_B,
                    int N)
{
    float sum = 0.0f;
    int2 global_id = (int2)(get_group_id(0) * get_local_size(0) + get_local_id(0),
                            get_group_id(1) * get_local_size(1) + get_local_id(1));

    for (int b = 0; b < get_num_groups(0); b++) {
        int2 gidA = (int2)(b * get_local_size(0) + get_local_id(0), global_id.y);
        int2 gidB = (int2)(global_id.x, b * get_local_size(1) + get_local_id(1));

        slm_A[THREAD_INDEX] = A[gidA.y * N + gidA.x];
        slm_B[THREAD_INDEX] = B[gidB.y * N + gidB.x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < get_local_size(0); i++) {
            sum += slm_A[get_local_size(0) * get_local_id(1) + i] *
                   slm_B[get_local_size(0) * i + get_local_id(0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[global_id.y * N + global_id.x] = sum;
}