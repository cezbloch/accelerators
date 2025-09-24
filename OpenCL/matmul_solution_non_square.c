#define THREAD_INDEX (get_local_id(1) * get_local_size(0) + get_local_id(0))

__kernel void non_square_matmul(__global float *A,
                                __global float *B,
                                __global float *C, 
                                __local float* slm_A, 
                                __local float* slm_B,
                                int M,
                                int N,
                                int P)
{
    float sum = 0.0f;
    int2 global_id = (int2)(get_group_id(0) * get_local_size(0) + get_local_id(0),
                            get_group_id(1) * get_local_size(1) + get_local_id(1));

    int nr_blocks = (N + get_local_size(0) - 1) / get_local_size(0);
        
    for (int b = 0; b < nr_blocks; b++) {
        int2 gidA = (int2)(b * get_local_size(0) + get_local_id(0), global_id.y);
        int2 gidB = (int2)(global_id.x, b * get_local_size(1) + get_local_id(1));

        slm_A[THREAD_INDEX] = A[gidA.y * N + gidA.x];
        slm_B[THREAD_INDEX] = B[gidB.y * P + gidB.x];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < get_local_size(0); i++) {
            sum += slm_A[get_local_size(0) * get_local_id(1) + i] *
                   slm_B[get_local_size(0) * i + get_local_id(0)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[global_id.y * P + global_id.x] = sum;
}