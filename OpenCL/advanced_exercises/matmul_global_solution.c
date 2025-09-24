__kernel void matmul(__global float* A, __global float* B, __global float* C, int N) {
    float sum = 0;
    int2 global_id = (int2)(get_global_id(0), get_global_id(1));

    if (global_id.x >= N || global_id.y >= N) {
        return;
    }
       
    for (int i = 0; i < N; i++) {
        int aij = global_id.y * N + i;
        int bij = i * N + global_id.x;
        sum  += A[aij] * B[bij];
    }

    int cij = global_id.y * N + global_id.x;
    C[cij] = sum;    
}