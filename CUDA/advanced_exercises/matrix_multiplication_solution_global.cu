__global__ void matmul(float *A, float *B, float *C, int N) {
    // blockIdx, blockDim, threadIdx, gridDim
    float sum = 0;
    int2 global_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                               blockIdx.y * blockDim.y + threadIdx.y);

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