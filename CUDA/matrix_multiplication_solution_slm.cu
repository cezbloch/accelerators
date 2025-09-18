    #define THREAD_INDEX (threadIdx.y * blockDim.x + threadIdx.x)
    #define BLOCK_LENGTH 8 * 8

    __global__ void matmul(float *A, float *B, float *C, int *debug, int N) {
        float sum = 0;
        int2 global_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);
                                   
        if (global_id.x >= N || global_id.y >= N) {
            return;
        }
                
        //extern __shared__ float slm[];
        __shared__ float slm_A[BLOCK_LENGTH];
        __shared__ float slm_B[BLOCK_LENGTH];
        
        for (int b = 0; b < gridDim.x; b++) {
            int2 gidA = make_int2(b * blockDim.x + threadIdx.x,  blockIdx.y * blockDim.y + threadIdx.y);
            int2 gidB = make_int2(blockIdx.x * blockDim.x + threadIdx.x,  b * blockDim.y + threadIdx.y);
                        
            slm_A[THREAD_INDEX] = A[gidA.y * N + gidA.x];
            slm_B[THREAD_INDEX] = B[gidB.y * N + gidB.x];
                
            __syncthreads();
            
            for (int i = 0; i < blockDim.x; i++) {
                sum += slm_A[blockDim.x * threadIdx.y + i] * slm_B[blockDim.x * i + threadIdx.x];
            }            
            
            __syncthreads();
        }
        
        C[global_id.y * N + global_id.x] = sum;
    }
    