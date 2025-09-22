    #define THREAD_INDEX (threadIdx.y * blockDim.x + threadIdx.x)

    __global__ void matmul(float *A, float *B, float *C, int N) {
        float sum = 0;
        int2 global_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                   blockIdx.y * blockDim.y + threadIdx.y);
                                   
        if (global_id.x >= N || global_id.y >= N) {
            return;
        }
                
        extern __shared__ float slm[];
        float *slm_A = &slm[0];
        float *slm_B = &slm[blockDim.x * blockDim.y];
        
        for (int b = 0; b < gridDim.x; b++) {
            int2 gidA = make_int2(b * blockDim.x + threadIdx.x,  global_id.y);
            int2 gidB = make_int2(global_id.x,  b * blockDim.y + threadIdx.y);
                        
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
