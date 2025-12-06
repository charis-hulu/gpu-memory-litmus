#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__device__ volatile int X = 0, Y = 0, rX = -1, rY = -1;

__device__ void time_noise(){
    for (int i = 0; i < (clock64() & 0xFF); i++) {asm volatile("");}
}

__global__ void sb_kernel(int *start, int *finished,
                          int *counter00, int *counter01,
                          int *counter10, int *counter11)
{
    int tid = blockIdx.x;

    // Simple barrier: increment start and wait for all threads
    atomicAdd(start, 1);
    while (atomicAdd(start, 0) < 2) {}

    time_noise();

    if (tid == 0) {
        // Writer: store data then flag
        X = 1;
        time_noise();
        Y = 1;
    } else {
        // Reader: read flag and data
        rY = Y;
        time_noise();
        rX = X;
    }

    // Device-side barrier: mark finished
    atomicAdd(finished, 1);
    while (atomicAdd(finished, 0) < 2) {}

    if (tid == 1) {
        // Count all 4 possible outcomes
        if (rY == 0 && rX == 0) atomicAdd(counter00, 1);
        if (rY == 0 && rX == 1) atomicAdd(counter01, 1);
        if (rY == 1 && rX == 0) atomicAdd(counter10, 1);
        if (rY == 1 && rX == 1) atomicAdd(counter11, 1);
    }
}

int main(int argc, char **argv)
{
    long long iterations = 1000000;
    if (argc >= 2) iterations = atoll(argv[1]);

    printf("Message Passing Litmus: %lld iterations\n", iterations);

    int *d_X, *d_Y, *d_start, *d_finished;
    int *d_counter00, *d_counter01, *d_counter10, *d_counter11;

    CUDA_CHECK(cudaMalloc(&d_X, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Y, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_finished, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_counter00, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter01, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter10, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter11, sizeof(int)));

    int host_00 = 0, host_01 = 0, host_10 = 0, host_11 = 0;

    for (long long i = 0; i < iterations; i++) {
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_X, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Y, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_start, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_finished, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_counter00, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_counter01, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_counter10, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_counter11, &zero, sizeof(int), cudaMemcpyHostToDevice));

        cudaMemcpyToSymbol(X, &zero, sizeof(int));
        cudaMemcpyToSymbol(Y, &zero, sizeof(int));

        sb_kernel<<<2, 1>>>(d_start, d_finished,
                            d_counter00, d_counter01, d_counter10, d_counter11);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int tmp;
        CUDA_CHECK(cudaMemcpy(&tmp, d_counter00, sizeof(int), cudaMemcpyDeviceToHost));
        host_00 += tmp;
        CUDA_CHECK(cudaMemcpy(&tmp, d_counter01, sizeof(int), cudaMemcpyDeviceToHost));
        host_01 += tmp;
        CUDA_CHECK(cudaMemcpy(&tmp, d_counter10, sizeof(int), cudaMemcpyDeviceToHost));
        host_10 += tmp;
        CUDA_CHECK(cudaMemcpy(&tmp, d_counter11, sizeof(int), cudaMemcpyDeviceToHost));
        host_11 += tmp;

        if ((i + 1) % 100000 == 0) {
            printf("After %lld iterations:\n", i + 1);
            printf("(rY,rX)=(0,0): %d\n", host_00);
            printf("(rY,rX)=(0,1): %d\n", host_01);
            printf("(rY,rX)=(1,0): %d\n", host_10);
            printf("(rY,rX)=(1,1): %d\n", host_11);
        }
    }

    printf("\nOutcome counts after %lld iterations:\n", iterations);
    printf("(rY,rX)=(0,0): %d\n", host_00);
    printf("(rY,rX)=(0,1): %d\n", host_01);
    printf("(rY,rX)=(1,0): %d\n", host_10);
    printf("(rY,rX)=(1,1): %d\n", host_11);

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_start);
    cudaFree(d_finished);
    cudaFree(d_counter00);
    cudaFree(d_counter01);
    cudaFree(d_counter10);
    cudaFree(d_counter11);

    return 0;
}