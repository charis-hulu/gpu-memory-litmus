// litmus_message_passing.cu
// Compile: nvcc -O2 litmus_message_passing.cu -o litmus_mp
// Run: ./litmus_mp [iterations]

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

__global__ void mp_kernel(int *X, int *Y, int *finished,
                          int *result, int *counter_weak)
{
    // printf("HELLOADSAD\n");
    int tid = threadIdx.x;
    int r = -1;

    // Wait for start signal
    // while (atomicAdd(start, 0) == 0) {}

    if (tid == 0) {
        // Writer: store data, then flag
        *X = 1;
        *Y = 1;
    }
    else {
        // Reader: wait for flag, then read data
        if (*Y == 1) {
            r = *X;
        } else {
            r = -1; // did not see flag
        }
    }

    // Store result so thread 0 can check it
    result[tid] = r;

    atomicAdd(finished, 1);

    // Wait for both threads to finish
    while (atomicAdd(finished, 0) < 2) {}

    if (tid == 0) {
        int reader_r = result[1];   // Thread1's read
        int flag_val = *Y;

        // Weak outcome: Reader sees flag but not data
        if (flag_val == 1 && reader_r == 0)
            atomicAdd(counter_weak, 1);
    }
}

int main(int argc, char **argv)
{
    long long iterations = 1000;
    if (argc >= 2) iterations = atoll(argv[1]);

    printf("Message Passing Litmus: %lld iterations\n", iterations);

    int *d_X, *d_Y, *d_finished, *d_result, *d_counter_weak;

    CUDA_CHECK(cudaMalloc(&d_X, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_Y, sizeof(int)));
    // CUDA_CHECK(cudaMalloc(&d_start, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_finished, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter_weak, sizeof(int)));

    int host_weak = 0;

    for (long long i = 0; i < iterations; i++) {
        // printf("HELLO\n");
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(d_X, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Y, &zero, sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_start, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_finished, &zero, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_counter_weak, &zero, sizeof(int), cudaMemcpyHostToDevice));

        mp_kernel<<<1, 2>>>(d_X, d_Y, d_finished, d_result, d_counter_weak);
        CUDA_CHECK(cudaGetLastError());
        // int host_start;
        // int one = 1;

        // CUDA_CHECK(cudaMemcpy(d_start, &one, sizeof(int), cudaMemcpyHostToDevice));

        // CUDA_CHECK(cudaMemcpy(&host_start, d_start, sizeof(int), cudaMemcpyDeviceToHost));
        // printf("d_start = %d\n", host_start);

        CUDA_CHECK(cudaDeviceSynchronize());

        int this_weak = 0;
        CUDA_CHECK(cudaMemcpy(&this_weak, d_counter_weak, sizeof(int),
                              cudaMemcpyDeviceToHost));

        host_weak += this_weak;

        if ((i + 1) % 100000 == 0) {
            printf("Iteration %lld, weak outcomes so far: %d\n",
                   i + 1, host_weak);
        }
    }

    printf("\nFinished. Weak outcomes (Y=1 && X=0): %d\n", host_weak);

    cudaFree(d_X);
    cudaFree(d_Y);
    // cudaFree(d_start);
    cudaFree(d_finished);
    cudaFree(d_result);
    cudaFree(d_counter_weak);

    return 0;
}
