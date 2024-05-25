#include <cuda_runtime.h>

__device__ int myMax(int a, int b) {
    return (a > b) ? a : b;
}

__global__ void MaxIncreasingSubKernel(int *arr, int n, int k, int *dp, int *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int j = 0; j < k; j++) {
            dp[idx * (k + 1) + j] = -1;
        }
        dp[idx * (k + 1) + 1] = arr[idx];
    }
    __syncthreads();

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (arr[j] < arr[i]) {
                for (int l = 1; l <= k - 1; l++) {
                    if (dp[j * (k + 1) + l] != -1) {
                        dp[i * (k + 1) + l + 1] = myMax(dp[i * (k + 1) + l + 1], dp[j * (k + 1) + l] + arr[i]);
                    }
                }
            }
        }
    }
    __syncthreads();

    int ans = -1;
    for (int i = 0; i < n; i++) {
        if (ans < dp[i * (k + 1) + k]) {
            ans = dp[i * (k + 1) + k];
        }
    }
    *result = ans;
}

extern "C" void launch_MaxIncreasingSubKernel(int *arr, int n, int k, int *dp, int *result) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    MaxIncreasingSubKernel<<<gridSize, blockSize>>>(arr, n, k, dp, result);
}