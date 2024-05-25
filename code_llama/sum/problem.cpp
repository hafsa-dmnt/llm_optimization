#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

extern "C" void launch_MaxIncreasingSubKernel(int *arr, int n, int k, int *dp, int *result);

int main() {
    int n, k;
    std::ifstream infile("sum.in"); // Open the file

    if (!infile) {
        std::cerr << "Unable to open file input.txt";
        return 1; // Exit if the file cannot be opened
    }

    infile >> n;
    infile >> k;
    int *arr = new int[n];
    for (int i = 0; i < n; i++) {
        infile >> arr[i];
    }

    infile.close(); // Close the file after reading

    std::cout << "Started to read \n";

    int *d_arr, *d_dp, *d_result;
    cudaMalloc((void **)&d_arr, n * sizeof(int));
    cudaMalloc((void **)&d_dp, n * (k + 1) * sizeof(int));
    cudaMalloc((void **)&d_result, sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    launch_MaxIncreasingSubKernel(d_arr, n, k, d_dp, d_result);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "Kernel executed in " << elapsedTime << " milliseconds.\n";

    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Answer: " << result << std::endl;

    cudaFree(d_arr);
    cudaFree(d_dp);
    cudaFree(d_result);

    delete[] arr;

    return 0;
}