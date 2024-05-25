#include <stdio.h>
#include <stdlib.h>
#include <openssl/md5.h>
#include <string.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define MAX 10

typedef unsigned char byte;

char letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";
int n_letters = sizeof(letters) - 1;

/*
 * Convert hexadecimal string to hash byte.
*/
void strHex_to_byte(char * str, byte * hash){
    char * pos = str;
    int i;

    for (i = 0; i < MD5_DIGEST_LENGTH/sizeof *hash; i++) {
        sscanf(pos, "%2hhx", &hash[i]);
        pos += 2;
    }
}

/*
 * GPU Kernel to generate all combinations of possible letters and compare hashes.
*/
__global__ void iterate_gpu(byte *hash1, int len, int n_letters, char *letters, int *ok) {
    char str[MAX+1];
    byte hash2[MD5_DIGEST_LENGTH];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = 0; i < len; i++) {
        int pos = (idx / (int)powf((float)n_letters, i)) % n_letters;
        str[i] = letters[pos];
    }
    str[len] = '\0';

    if (*ok) return;

    MD5((const byte *)str, len, hash2);

    bool match = true;
    for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        if (hash1[i] != hash2[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        printf("Found: %s\n", str);
        *ok = 1;
    }
}

int main(int argc, char **argv) {
    char str[MAX+1];
    int lenMax = MAX;
    int len;
    int ok = 0, r;
    char hash1_str[2*MD5_DIGEST_LENGTH+1];
    byte hash1[MD5_DIGEST_LENGTH]; // password hash
    byte hash1_d[MD5_DIGEST_LENGTH];
    int *ok_d;
    char *letters_d;
    struct timeval time_start;
    struct timeval time_end;

    // Input:
    r = scanf("%s", hash1_str);

    // Check input.
    if (r == EOF || r == 0) {
        fprintf(stderr, "Error!\n");
        exit(1);
    }

    gettimeofday(&time_start, NULL);
    printf("Starting searching\n");

    // Convert hexadecimal string to hash byte.
    strHex_to_byte(hash1_str, hash1);

    // Allocate device memory
    cudaMalloc((void**)&hash1_d, MD5_DIGEST_LENGTH);
    cudaMalloc((void**)&ok_d, sizeof(int));
    cudaMalloc((void**)&letters_d, n_letters * sizeof(char));

    // Copy data to device
    cudaMemcpy(hash1_d, hash1, MD5_DIGEST_LENGTH, cudaMemcpyHostToDevice);
    cudaMemcpy(ok_d, &ok, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(letters_d, letters, n_letters * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (int)powf((float)n_letters, lenMax) / blockSize + 1;

    // Generate all possible passwords of different sizes.
    for (len = 1; len <= lenMax; len++) {
        iterate_gpu<<<numBlocks, blockSize>>>(hash1_d, len, n_letters, letters_d, ok_d);
        cudaDeviceSynchronize();
        cudaMemcpy(&ok, ok_d, sizeof(int), cudaMemcpyDeviceToHost);
        if (ok) break;
    }

    gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                      (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    printf("\nKernel executed in %lf seconds.\n", exec_time);

    // Free device memory
    cudaFree(hash1_d);
    cudaFree(ok_d);
    cudaFree(letters_d);

    return EXIT_SUCCESS;
}
