#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

__device__ void iterate(uint64_t *hash1, uint64_t *hash2, char *letters, int len, int *ok) {
    int i = threadIdx.x;
    if (i < len) {
        char password[5];
        password[0] = letters[i];
        password[1] = '\0';

        uint64_t h1 = 0x67452301;
        uint64_t h2 = 0xefcdab89;
        uint64_t h3 = 0x98badcfe;
        uint64_t h4 = 0x10325476;

        uint64_t w[16];
        for (int j = 0; j < 4; j++) {
            w[j] = ((uint64_t)password[j]) << 24;
        }

        for (int j = 0; j < 64; j += 4) {
            uint64_t f = 0;
            if (j < 16) {
                f = (h1 & h2) | (~h1 & h3);
            } else if (j < 32) {
                f = (h1 & h3) | (h2 & ~h3);
            } else if (j < 48) {
                f = h1 ^ h2 ^ h3;
            } else {
                f = h1 ^ (h2 | ~h3);
            }

            f = (f << 3) | (f >> 61);
            h1 = h2;
            h2 = h3;
            h3 = h4;
            h4 = h4 + f + w[j / 4];
        }

        hash2[0] = h1;
        hash2[1] = h2;
        hash2[2] = h3;
        hash2[3] = h4;

        int j;
        for (j = 0; j < 4; j++) {
            if (hash1[j] != hash2[j]) {
                break;
            }
        }

        if (j == 4) {
            *ok = 1;
        }
    }
}

__global__ void iterateKernel(uint64_t *hash1, uint64_t *hash2, char *letters, int len, int *ok) {
    iterate(hash1, hash2, letters, len, ok);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <hash_to_crack>\n", argv[0]);
        return 1;
    }

    uint64_t hash1[4]; // hash to crack
    char letters[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"; // characters to try
    int len = strlen(letters);

    // initialize hash1 with the hash to crack
    char *hash_str = argv[1];
    for (int i = 0; i < 4; i++) {
        sscanf(hash_str + i*16, "%16llx", &hash1[i]);
    }

    int ok = 0;

    uint64_t *d_hash1, *d_hash2;
    char *d_letters;
    int *d_ok;

    cudaMalloc((void **)&d_hash1, 4 * sizeof(uint64_t));
    cudaMalloc((void **)&d_hash2, 4 * sizeof(uint64_t));
    cudaMalloc((void **)&d_letters, len * sizeof(char));
    cudaMalloc((void **)&d_ok, sizeof(int));

    cudaMemcpy(d_hash1, hash1, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_letters, letters, len * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ok, &ok, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((len + blockSize.x - 1) / blockSize.x);

    clock_t start = clock();
    iterateKernel<<<gridSize, blockSize>>>(d_hash1, d_hash2, d_letters, len, d_ok);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(&ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);

    if (!ok) {
        printf("Password not found.\n");
    } else {
        printf("Password found!\n");
    }

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", elapsed);

    cudaFree(d_hash1);
    cudaFree(d_hash2);
    cudaFree(d_letters);
    cudaFree(d_ok);

    return 0;
}