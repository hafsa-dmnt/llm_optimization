#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define MAX 100010
#define LEN 25001

char seq[MAX][LEN];

void add(int a, int b) {
    int i, aux, s;

    for (i = 0, aux = 0; seq[a][i] != '\0' && seq[b][i] != '\0'; i++) {
        s = seq[a][i] + seq[b][i] + aux - '0' - '0';
        aux = s / 10;
        seq[a + 1][i] = s % 10 + '0';
    }

    while (seq[a][i] != '\0') {
        s = seq[a][i] + aux - '0';
        aux = s / 10;
        seq[a + 1][i] = s % 10 + '0';
        i++;
    }

    while (seq[b][i] != '\0') {
        s = seq[b][i] + aux - '0';
        aux = s / 10;
        seq[a + 1][i] = s % 10 + '0';
        i++;
    }

    if (aux != 0)
        seq[a + 1][i++] = aux + '0';
    seq[a + 1][i] = '\0';
}

int main(int argc, char *argv[]) {
    int n, i, len, rank, size;
    struct timeval time_start;
    struct timeval time_end;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gettimeofday(&time_start, NULL);

    if (rank == 0) {
        seq[0][0] = '0';
        seq[0][1] = '\0';
        seq[1][0] = '1';
        seq[1][1] = '\0';
    }

    char *seq_buf = malloc((size_t)MAX * LEN * sizeof(char));

    if (rank == 0) {
        size_t j = 0;
        for (j = 0; j < MAX; j++) {
            strcpy(seq_buf + j * LEN, seq[j]);
        }
    }

    MPI_Bcast(seq_buf, (size_t)MAX * LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        size_t j = 0;
        for (j = 0; j < MAX; j++) {
            strcpy(seq[j], seq_buf + j * LEN);
        }
    }

    free(seq_buf);

    int start = 2 + rank;
    int end = start + size - 1;
    for (i = start; i < end; i += size)
        add(i - 1, i - 2);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        n = atoi(argv[1]);

        len = strlen(seq[n]);
        printf("Fibonacci number %d: ", n);
        for (i = 0; i <= len - 1; i++)
            printf("%c", seq[n][len - 1 - i]);
        printf("\n");

        gettimeofday(&time_end, NULL);

        double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                          (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

        printf("Kernel executed in %lf seconds.\n", exec_time);

        printf("\n");
        fflush(stdout);
    }

    MPI_Finalize();

    return 0;
}