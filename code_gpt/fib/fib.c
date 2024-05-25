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
    int n, i, len;
    struct timeval time_start;
    struct timeval time_end;

    MPI_Init(&argc, &argv);

    int rank, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int local_n = MAX / numprocs;
    char *local_seq = (char *)malloc(local_n * LEN * sizeof(char));

    if (rank == 0) {
        gettimeofday(&time_start, NULL);

        seq[0][0] = '0';
        seq[0][1] = '\0';
        seq[1][0] = '1';
        seq[1][1] = '\0';
    }

    MPI_Scatter(seq, local_n * LEN, MPI_CHAR, local_seq, local_n * LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (i = 2 + rank * local_n; i < 2 + (rank + 1) * local_n; i++) {
        add(i - 1, i - 2);
    }

    MPI_Gather(local_seq, local_n * LEN, MPI_CHAR, seq, local_n * LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        n = atoi(argv[1]);

        len = strlen(seq[n]);
        printf("Fibonacci number at index %d: ", n);
        for (i = 0; i <= len - 1; i++)
            printf("%c", seq[n][len - 1 - i]);

        printf("\nContents of seq after computation:\n");
        for (i = 0; i < 20; i++) {
            printf("seq[%d]: %s\n", i, seq[i]);
        }

        gettimeofday(&time_end, NULL);

        double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                        (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

        printf("\nKernel executed in %lf seconds.\n", exec_time);
        printf("\n");
        fflush(stdout);
    }


    free(local_seq);

    MPI_Finalize();
    return 0;
}
