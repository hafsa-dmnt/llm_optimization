#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>

#define MAX_DIGITS 2097152
#define KARAT_CUTOFF 4

void karatsuba(int *a, int *b, int *ret, int d);
void karatsuba_parallel(int *a, int *b, int *ret, int d, int rank, int numprocs);
void gradeSchool(int *a, int *b, int *ret, int d);
void doCarry(int *a, int d);
void getNum(FILE *fp, int *a, int *d_a);
void printNum(int *a, int d);

int main(int argc, char *argv[]) {
    int d_a, d_b, d;
    struct timeval time_start, time_end;
    int *a, *b, *r;

    MPI_Init(&argc, &argv);

    int rank, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    if (rank == 0) {
        FILE *fp = fopen(argv[1], "r");
        if (fp == NULL) {
            perror("Error opening file");
            MPI_Finalize();
            return 1;
        }

        gettimeofday(&time_start, NULL);

        a = (int *)malloc(MAX_DIGITS * sizeof(int));
        b = (int *)malloc(MAX_DIGITS * sizeof(int));
        r = (int *)malloc(6 * MAX_DIGITS * sizeof(int));

        if (a == NULL || b == NULL || r == NULL) {
            perror("Memory allocation error");
            fclose(fp);
            MPI_Finalize();
            return 1;
        }

        getNum(fp, a, &d_a);
        getNum(fp, b, &d_b);

        fclose(fp);

        if (d_a < 0 || d_b < 0) {
            printf("0\n");
            free(a);
            free(b);
            free(r);
            MPI_Finalize();
            return 0;
        }

        d = (d_a > d_b) ? d_a : d_b;
        int i = 0;
        for (i = d_a; i < d; i++) a[i] = 0;
        for (i = d_b; i < d; i++) b[i] = 0;
    }

    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        a = (int *)malloc(d * sizeof(int));
        b = (int *)malloc(d * sizeof(int));
        r = (int *)malloc(6 * d * sizeof(int));

        if (a == NULL || b == NULL || r == NULL) {
            perror("Memory allocation error");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(a, d, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, d, MPI_INT, 0, MPI_COMM_WORLD);

    karatsuba_parallel(a, b, r, d, rank, numprocs);

    if (rank == 0) {
        doCarry(r, 2 * d);
        printNum(r, 2 * d);

        gettimeofday(&time_end, NULL);

        double exec_time = (double)(time_end.tv_sec - time_start.tv_sec) +
                           (double)(time_end.tv_usec - time_start.tv_usec) / 1000000.0;

        printf("\nKernel executed in %lf seconds.\n", exec_time);
    }

    free(a);
    free(b);
    free(r);

    MPI_Finalize();
    return 0;
}


void karatsuba(int *a, int *b, int *ret, int d) {
  int i;
  int *ar = &a[0];
  int *al = &a[d/2];
  int *br = &b[0];
  int *bl = &b[d/2];
  int *asum = &ret[d * 5];
  int *bsum = &ret[d * 5 + d/2];
  int *x1 = &ret[d * 0];
  int *x2 = &ret[d * 1];
  int *x3 = &ret[d * 2];

  if (d <= KARAT_CUTOFF) {
    gradeSchool(a, b, ret, d);
    return;
  }

  for (i = 0; i < d / 2; i++) {
    asum[i] = al[i] + ar[i];
    bsum[i] = bl[i] + br[i];
  }

  karatsuba(ar, br, x1, d/2);
  karatsuba(al, bl, x2, d/2);
  karatsuba(asum, bsum, x3, d/2);

  for (i = 0; i < d; i++) x3[i] = x3[i] - x1[i] - x2[i];
  for (i = 0; i < d; i++) ret[i + d/2] += x3[i];
}

void karatsuba_parallel(int *a, int *b, int *ret, int d, int rank, int numprocs) {
    int i;
    int *ar = &a[0];
    int *al = &a[d / 2];
    int *br = &b[0];
    int *bl = &b[d / 2];
    int *asum = &ret[d * 5];
    int *bsum = &ret[d * 5 + d / 2];
    int *x1 = &ret[d * 0];
    int *x2 = &ret[d * 1];
    int *x3 = &ret[d * 2];

    if (d <= KARAT_CUTOFF) {
        gradeSchool(a, b, ret, d);
        return;
    }

    for (i = 0; i < d / 2; i++) {
        asum[i] = al[i] + ar[i];
        bsum[i] = bl[i] + br[i];
    }

    if (numprocs > 1) {
        MPI_Request reqs[4];
        MPI_Status stats[4];

        int new_d = d / 2;
        int half_procs = numprocs / 2;
        int dest_rank = (rank + half_procs) % numprocs; // Destinataire pour l'envoi MPI

        MPI_Isend(ar, new_d, MPI_INT, dest_rank, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(br, new_d, MPI_INT, dest_rank, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(&ret[d], new_d * 3, MPI_INT, dest_rank, 2, MPI_COMM_WORLD, &reqs[2]);

        karatsuba_parallel(al, bl, x2, new_d, rank, half_procs);

        MPI_Waitall(3, reqs, stats);

        MPI_Recv(x1, new_d * 2, MPI_INT, dest_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(x3, new_d * 2, MPI_INT, dest_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        karatsuba(ar, br, x1, d / 2);
        karatsuba(al, bl, x2, d / 2);
        karatsuba(asum, bsum, x3, d / 2);
    }

    for (i = 0; i < d; i++) x3[i] = x3[i] - x1[i] - x2[i];
    for (i = 0; i < d; i++) ret[i + d / 2] += x3[i];
}




void gradeSchool(int *a, int *b, int *ret, int d) {
  int i, j;

  for (i = 0; i < 2 * d; i++) ret[i] = 0;
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++) ret[i + j] += a[i] * b[j];
  }
}

void doCarry(int *a, int d) {
  int c;
  int i;

  c = 0;
  for (i = 0; i < d; i++) {
    a[i] += c;
    if (a[i] < 0) {
      c = -(-(a[i] + 1) / 10 + 1);
    } else {
      c = a[i] / 10;
    }
    a[i] -= c * 10;
  }
  if (c != 0) fprintf(stderr, "Overflow %d\n", c);
}

void getNum(FILE *fp, int *a, int *d_a) {
  int c;
  int i;

  *d_a = 0;
  while (true) {
    c = fgetc(fp);
    if (c == '\n' || c == EOF) break;
    if (*d_a >= MAX_DIGITS) {
      fprintf(stderr, "using only first %d digits\n", MAX_DIGITS);
      while (c != '\n' && c != EOF) c = fgetc(fp);
    }
    a[*d_a] = c - '0';
    ++(*d_a);
  }

  for (i = 0; i * 2 < *d_a - 1; i++) {
    c = a[i], a[i] = a[*d_a - i - 1], a[*d_a - i - 1] = c;
  }
}

void printNum(int *a, int d) {
  int i;
  for (i = d - 1; i > 0; i--) if (a[i] != 0) break;
  for (; i >= 0; i--) printf("%d", a[i]);
  printf("\n");
}
