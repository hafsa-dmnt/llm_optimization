#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>

#define MAX_DIGITS 2097152
#define KARAT_CUTOFF 4

void karatsuba(int *a, int *b, int *ret, int d, int rank);
void gradeSchool(int *a, int *b, int *ret, int d);
void doCarry(int *a, int d);
void getNum(FILE *fp, int *a, int *d_a);
void printNum(int *a, int d);

int main(int argc, char *argv[]) {
  int d_a, d_b, d;
  int i;
  int rank, size;
  struct timeval time_start, time_end;

  FILE *fp;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int *a = (int *)malloc(MAX_DIGITS * sizeof(int));
  int *b = (int *)malloc(MAX_DIGITS * sizeof(int));
  int *ret = (int *)malloc(6 * MAX_DIGITS * sizeof(int));
  if (a == NULL || b == NULL || ret == NULL) {
    printf("Error allocating memory\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rank == 0) {
    fp = fopen(argv[1], "r");
    if (fp == NULL) {
      perror("Error opening file");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    gettimeofday(&time_start, NULL);
    getNum(fp, a, &d_a);
    getNum(fp, b, &d_b);
    fclose(fp);
    if (d_a < 0 || d_b < 0) {
      printf("0\n");
      MPI_Finalize();
      return 0;
    }
    i = (d_a > d_b) ? d_a : d_b;
    for (d = 1; d < i; d *= 2);
    for (i = d_a; i < d; i++) a[i] = 0;
    for (i = d_b; i < d; i++) b[i] = 0;
  }

  MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(a, d, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b, d, MPI_INT, 0, MPI_COMM_WORLD);

  karatsuba(a, b, ret, d, rank);
  doCarry(ret, 2 * d);

  if (rank == 0) {
    printNum(ret, 2 * d);
    gettimeofday(&time_end, NULL);
    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                       (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;
    printf("\nKernel executed in %lf seconds.\n", exec_time);
  }

  free(a);
  free(b);
  free(ret);
  MPI_Finalize();
  return 0;
}

void karatsuba(int *a, int *b, int *ret, int d, int rank) {
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

  MPI_Comm comm;
  int err = MPI_Comm_split(MPI_COMM_WORLD, rank < 3, rank, &comm);
  if (err != MPI_SUCCESS) {
    printf("Error creating new communicator: %d\n", err);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rank == 0) {
    karatsuba(ar, br, x1, d/2, rank);
  } else if (rank == 1) {
    karatsuba(al, bl, x2, d/2, rank);
  } else {
    karatsuba(asum, bsum, x3, d/2, rank);
  }

  err = MPI_Comm_free(&comm);
  if (err != MPI_SUCCESS) {
    printf("Error freeing communicator: %d\n", err);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  err = MPI_Allreduce(MPI_IN_PLACE, x1, d, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS) {
    printf("Error in Allreduce 1: %d\n", err);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  err = MPI_Allreduce(MPI_IN_PLACE, x2, d, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS) {
    printf("Error in Allreduce 2: %d\n", err);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  err = MPI_Allreduce(MPI_IN_PLACE, x3, d, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (err != MPI_SUCCESS) {
    printf("Error in Allreduce 3: %d\n", err);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (i = 0; i < d; i++) {
    if (i + d/2 < 2 * d) {
      ret[i + d/2] += x3[i] - x1[i] - x2[i];
    }
  }
}

void gradeSchool(int *a, int *b, int *ret, int d) {
  int i, j;

  for (i = 0; i < 2 * d; i++) ret[i] = 0;
  for (i = 0; i < d; i++) {
    for (j = 0; j < d; j++) {
      if (i + j < 2 * d) {
        ret[i + j] += a[i] * b[j];
      }
    }
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