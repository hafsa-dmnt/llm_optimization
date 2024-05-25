#include <stdio.h>
#include <stdlib.h>
#include "shellsort.h"
#include <time.h>
#include <sys/time.h>

#define LENGTH 8

FILE *fin, *fout;

char *strings;
long int N;

void openfiles() {
	fin = fopen("./test_input.txt", "r+");
	if (fin == NULL) {
		perror("fopen fin");
		exit(EXIT_FAILURE);
	}

	fout = fopen("./shellsort.out", "w");
	if (fout == NULL) {
		perror("fopen fout");
		exit(EXIT_FAILURE);
	}
}

void closefiles(void) {
	fclose(fin);
	fclose(fout);
}

int main(int argc, char* argv[]) {

	long int i;
	struct timeval time_start;
    struct timeval time_end;

	gettimeofday(&time_start, NULL);
	openfiles();

	fscanf(fin, "%ld", &N);
	strings = (char*) malloc(N * LENGTH);
	if (strings == NULL) {
		perror("malloc strings");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < N; i++)
		fscanf(fin, "%s", strings + (i * LENGTH));

	shell_sort(strings, LENGTH, N);

	for (i = 0; i < N; i++)
		fprintf(fout, "%s\n", strings + (i * LENGTH));

	gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                      (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    printf("\nKernel executed in %lf seconds.\n", exec_time);

	free(strings);
	closefiles();

	return EXIT_SUCCESS;
}
