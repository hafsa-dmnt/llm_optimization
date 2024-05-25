#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

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

	gettimeofday(&time_start, NULL);

	seq[0][0] = '0';
	seq[0][1] = '\0';
	seq[1][0] = '1';
	seq[1][1] = '\0';
	for (i = 2; i < MAX; i++)
		add(i - 1, i - 2);

	n = atoi(argv[1]);

	len = strlen(seq[n]);
	for (i = 0; i <= len - 1; i++)
		printf("%c", seq[n][len - 1 - i]);
	
	gettimeofday(&time_end, NULL);

    double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                      (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

    printf("\nKernel executed in %lf seconds.\n", exec_time);

	printf("\n");
	fflush(stdout);

	return 0;
}
