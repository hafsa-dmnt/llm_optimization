#include "shellsort.h"
#include <string.h>

void shell_sort_pass(char *a, int length, long int size, int interval) {
	int i;
	for (i = 0; i < size; i++) {
		/* Insert a[i] into the sorted sublist */
		int j;

		char v[length];
		strcpy(v, a + i * length);

		for (j = i - interval; j >= 0; j -= interval) {
			if (strcmp(a + j * length, v) <= 0)
				break;
			strcpy(a + (j + interval) * length, a + j * length);
		}
		strcpy(a + (j + interval) * length, v);
	}
}

void shell_sort(char *a, int length, long int size) {
	int ciura_intervals[] = { 701, 301, 132, 57, 23, 10, 4, 1 };
	double extend_ciura_multiplier = 2.3;

	int interval_idx = 0;
	int interval = ciura_intervals[0];
	if (size > interval) {
		while (size > interval) {
			interval_idx--;
			interval = (int) (interval * extend_ciura_multiplier);
		}
	} else {
		while (size < interval) {
			interval_idx++;
			interval = ciura_intervals[interval_idx];
		}
	}

	while (interval > 1) {
		interval_idx++;
		if (interval_idx >= 0) {
			interval = ciura_intervals[interval_idx];
		} else {
			interval = (int) (interval / extend_ciura_multiplier);
		}

		shell_sort_pass(a, length, size, interval);
	}

}

