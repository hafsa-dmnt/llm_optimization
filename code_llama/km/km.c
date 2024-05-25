#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define RANDNUM_W 521288629
#define RANDNUM_Z 362436069

unsigned int randum_w = RANDNUM_W;
unsigned int randum_z = RANDNUM_Z;

void srandnum(int seed) {
    unsigned int w, z;
    w = (seed * 104623) & 0xffffffff;
    randum_w = (w) ? w : RANDNUM_W;
    z = (seed * 48947) & 0xffffffff;
    randum_z = (z) ? z : RANDNUM_Z;
}

unsigned int randnum(void) {
    unsigned int u;
    randum_z = 36969 * (randum_z & 65535) + (randum_z >> 16);
    randum_w = 18000 * (randum_w & 65535) + (randum_w >> 16);
    u = (randum_z << 16) + randum_w;
    return (u);
}

typedef float* vector_t;

int npoints;
int dimension;
int ncentroids;
float mindistance;
int seed;
vector_t *data;
int *map;
vector_t *centroids;
int *dirty;
int too_far;
int has_changed;

float v_distance(vector_t a, vector_t b) {
    int i;          
    float distance = 0; 
    for (i = 0; i < dimension; i++)
        distance +=  pow(a[i] - b[i], 2);
    return sqrt(distance);
}

static void populate(void) {
    int i, j;
    float tmp;
    float distance;
    too_far = 0;
    for (i = 0; i < npoints; i++) {	
        distance = v_distance(centroids[map[i]], data[i]);		
        /* Look for closest cluster. */
        for (j = 0; j < ncentroids; j++) {
            /* Point is in this cluster. */
            if (j == map[i]) continue;				
            tmp = v_distance(centroids[j], data[i]);
            if (tmp < distance) {
                map[i] = j;
                distance = tmp;
                dirty[j] = 1;
            }
        }		
        /* Cluster is too far away. */
        if (distance > mindistance)
            too_far = 1;
    }
}

static void compute_centroids(void) {
    int i, j, k;
    int population;
    has_changed = 0;
    /* Compute means. */	
    for (i = 0; i < ncentroids; i++) {		
        if (!dirty[i]) continue;
        memset(centroids[i], 0, sizeof(float) * dimension);
        /* Compute cluster's mean. */
        population = 0;
        for (j = 0; j < npoints; j++) {
            if (map[j] != i)
                continue;
            for (k = 0; k < dimension; k++)
                centroids[i][k] += data[j][k];
            population++;
        }		
        if (population > 1) {
            for (k = 0; k < dimension; k++)
                centroids[i][k] *= 1.0/population;
        }
        has_changed = 1;
    }
    memset(dirty, 0, ncentroids * sizeof(int));
}

int* kmeans(void) {
    int i, j, k; 
    too_far = 0;
    has_changed = 0;
    if (!(map  = calloc(npoints, sizeof(int)))) exit (1);
    if (!(dirty = malloc(ncentroids*sizeof(int)))) exit (1);
    if (!(centroids = malloc(ncentroids*sizeof(vector_t)))) exit (1);
    for (i = 0; i < ncentroids; i++)
        centroids[i] = malloc(sizeof(float) * dimension);
    for (i = 0; i < npoints; i++) map[i] = -1;
    for (i = 0; i < ncentroids; i++) {
        dirty[i] = 1;
        j = randnum() % npoints;
        for (k = 0; k < dimension; k++)
            centroids[i][k] = data[j][k];
        map[j] = i;
    }
    /* Map unmapped data points. */
    for (i = 0; i < npoints; i++)
        if (map[i] < 0) map[i] = randnum() % ncentroids;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int npoints_per_proc = npoints / MPI_Comm_size(MPI_COMM_WORLD, NULL);
    int start = rank * npoints_per_proc;
    int end = start + npoints_per_proc;

    do 	{/* Cluster data. */
        MPI_Barrier(MPI_COMM_WORLD);
        for (i = start; i < end; i++) {
            float distance;
            for (j = 0; j < ncentroids; j++) {
                distance = v_distance(centroids[j], data[i]);
                if (distance > mindistance)
                    too_far = 1;
                if (j == map[i]) continue;
                if (distance < v_distance(centroids[map[i]], data[i])) {
                    map[i] = j;
                    dirty[j] = 1;
                }
            }
        }
        MPI_Allreduce(&too_far, &too_far, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        compute_centroids();
        MPI_Allreduce(&has_changed, &has_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    } while (too_far && has_changed);

    for (i = 0; i < ncentroids; i++)
        free(centroids[i]);
    free(centroids);
    free(dirty);	
    return map;
}

int main(int argc, char ** argv) {
    int i, j, tmp;
    int rank, size;
    struct timeval time_start;
    struct timeval time_end;
    int ierr;

    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }

    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Comm_rank failed\n");
        return 1;
    }

    ierr = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (ierr != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Comm_size failed\n");
        return 1;
    }

    printf("Size: %d\n", size);

    if (argc != 6) {
        fprintf(stderr, "Usage: %s <npoints> <dimension> <ncentroids> <mindistance> <seed>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    gettimeofday(&time_start, NULL);

    npoints = atoi(argv[1]);
    dimension = atoi(argv[2]);
    ncentroids = atoi(argv[3]);
    mindistance = strtof(argv[4], NULL);
    seed = atoi(argv[5]);

    srandnum(seed);
    
    if (!(data = malloc(npoints*sizeof(vector_t)))) exit(1);

        for (i = 0; i < npoints; i++)
        data[i] = malloc(sizeof(float) * dimension);

    for (i = 0; i < npoints; i++) {
        for (j = 0; j < dimension; j++) {
            data[i][j] = (float)randnum() / RANDNUM_W;
        }
    }

    map = kmeans();

    gettimeofday(&time_end, NULL);

    printf("Time taken: %f seconds\n", (time_end.tv_sec - time_start.tv_sec) + (time_end.tv_usec - time_start.tv_usec) / 1000000.0);

    for (i = 0; i < npoints; i++)
        free(data[i]);
    free(data);
    free(map);

    MPI_Finalize(); // Finalize MPI environment

    return 0;
}