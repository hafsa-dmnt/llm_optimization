#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define STABILITY 1.0f/sqrt(3.0f)

void mdf_heat(double ***  __restrict__ u0, 
              double ***  __restrict__ u1, 
              const unsigned int npX, 
              const unsigned int npY, 
              const unsigned int npZ,
              const double deltaH,
              const double deltaT,
              const double inErr,
              const double boundaries, 
              int rank, int size) {
    register double alpha = deltaT / (deltaH * deltaH);
    int continued = 1;
    unsigned int steps = 0;
    MPI_Request req[6];
    MPI_Status stat[6];

    const unsigned int maxIterations = 1500; // adjust this value as needed
    unsigned int iteration = 0;

    while (continued && iteration <= maxIterations){
        steps++;
        for (unsigned int i = rank; i < npZ; i += size){
            for (unsigned int j = 0; j < npY; j++){
                for (unsigned int k = 0; k < npX; k++){
                    register double left   = boundaries;
                    register double right  = boundaries;
                    register double up     = boundaries;
                    register double down   = boundaries;
                    register double top    = boundaries;
                    register double bottom = boundaries;
                    
                    if ((k > 0) && (k < (npX - 1))){
                        left  = u0[i][j][k-1];
                        right = u0[i][j][k+1];
                    }else if (k == 0) right = u0[i][j][k+1];
                    else left = u0[i][j][k-1];
                    
                    if ((j > 0) && (j < (npY - 1))){
                        up  = u0[i][j-1][k];
                        down = u0[i][j+1][k];
                    }else if (j == 0) down = u0[i][j+1][k];
                    else up = u0[i][j-1][k];
                    
                    if ((i > 0) && (i < (npZ - 1))){
                        top  = u0[i-1][j][k];
                        bottom = u0[i+1][j][k];
                    }else if (i == 0) bottom = u0[i+1][j][k];
                    else top = u0[i-1][j][k];
                    
                    u1[i][j][k] =  alpha * ( top + bottom + up + down + left + right  - (6.0f * u0[i][j][k] )) + u0[i][j][k];
                }
            }
        } 
        
        double ***ptr = u0;
        u0 = u1;
        u1 = ptr;
        
        double local_err = 0.0f;
        double global_err = 0.0f;
        int local_continued = 1;
        int global_continued = 1;
        
        for (unsigned int i = rank; i < npZ; i += size){
            for (unsigned int j = 0; j < npY; j++){
                for (unsigned int k = 0; k < npX; k++){
                    local_err = fabs(u0[i][j][k] - boundaries);
                    if (local_err > inErr)
                        local_continued = 1;
                    else
                        local_continued = 0;
                }
            }
        }
        
        MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_continued, &global_continued, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        continued = global_continued;
        iteration++;
    }
    int steps_local = steps;
    int steps_global;

    MPI_Reduce(&steps_local, &steps_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        fprintf(stdout, "%d\n", steps_global);
    }
}

int main (int ac, char **av){
  double ***u0;
  double ***u1;
  double deltaT = 0.01;
  double deltaH = 0.25f;
  double sizeX = 20.0f;
  double sizeY = 20.0f;
  double sizeZ = 20.0f;
  
  unsigned int npX = (unsigned int) (sizeX / deltaH); //Number of points in X axis
  unsigned int npY = (unsigned int) (sizeY / deltaH);
  unsigned int npZ = (unsigned int) (sizeZ / deltaH);

  int rank, size;
  MPI_Init(&ac, &av);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  struct timeval time_start;
  struct timeval time_end;

  gettimeofday(&time_start, NULL);
  
  
  //printf("p(%u, %u, %u)\n", npX, npY, npZ);
  //Allocing memory
  u0 = (double***) malloc (npZ * sizeof(double**));
  u1 = (double***) malloc (npZ * sizeof(double**));
  
  for (unsigned int i = 0; i < npZ; i++){
    u0[i] = (double**) malloc (npY * sizeof(double*));
    u1[i] = (double**) malloc (npY * sizeof(double*));
  }
  
  
  for (unsigned int i = 0; i < npZ; i++){
    for (unsigned int j = 0; j < npY; j++){
      double *aux0 = (double *) malloc (npX * sizeof(double));
      double *aux1 = (double *) malloc (npX * sizeof(double));
      //initial condition - zero in all points
      memset(aux0, 0x00, npX * sizeof(double));
      memset(aux1, 0x00, npX * sizeof(double));
      u0[i][j] = aux0;
      u1[i][j] = aux1;
    }
  }

  printf("Starting mdf_heat calculation\n");
  mdf_heat(u0, u1, npX, npY, npZ, deltaH, deltaT, 1e-15, 100.0f, rank, size);
  //mdf_print(u1,  npX, npY, npZ);

  gettimeofday(&time_end, NULL);

  double exec_time = (double) (time_end.tv_sec - time_start.tv_sec) +
                    (double) (time_end.tv_usec - time_start.tv_usec) / 1000000.0;

  printf("\nKernel executed in %lf seconds.\n", exec_time);
  
  //Free memory
  for (unsigned int i = 0; i < npZ; i++){
    for (unsigned int j = 0; j < npY; j++){
      free(u0[i][j]);
      free(u1[i][j]);
    }
  }
  
  for (unsigned int i = 0; i < npZ; i++){
    free(u0[i]);
    free(u1[i]);
  }
  
  free(u0);
  free(u1);
  
  MPI_Finalize();
  
  return EXIT_SUCCESS;
}
