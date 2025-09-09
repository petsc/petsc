/*
  An adaption of the Stream benchmark for MPI
  Original code developed by John D. McCalpin
*/
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <petscsys.h>

#define NTIMESINNER 1
//#define N 2*4*20000000
#define N 80000000
//#define N 1200000
//#define N 120000
#define NTIMES 50
#define OFFSET 0

static double a[N+OFFSET],b[N+OFFSET],c[N+OFFSET];
static double mintime = FLT_MAX;
static double bytes = 3 * sizeof(double) * N;

int main(int argc,char **args)
{
  const double scalar = 3.0;
  double       times[NTIMES],rate;
  PetscMPIInt  rank,size,resultlen;
  char         hostname[MPI_MAX_PROCESSOR_NAME] = {0};
  PetscInt     n = PETSC_DECIDE,NN;

  PetscCall(PetscInitialize(&argc,&args,NULL,NULL));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD,&size));

  PetscCallMPI(MPI_Get_processor_name(hostname,&resultlen));(void)resultlen;
  if (rank) PetscCallMPI(MPI_Send(hostname,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,0,MPI_COMM_WORLD));
  else {
    // printf("Rank %d host %s\n",0,hostname);
    for (int j = 1; j < size; ++j) {
      PetscCallMPI(MPI_Recv(hostname,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,j,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE));
      // printf("Rank %d host %s\n",j,hostname);
    }
  }

  NN = N;
  PetscCall(PetscSplitOwnership(MPI_COMM_WORLD,&n,&NN));
  for (int j = 0; j < n; ++j) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 3.0;
  }

  /*   --- MAIN LOOP --- repeat test cases NTIMES times --- */
  for (int k = 0; k < NTIMES; ++k) {
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    // Do not include barrier in the timed region
    times[k] = MPI_Wtime();
    for (int l=0; l<NTIMESINNER; l++){
      for (register int j = 0; j < n; j++) a[j] = b[j]+scalar*c[j];
      if (size == 65) printf("never printed %g\n",a[11]);
    }
    //   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    times[k] = MPI_Wtime() - times[k];
  }
  // use maximum time over all MPI processes
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE,times,NTIMES,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD));
  for (int k = 1; k < NTIMES; ++k) {   /* note -- skip first iteration */
    mintime = PetscMin(mintime,times[k]);
  }
  rate = 1.0E-06 * bytes*NTIMESINNER/mintime;

  if (rank == 0) {
    FILE *fd;

    if (size != 1) {
      double prate;

      fd = fopen("flops","r");
      fscanf(fd,"%lg",&prate);
      fclose(fd);
      printf("%d %11.4f   Rate (MB/s) %g \n",size,rate,rate/prate);
    } else {
      fd = fopen("flops","w");
      fprintf(fd,"%g\n",rate);
      fclose(fd);
      printf("%d %11.4f   Rate (MB/s) 1\n",size,rate);
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}
