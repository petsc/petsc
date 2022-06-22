static char help[] = "Benchmarking CUDA kernel launch time\n";
/*
  Running example on Summit at OLCF:
  # run with total 1 resource set (RS) (-n1), 1 RS per node (-r1), 1 MPI rank (-a1), 7 cores (-c7) and 1 GPU (-g1) per RS
  $ jsrun -n1 -a1 -c7 -g1 -r1  ./ex1cu
  Average asynchronous CUDA kernel launch time = 4.86 microseconds
  Average synchronous  CUDA kernel launch time = 12.83 microseconds
*/
#include <petscsys.h>
#include <petscdevice.h>

__global__ void NullKernel(){}

int main(int argc,char **argv)
{
  PetscInt       i,n=100000;
  PetscLogDouble tstart,tend,time;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCallCUDA(cudaStreamSynchronize(NULL)); /* Initialize CUDA runtime to get more accurate timing below */

  /* Launch a sequence of kernels asynchronously. Previous launched kernels do not need to be completed before launching a new one */
  PetscCall(PetscTime(&tstart));
  for (i=0; i<n; i++) {NullKernel<<<1,1,0,NULL>>>();}
  PetscCall(PetscTime(&tend));
  PetscCallCUDA(cudaStreamSynchronize(NULL)); /* Sync after tend since we don't want to count kernel execution time */
  time = (tend-tstart)*1e6/n;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Average asynchronous CUDA kernel launch time = %.2f microseconds\n",time));

  /* Launch a sequence of kernels synchronously. Only launch a new kernel after the one before it has been completed */
  PetscCall(PetscTime(&tstart));
  for (i=0; i<n; i++) {
    NullKernel<<<1,1,0,NULL>>>();
    PetscCallCUDA(cudaStreamSynchronize(NULL));
  }
  PetscCall(PetscTime(&tend));
  time = (tend-tstart)*1e6/n;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Average synchronous  CUDA kernel launch time = %.2f microseconds\n",time));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: cuda

  test:
    requires: cuda
    args: -n 2
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

TEST*/
