static char help[] = "Benchmarking HIP kernel launch time\n";
/*
  Running example on Crusher at OLCF:
  # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex1hip
  Average asynchronous HIP kernel launch time = 3.74 microseconds
  Average synchronous  HIP kernel launch time = 6.66 microseconds
*/
#include <petscsys.h>
#include <petscdevice.h>

__global__ void NullKernel(){}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,n=100000;
  hipError_t     cerr;
  PetscLogDouble tstart,tend,time;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* Launch a sequence of kernels asynchronously. Previous launched kernels do not need to be completed before launching a new one */
  ierr = PetscTime(&tstart);CHKERRQ(ierr);
  for (i=0; i<n; i++) {hipLaunchKernelGGL(NullKernel,dim3(1),dim3(1),0,NULL);}
  ierr = PetscTime(&tend);CHKERRQ(ierr);
  cerr = hipStreamSynchronize(NULL);CHKERRHIP(cerr); /* Sync after tend since we don't want to count kernel execution time */
  time = (tend-tstart)*1e6/n;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average asynchronous HIP kernel launch time = %.2f microseconds\n",time);CHKERRQ(ierr);

  /* Launch a sequence of kernels synchronously. Only launch a new kernel after the one before it has been completed */
  ierr = PetscTime(&tstart);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    hipLaunchKernelGGL(NullKernel,dim3(1),dim3(1),0,NULL);
    cerr = hipStreamSynchronize(NULL);CHKERRHIP(cerr);
  }
  ierr = PetscTime(&tend);CHKERRQ(ierr);
  time = (tend-tstart)*1e6/n;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average synchronous  HIP kernel launch time = %.2f microseconds\n",time);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: hip

  test:
    requires: hip
    args: -n 2
    output_file: output/empty.out
    filter: grep "DOES_NOT_EXIST"

TEST*/
