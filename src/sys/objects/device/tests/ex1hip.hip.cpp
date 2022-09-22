static char help[] = "Benchmarking HIP kernel launch time\n";
/*
  Running example on Crusher at OLCF:
  # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex1hip
  Average asynchronous HIP kernel launch time = 1.34 microseconds
  Average synchronous  HIP kernel launch time = 6.66 microseconds
*/
#include <petscsys.h>
#include <petscdevice_hip.h>

__global__ void NullKernel() { }

int main(int argc, char **argv)
{
  PetscInt       i, n = 100000;
  PetscLogDouble tstart, tend, time;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCallHIP(hipStreamSynchronize(NULL)); /* Initialize HIP runtime to get more accurate timing below */

  /* Launch a sequence of kernels asynchronously. Previous launched kernels do not need to be completed before launching a new one */
  PetscCall(PetscTime(&tstart));
  for (i = 0; i < n; i++) hipLaunchKernelGGL(NullKernel, dim3(1), dim3(1), 0, NULL);
  PetscCall(PetscTime(&tend));
  PetscCallHIP(hipStreamSynchronize(NULL)); /* Sync after tend since we don't want to count kernel execution time */
  time = (tend - tstart) * 1e6 / n;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Average asynchronous HIP kernel launch time = %.2f microseconds\n", time));

  /* Launch a sequence of kernels synchronously. Only launch a new kernel after the one before it has been completed */
  PetscCall(PetscTime(&tstart));
  for (i = 0; i < n; i++) {
    hipLaunchKernelGGL(NullKernel, dim3(1), dim3(1), 0, NULL);
    PetscCallHIP(hipStreamSynchronize(NULL));
  }
  PetscCall(PetscTime(&tend));
  time = (tend - tstart) * 1e6 / n;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Average synchronous  HIP kernel launch time = %.2f microseconds\n", time));

  PetscCall(PetscFinalize());
  return 0;
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
