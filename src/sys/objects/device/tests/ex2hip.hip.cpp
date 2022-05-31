static char help[] = "Benchmarking hipPointerGetAttributes() time\n";
/*
  Running example on Crusher at OLCF:
    # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex2hip
    Average hipPointerGetAttributes() time = 0.24 microseconds
*/
#include <petscsys.h>
#include <petscdevice.h>

int main(int argc,char **argv)
{
  PetscInt                     i,n=4000;
  hipError_t                   cerr;
  PetscScalar                  **ptrs;
  PetscLogDouble               tstart,tend,time;
  hipPointerAttribute_t        attr;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCallHIP(hipStreamSynchronize(NULL)); /* Initialize HIP runtime to get more accurate timing below */

  PetscCall(PetscMalloc1(n,&ptrs));
  for (i=0; i<n; i++) {
    if (i%2) PetscCall(PetscMalloc1(i+16,&ptrs[i]));
    else PetscCallHIP(hipMalloc((void**)&ptrs[i],(i+16)*sizeof(PetscScalar)));
  }

  PetscCall(PetscTime(&tstart));
  for (i=0; i<n; i++) {
    cerr = hipPointerGetAttributes(&attr,ptrs[i]);
    if (cerr) cerr = hipGetLastError();
  }
  PetscCall(PetscTime(&tend));
  time = (tend-tstart)*1e6/n;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Average hipPointerGetAttributes() time = %.2f microseconds\n",time));

  for (i=0; i<n; i++) {
    if (i%2) PetscCall(PetscFree(ptrs[i]));
    else PetscCallHIP(hipFree(ptrs[i]));
  }
  PetscCall(PetscFree(ptrs));
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
