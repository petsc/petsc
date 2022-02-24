static char help[] = "Benchmarking hipPointerGetAttributes() time\n";
/*
  Running example on Crusher at OLCF:
    # run with 1 mpi rank (-n1), 32 CPUs (-c32), and map the process to CPU 0 and GPU 0
  $ srun -n1 -c32 --cpu-bind=map_cpu:0 --gpus-per-node=8 --gpu-bind=map_gpu:0 ./ex2hip
    Average hipPointerGetAttributes() time = 0.10 microseconds
*/
#include <petscsys.h>
#include <petscdevice.h>

int main(int argc,char **argv)
{
  PetscErrorCode               ierr;
  PetscInt                     i,n=2000;
  hipError_t                   cerr;
  PetscScalar                  **ptrs;
  PetscLogDouble               tstart,tend,time;
  hipPointerAttribute_t        attr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(PetscMalloc1(n,&ptrs));
  for (i=0; i<n; i++) {
    if (i%2) CHKERRQ(PetscMalloc1(i+16,&ptrs[i]));
    else CHKERRHIP(hipMalloc((void**)&ptrs[i],(i+16)*sizeof(PetscScalar)));
  }

  CHKERRQ(PetscTime(&tstart));
  for (i=0; i<n; i++) {
    cerr = hipPointerGetAttributes(&attr,ptrs[i]);
    if (cerr) hipGetLastError();
  }
  CHKERRQ(PetscTime(&tend));
  time = (tend-tstart)*1e6/n;

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Average hipPointerGetAttributes() time = %.2f microseconds\n",time));

  for (i=0; i<n; i++) {
    if (i%2) CHKERRQ(PetscFree(ptrs[i]));
    else CHKERRHIP(hipFree(ptrs[i]));
  }
  CHKERRQ(PetscFree(ptrs));
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
