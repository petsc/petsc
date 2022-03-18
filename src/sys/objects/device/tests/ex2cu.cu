static char help[] = "Benchmarking cudaPointerGetAttributes() time\n";
/*
  Running example on Summit at OLCF:
  # run with total 1 resource set (RS) (-n1), 1 RS per node (-r1), 1 MPI rank (-a1), 7 cores (-c7) and 1 GPU (-g1) per RS
  $ jsrun -n1 -a1 -c7 -g1 -r1  ./ex2cu
    Average cudaPointerGetAttributes() time = 0.29 microseconds
*/
#include <petscsys.h>
#include <petscdevice.h>

int main(int argc,char **argv)
{
  PetscErrorCode               ierr;
  PetscInt                     i,n=2000;
  cudaError_t                  cerr;
  PetscScalar                  **ptrs;
  PetscLogDouble               tstart,tend,time;
  struct cudaPointerAttributes attr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc1(n,&ptrs);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (i%2) {ierr = PetscMalloc1(i+16,&ptrs[i]);CHKERRQ(ierr);}
    else {cerr = cudaMalloc((void**)&ptrs[i],(i+16)*sizeof(PetscScalar));CHKERRCUDA(cerr);}
  }

  ierr = PetscTime(&tstart);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    cerr = cudaPointerGetAttributes(&attr,ptrs[i]);
    if (cerr) cudaGetLastError();
  }
  ierr = PetscTime(&tend);CHKERRQ(ierr);
  time = (tend-tstart)*1e6/n;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average cudaPointerGetAttributes() time = %.2f microseconds\n",time);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    if (i%2) {ierr = PetscFree(ptrs[i]);CHKERRQ(ierr);}
    else {cerr = cudaFree(ptrs[i]);CHKERRCUDA(cerr);}
  }
  ierr = PetscFree(ptrs);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
