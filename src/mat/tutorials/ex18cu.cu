#include <cuda_runtime.h>
#include <petscdevice.h>
#include "ex18.h"

__global__ void FillValues(PetscInt n, PetscScalar *v)
{
  PetscInt     i = blockIdx.x * blockDim.x + threadIdx.x;
  PetscScalar  *s;
  if (i < n) {
    s = &v[3*3*i];
    for (PetscInt vi=0; vi<3; vi++) {
      for (PetscInt vj=0; vj<3; vj++) {
        s[vi*3+vj] = vi+2*vj;
      }
    }
  }
}

PetscErrorCode FillMatrixCUDACOO(FEStruct *fe,Mat A)
{
  PetscErrorCode             ierr;
  cudaError_t                cerr;
  PetscScalar                *v;

  PetscFunctionBeginUser;
  cerr = cudaMalloc((void**)&v,3*3*fe->Ne*sizeof(PetscScalar));CHKERRCUDA(cerr);
  FillValues<<<(fe->Ne+255)/256,256>>>(fe->Ne,v);
  ierr = MatSetValuesCOO(A,v,INSERT_VALUES);CHKERRQ(ierr);
  cerr = cudaFree(v);CHKERRCUDA(cerr);
  PetscFunctionReturn(0);
}
