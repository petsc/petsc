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
  PetscScalar *v;

  PetscFunctionBeginUser;
  CHKERRCUDA(cudaMalloc((void**)&v,3*3*fe->Ne*sizeof(PetscScalar)));
  FillValues<<<(fe->Ne+255)/256,256>>>(fe->Ne,v);
  CHKERRQ(MatSetValuesCOO(A,v,INSERT_VALUES));
  CHKERRCUDA(cudaFree(v));
  PetscFunctionReturn(0);
}
