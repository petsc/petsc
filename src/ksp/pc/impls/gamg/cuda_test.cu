#include <petscconf.h>
#include <petscsys.h>
#include <petsctime.h>
#include <petscdevice.h>
#include <../src/ksp/pc/impls/gamg/cuda_test.h>

__global__ void static RamdonSet(PetscInt n, PetscReal *ramdom, PetscInt *permute, PetscBool *bIndexSet)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  PetscInt iSwapIndex;
  if(i < n){
      permute[i] = i;
    iSwapIndex = (PetscInt) (ramdom[i]*n)%n;
    if (!bIndexSet[iSwapIndex] && iSwapIndex != i) {
    PetscInt iTemp = permute[iSwapIndex];
    permute[iSwapIndex] = permute[i];
    permute[i] = iTemp;
    bIndexSet[iSwapIndex] = PETSC_TRUE;
    }
  }
}
__global__ void static getCOOValue(PetscInt      Istart, PetscInt Iend, PetscReal vfilter, const PetscInt *ia, const PetscInt *ja, const PetscScalar *aa, PetscInt *coo_i, PetscInt *coo_j, PetscScalar *coo_values)
{
  int Ii = blockIdx.x*blockDim.x + threadIdx.x;
  PetscInt ncols;
  const PetscScalar *vals;
  const PetscInt *idx;
  if(Ii < Iend && Ii > Istart){
    ncols = ia[Ii+1] - ia[Ii];
    idx  = ja + ja[Ii];
    vals = (PetscScalar*)(aa + ia[Ii]);
    for (int jj=0; jj<ncols; jj++) {
      PetscScalar sv = PetscAbs(PetscRealPart(vals[jj]));
      if (PetscRealPart(sv) > vfilter) {
        coo_values[ia[Ii]+jj] = sv;
        coo_i[ia[Ii]+jj] = Ii;
        coo_j[ia[Ii]+jj] = idx[jj];
      }
    }
  }
}

PetscErrorCode getCOOValueC(PetscInt      Istart, PetscInt Iend, PetscReal vfilter, const PetscInt *ia, const PetscInt *ja, const PetscScalar *aa, PetscInt *coo_i, PetscInt *coo_j, PetscScalar *coo_values)
{
  getCOOValue<<<(Iend+255)/256,256,0>>>(Istart, Iend, vfilter, ia, ja, aa, coo_i, coo_j, coo_values);
  PetscFunctionReturn(0);
}

PetscErrorCode RamdonSetC(PetscInt n, PetscReal *ramdom, PetscInt *permute, PetscBool *bIndexSet){
  RamdonSet<<<(n+255)/256,256,0>>>(n, ramdom, permute, bIndexSet);
  PetscFunctionReturn(0);
} 


