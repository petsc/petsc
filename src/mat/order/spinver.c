#define PETSCMAT_DLL

#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
#undef __FUNCT__  
#define __FUNCT__ "MatInvertPermutation_Private"
void MatInvertPermutation_Private(PetscInt n,PetscInt *perm,PetscInt *iperm)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


