
#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
#undef __FUNCT__  
#define __FUNCT__ "MatInvertPermutation_Private"
void MatInvertPermutation_Private(int n,int *perm,int *iperm)
{
  int i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


