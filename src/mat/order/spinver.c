/*$Id: spinver.c,v 1.17 2000/04/12 04:24:20 bsmith Exp bsmith $*/

#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
#undef __FUNC__  
#define __FUNC__ "MatInvertPermutation_Private"
void MatInvertPermutation_Private(int n,int *perm,int *iperm)
{
  int i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


