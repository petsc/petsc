/*$Id: spinver.c,v 1.15 2000/01/11 21:01:25 bsmith Exp bsmith $*/

#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"MatInvertPermutation_Private"
void MatInvertPermutation_Private(int n,int *perm,int *iperm)
{
  int i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


