/*$Id: spinver.c,v 1.13 1997/10/19 03:25:56 bsmith Exp bsmith $*/

#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
#undef __FUNC__  
#define __FUNC__ "MatInvertPermutation_Private"
void MatInvertPermutation_Private( int n, int *perm, int *iperm )
{
  int i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


