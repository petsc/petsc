#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: spinver.c,v 1.12 1997/08/22 15:14:10 bsmith Exp bsmith $";
#endif

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


