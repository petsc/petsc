#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: spinver.c,v 1.11 1997/07/09 20:54:49 balay Exp bsmith $";
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
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


