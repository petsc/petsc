#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: spinver.c,v 1.10 1997/02/22 02:25:40 bsmith Exp balay $";
#endif

#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
#undef __FUNC__  
#define __FUNC__ "MatInvertPermutation_Private" /* ADIC Ignore */
void MatInvertPermutation_Private( int n, int *perm, int *iperm )
{
  int i;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


