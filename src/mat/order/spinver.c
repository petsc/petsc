#ifndef lint
static char vcid[] = "$Id: spinver.c,v 1.9 1997/01/06 20:25:07 balay Exp bsmith $";
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


