#ifndef lint
static char vcid[] = "$Id: spinver.c,v 1.6 1995/05/29 20:29:21 bsmith Exp bsmith $";
#endif

#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
void MatInvertPermutation_Private( int n, int *perm, int *iperm )
{
  int i;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


