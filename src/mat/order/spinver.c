#ifndef lint
static char vcid[] = "$Id: spinver.c,v 1.8 1996/12/17 16:51:43 balay Exp balay $";
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


