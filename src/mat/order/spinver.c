#ifndef lint
static char vcid[] = "$Id: spinver.c,v 1.7 1995/11/09 22:29:39 bsmith Exp balay $";
#endif

#include "petsc.h"

/*
  MatInvertPermutation_Private - Compute the inverse ordering of a permutation
 */
#undef __FUNCTION__  
#define __FUNCTION__ "MatInvertPermutation_Private"
void MatInvertPermutation_Private( int n, int *perm, int *iperm )
{
  int i;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


