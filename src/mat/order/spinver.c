#ifndef lint
static char vcid[] = "$Id: spinver.c,v 1.5 1995/03/06 04:04:11 bsmith Exp bsmith $";
#endif

#include "petsc.h"

/*
  SpInverse - Compute the inverse ordering of a permutation
 */
void SpInverse( int n, int *perm, int *iperm )
{
  int i;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


