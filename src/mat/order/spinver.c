

#include "petsc.h"

#if defined(FORTRANCAPS)
#define revrse_ REVRSE
#elif !defined(FORTRANUNDERSCORE)
#define revrse_ revrse
#endif


/*
  SpInverse - Compute the inverse ordering of a permutation

  Input Parameters:
. n - size of permutation
. perm - permutation array

  Output Parameters:
. iperm - inverse of perm
 */
void SpInverse( int n, int *perm, int *iperm )
{
  int i;
  for (i=0; i<n; i++) {
    iperm[*perm++] = i;
  }
}


