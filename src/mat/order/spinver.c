#ifndef lint
static char vcid[] = "$Id: spinver.c,v 1.1 1994/03/18 00:27:02 gropp Exp $";
#endif

#include "tools.h"
#include "sparse/spmat.h"

#if defined(FORTRANCAPS)
#define revrse_ REVRSE
#elif !defined(FORTRANUNDERSCORE)
#define revrse_ revrse
#endif


/*@
  SpInverse - Compute the inverse ordering of a permutation

  Input Parameters:
. n - size of permutation
. perm - permutation array

  Output Parameters:
. iperm - inverse of perm
 @*/
void SpInverse( n, perm, iperm )
int *perm, *iperm, n;
{
int i;

for (i=0; i<n; i++) {
    iperm[*perm++] = i;
    }
}

/* Interface to the sparsepak routines. This makes perm[k] -> perm[n-k] */
void revrse_( n, perm )
int *n, *perm;
{
int i, in = *n-1, m = *n/2, swap;

for (i=0; i<m; i++) {
    swap       = perm[i];
    perm[i]    = perm[in];
    perm[in--] = swap;
    }
}

