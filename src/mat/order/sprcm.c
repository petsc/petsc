#ifndef lint
static char vcid[] = "$Id: sprcm.c,v 1.1 1994/03/18 00:27:06 gropp Exp $";
#endif

#include "petsc.h"

#if defined(FORTRANCAPS)
#define genrcm_ GENRCM
#elif !defined(FORTRANUNDERSCORE)
#define genrcm_ genrcm
#endif

/*
    SpOrderRCM - Find the Reverse Cuthill-McGee ordering of a given matrix.

    Input Paramter:
.    Matrix - matrix to find ordering for

    Output Parameters:
.    perm   - permutation vector (0-origin)
.    iperm  - inverse permutation vector.  If NULL, ignored.
*/    
int SpOrderRCM( int nrow, int *ia, int *ja,int *perm, int *iperm )
{
int i, nrow,  *mask, *xls;

mask = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(mask);
xls  = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(xls);

genrcm_( &nrow, ia, ja, perm, mask, xls );
FREE( mask ); FREE( xls );
FREE( ia );  FREE( ja );

for (i=0; i<nrow; i++) perm[i]--;
if (iperm) SpInverse( nrow, perm, iperm );
return 0;
}
