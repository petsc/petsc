#ifndef lint
static char vcid[] = "$Id: spnd.c,v 1.1 1994/03/18 00:27:02 gropp Exp $";
#endif

#include "petsc.h"

#if defined(FORTRANCAPS)
#define gennd_ GENND
#elif !defined(FORTRANUNDERSCORE)
#define gennd_ gennd
#endif 

/*
    SpOrderND - Find the nested dissection ordering of a given matrix.

    Input Paramter:
.    Matrix - matrix to find ordering for

    Output Parameters:
.    perm   - permutation vector (0-origin)
.    iperm  - inverse permutation vector.  If NULL, ignored.
*/    
int SpOrderND( int nrow, int *ia, int *ja, int* perm, int *iperm )
{
int i, nrow,  *mask, *xls, *ls;

mask = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(mask);
xls  = (int *)MALLOC( (nrow + 1) * sizeof(int) ); CHKPTR(xls);
ls   = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(ls);

gennd_( &nrow, ia, ja, mask, perm, xls, ls );
FREE( ia );  FREE( ja );
FREE( mask ); FREE( xls ); FREE( ls );

for (i=0; i<nrow; i++) perm[i]--;
if (iperm) SpInverse( nrow, perm, iperm );
return 0;
}


