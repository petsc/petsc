#ifndef lint
static char vcid[] = "$Id: spnd.c,v 1.4 1995/03/10 00:02:02 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrderND - Find the nested dissection ordering of a given matrix.

    Input Paramter:
.    Matrix - matrix to find ordering for

    Output Parameters:
.    perm   - permutation vector (0-origin)
.    iperm  - inverse permutation vector.  If NULL, ignored.
*/    
int SpOrderND( int nrow, int *ia, int *ja, int* perm )
{
int i,  *mask, *xls, *ls;

mask = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(mask);
xls  = (int *)MALLOC( (nrow + 1) * sizeof(int) ); CHKPTR(xls);
ls   = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(ls);

gennd_( &nrow, ia, ja, mask, perm, xls, ls );
FREE( mask ); FREE( xls ); FREE( ls );

for (i=0; i<nrow; i++) perm[i]--;
return 0;
}


