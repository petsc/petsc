#ifndef lint
static char vcid[] = "$Id: sprcm.c,v 1.5 1995/03/27 22:57:56 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrderRCM - Find the Reverse Cuthill-McGee ordering of a given matrix.

    Input Paramter:
.    Matrix - matrix to find ordering for

    Output Parameters:
.    perm   - permutation vector (0-origin)
.    iperm  - inverse permutation vector.  If NULL, ignored.
*/    
int SpOrderRCM( int nrow, int *ia, int *ja,int *perm )
{
int i,   *mask, *xls;

mask = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(mask);
/*
  There appears to be a bug in genrcm_ it requires xls to be longer
than nrow, I have made it 2nrow to be safe.
*/
xls  = (int *)MALLOC( 2*nrow * sizeof(int) ); CHKPTR(xls);

genrcm( &nrow, ia, ja, perm, mask, xls );
FREE( mask );
FREE( xls );

for (i=0; i<nrow; i++) perm[i]--;

return 0;
}
