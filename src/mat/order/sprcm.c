#ifndef lint
static char vcid[] = "$Id: sprcm.c,v 1.7 1995/05/29 20:29:21 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrderRCM - Find the Reverse Cuthill-McGee ordering of a given matrix.
*/    
int SpOrderRCM( int nrow, int *ia, int *ja,int *perm )
{
int i,   *mask, *xls;
mask = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(mask);
/*
  There appears to be a bug in genrcm_ it requires xls to be longer
than nrow, I have made it 2nrow to be safe.
*/
xls  = (int *)PETSCMALLOC( 2*nrow * sizeof(int) ); CHKPTRQ(xls);
genrcm( &nrow, ia, ja, perm, mask, xls );
PETSCFREE( mask );
PETSCFREE( xls );
for (i=0; i<nrow; i++) perm[i]--;
return 0;
}
