#ifndef lint
static char vcid[] = "$Id: spnd.c,v 1.7 1995/05/29 20:29:21 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrderND - Find the nested dissection ordering of a given matrix.
*/    
int SpOrderND( int nrow, int *ia, int *ja, int* perm )
{
int i,  *mask, *xls, *ls;

mask = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(mask);
xls  = (int *)PETSCMALLOC( (nrow + 1) * sizeof(int) ); CHKPTRQ(xls);
ls   = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(ls);

gennd( &nrow, ia, ja, mask, perm, xls, ls );
PETSCFREE( mask ); PETSCFREE( xls ); PETSCFREE( ls );

for (i=0; i<nrow; i++) perm[i]--;
return 0;
}


