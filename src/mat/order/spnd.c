#ifndef lint
static char vcid[] = "$Id: spnd.c,v 1.6 1995/04/27 20:15:53 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrderND - Find the nested dissection ordering of a given matrix.
*/    
int SpOrderND( int nrow, int *ia, int *ja, int* perm )
{
int i,  *mask, *xls, *ls;

mask = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(mask);
xls  = (int *)MALLOC( (nrow + 1) * sizeof(int) ); CHKPTR(xls);
ls   = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(ls);

gennd( &nrow, ia, ja, mask, perm, xls, ls );
FREE( mask ); FREE( xls ); FREE( ls );

for (i=0; i<nrow; i++) perm[i]--;
return 0;
}


