#ifndef lint
static char vcid[] = "$Id: sp1wd.c,v 1.6 1995/04/27 20:15:53 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrder1WD - Find the 1-way dissection ordering of a given matrix.
*/    
int SpOrder1WD(int nrow, int *ia, int *ja, int * perm )
{
int i,   *mask, *xls, nblks, *xblk, *ls;

mask = (int *)MALLOC( nrow * sizeof(int) );     CHKPTR(mask);
xls  = (int *)MALLOC( (nrow+1) * sizeof(int) ); CHKPTR(xls);
ls   = (int *)MALLOC( nrow * sizeof(int) );     CHKPTR(ls);
xblk = (int *)MALLOC( nrow * sizeof(int) );     CHKPTR(xblk);
gen1wd( &nrow, ia, ja, mask, &nblks, xblk, perm, xls, ls );
FREE( mask ); FREE( xls ); FREE( ls ); FREE( xblk );
for (i=0; i<nrow; i++) perm[i]--;
return 0;
}

