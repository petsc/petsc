#ifndef lint
static char vcid[] = "$Id: sp1wd.c,v 1.7 1995/05/29 20:29:21 bsmith Exp curfman $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrder1WD - Find the 1-way dissection ordering of a given matrix.
*/    
int SpOrder1WD(int nrow, int *ia, int *ja, int * perm )
{
int i,   *mask, *xls, nblks, *xblk, *ls;

mask = (int *)PETSCMALLOC( nrow * sizeof(int) );     CHKPTRQ(mask);
xls  = (int *)PETSCMALLOC( (nrow+1) * sizeof(int) ); CHKPTRQ(xls);
ls   = (int *)PETSCMALLOC( nrow * sizeof(int) );     CHKPTRQ(ls);
xblk = (int *)PETSCMALLOC( nrow * sizeof(int) );     CHKPTRQ(xblk);
gen1wd( &nrow, ia, ja, mask, &nblks, xblk, perm, xls, ls );
PETSCFREE( mask ); PETSCFREE( xls ); PETSCFREE( ls ); PETSCFREE( xblk );
for (i=0; i<nrow; i++) perm[i]--;
return 0;
}

