#ifndef lint
static char vcid[] = "$Id: sp1wd.c,v 1.9 1995/08/15 20:28:10 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    MatOrder1WD - Find the 1-way dissection ordering of a given matrix.
*/    
int MatOrder1WD(int *Nrow, int *ia, int *ja, int * perm,int *permc )
{
int i,   *mask, *xls, nblks, *xblk, *ls, nrow = *Nrow;

mask = (int *)PETSCMALLOC( nrow * sizeof(int) );     CHKPTRQ(mask);
xls  = (int *)PETSCMALLOC( (nrow+1) * sizeof(int) ); CHKPTRQ(xls);
ls   = (int *)PETSCMALLOC( nrow * sizeof(int) );     CHKPTRQ(ls);
xblk = (int *)PETSCMALLOC( nrow * sizeof(int) );     CHKPTRQ(xblk);
gen1wd( &nrow, ia, ja, mask, &nblks, xblk, perm, xls, ls );
PETSCFREE( mask ); PETSCFREE( xls ); PETSCFREE( ls ); PETSCFREE( xblk );
for (i=0; i<nrow; i++) perm[i]--;
PetscMemcpy(permc,perm,nrow*sizeof(int));
return 0;
}

