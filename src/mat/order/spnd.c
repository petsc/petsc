#ifndef lint
static char vcid[] = "$Id: spnd.c,v 1.8 1995/06/08 03:20:18 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    MatOrderND - Find the nested dissection ordering of a given matrix.
*/    
int MatOrderND( int *Nrow, int *ia, int *ja, int* perm,int *permc )
{
int i,  *mask, *xls, *ls, nrow = *Nrow;

mask = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(mask);
xls  = (int *)PETSCMALLOC( (nrow + 1) * sizeof(int) ); CHKPTRQ(xls);
ls   = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(ls);

gennd( &nrow, ia, ja, mask, perm, xls, ls );
PETSCFREE( mask ); PETSCFREE( xls ); PETSCFREE( ls );

for (i=0; i<nrow; i++) perm[i]--;
PETSCMEMCPY(permc,perm,nrow*sizeof(int));
return 0;
}


