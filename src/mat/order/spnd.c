#ifndef lint
static char vcid[] = "$Id: spnd.c,v 1.10 1995/09/30 19:29:19 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    MatOrderND - Find the nested dissection ordering of a given matrix.
*/    
int MatOrderND( int *Nrow, int *ia, int *ja, int* perm,int *permc )
{
int i,  *mask, *xls, *ls, nrow = *Nrow;

mask = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(mask);
xls  = (int *)PetscMalloc( (nrow + 1) * sizeof(int) ); CHKPTRQ(xls);
ls   = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(ls);

gennd( &nrow, ia, ja, mask, perm, xls, ls );
PetscFree( mask ); PetscFree( xls ); PetscFree( ls );

for (i=0; i<nrow; i++) perm[i]--;
PetscMemcpy(permc,perm,nrow*sizeof(int));
return 0;
}


