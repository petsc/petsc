#ifndef lint
static char vcid[] = "$Id: sp1wd.c,v 1.12 1995/11/03 03:05:01 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "src/mat/impls/order/order.h"

/*
    MatOrder_1WD - Find the 1-way dissection ordering of a given matrix.
*/    
int MatOrder_1WD(int *Nrow, int *ia, int *ja, int * perm,int *permc )
{
  int i,   *mask, *xls, nblks, *xblk, *ls, nrow = *Nrow;

  mask = (int *)PetscMalloc( nrow * sizeof(int) );     CHKPTRQ(mask);
  xls  = (int *)PetscMalloc( (nrow+1) * sizeof(int) ); CHKPTRQ(xls);
  ls   = (int *)PetscMalloc( nrow * sizeof(int) );     CHKPTRQ(ls);
  xblk = (int *)PetscMalloc( nrow * sizeof(int) );     CHKPTRQ(xblk);
  gen1wd( &nrow, ia, ja, mask, &nblks, xblk, perm, xls, ls );
  PetscFree( mask ); PetscFree( xls ); PetscFree( ls ); PetscFree( xblk );
  for (i=0; i<nrow; i++) perm[i]--;
  PetscMemcpy(permc,perm,nrow*sizeof(int));
  return 0;
}

