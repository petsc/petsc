#ifndef lint
static char vcid[] = "$Id: sprcm.c,v 1.10 1995/09/30 19:29:19 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    MatOrderRCM - Find the Reverse Cuthill-McGee ordering of a given matrix.
*/    
int MatOrderRCM( int *Nrow, int *ia, int *ja,int *perm,int *permc )
{
  int i,   *mask, *xls, nrow = *Nrow;
  mask = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(mask);
/*
  There appears to be a bug in genrcm_ it requires xls to be longer
than nrow, I have made it 2nrow to be safe.
*/
  xls  = (int *)PetscMalloc( 2*nrow * sizeof(int) ); CHKPTRQ(xls);
  genrcm( &nrow, ia, ja, perm, mask, xls );
  PetscFree( mask );
  PetscFree( xls );
  for (i=0; i<nrow; i++) perm[i]--;
  PetscMemcpy(permc,perm,nrow*sizeof(int));
  return 0;
}
