#ifndef lint
static char vcid[] = "$Id: spqmd.c,v 1.11 1995/11/01 23:19:07 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    MatOrder_QMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/    
int MatOrder_QMD( int *Nrow, int *ia, int *ja, int *perm,int *permc )
{
  int i,   *deg, *marker, *rchset, *nbrhd, *qsize, *qlink, nofsub, *iperm, nrow = *Nrow;

  iperm = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(iperm);
  deg    = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(deg);
  marker = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(marker);
  rchset = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(rchset);
  nbrhd  = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(nbrhd);
  qsize  = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(qsize);
  qlink  = (int *)PetscMalloc( nrow * sizeof(int) ); CHKPTRQ(qlink);
  /* WARNING - genqmd trashes ja */    
  genqmd( &nrow, ia, ja, perm, iperm, deg, marker, rchset, nbrhd, qsize,qlink, &nofsub );
  PetscFree( deg ); PetscFree( marker ); PetscFree( rchset ); 
  PetscFree( nbrhd ); PetscFree( qsize );
  PetscFree( qlink ); PetscFree(iperm);
  for (i=0; i<nrow; i++) perm[i]--;
  PetscMemcpy(permc,perm,nrow*sizeof(int));
  return 0;
}

