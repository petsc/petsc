#ifndef lint
static char vcid[] = "$Id: spqmd.c,v 1.8 1995/06/08 03:20:44 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    MatOrderQMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/    
int MatOrderQMD( int *Nrow, int *ia, int *ja, int *perm,int *permc )
{
  int i,   *deg, *marker, *rchset, *nbrhd, *qsize,
      *qlink, nofsub, *iperm, nrow = *Nrow;

  iperm = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(iperm);
  deg    = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(deg);
  marker = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(marker);
  rchset = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(rchset);
  nbrhd  = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(nbrhd);
  qsize  = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(qsize);
  qlink  = (int *)PETSCMALLOC( nrow * sizeof(int) ); CHKPTRQ(qlink);
  /* WARNING - genqmd trashes ja */    
  genqmd( &nrow, ia, ja, perm, iperm, deg, marker, rchset, nbrhd, qsize,
	 qlink, &nofsub );
  PETSCFREE( deg ); PETSCFREE( marker ); PETSCFREE( rchset ); 
  PETSCFREE( nbrhd ); PETSCFREE( qsize );
  PETSCFREE( qlink ); PETSCFREE(iperm);
  for (i=0; i<nrow; i++) perm[i]--;
  PETSCMEMCPY(permc,perm,nrow*sizeof(int));
  return 0;
}

