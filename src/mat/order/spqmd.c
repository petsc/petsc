#ifndef lint
static char vcid[] = "$Id: spqmd.c,v 1.7 1995/05/29 20:29:21 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "order.h"

/*
    SpOrderQMD - Find the Quotient Minimum Degree ordering of a given matrix.
*/    
int SpOrderQMD( int nrow, int *ia, int *ja, int *perm )
{
  int i,   *deg, *marker, *rchset, *nbrhd, *qsize,
      *qlink, nofsub, *iperm;

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
  return 0;
}

