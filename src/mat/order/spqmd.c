#ifndef lint
static char vcid[] = "$Id: spqmd.c,v 1.6 1995/04/27 20:15:53 bsmith Exp bsmith $";
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

  iperm = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(iperm);
  deg    = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(deg);
  marker = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(marker);
  rchset = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(rchset);
  nbrhd  = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(nbrhd);
  qsize  = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(qsize);
  qlink  = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(qlink);
  /* WARNING - genqmd trashes ja */    
  genqmd( &nrow, ia, ja, perm, iperm, deg, marker, rchset, nbrhd, qsize,
	 qlink, &nofsub );
  FREE( deg ); FREE( marker ); FREE( rchset ); FREE( nbrhd ); FREE( qsize );
  FREE( qlink ); FREE(iperm);
  for (i=0; i<nrow; i++) perm[i]--;
  return 0;
}

