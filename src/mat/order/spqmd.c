#ifndef lint
static char vcid[] = "$Id: spqmd.c,v 1.1 1994/03/18 00:27:05 gropp Exp $";
#endif

#include "petsc.h"

#if defined(FORTRANCAPS)
#define genqmd_ GENQMD 
#elif !defined(FORTRANUNDERSCORE)
#define genqmd_ genqmd
#endif

/*
    SpOrderQMD - Find the Quotient Minimum Degree ordering of a given matrix.

    Input Paramter:
.    Matrix - matrix to find ordering for

    Output Parameters:
.    perm   - permutation vector (0-origin)
.    iperm  - inverse permutation vector.  If NULL, ignored.
*/    
int SpOrderQMD( int nrow, int *ia, int *ja, int *perm, int *iperm )
{
int i, nrow,  *deg, *marker, *rchset, *nbrhd, *qsize,
    *qlink, nofsub, hadiperm;

if (iperm) hadiperm = 1;
else {
    hadiperm = 0;
    iperm = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(iperm);
    }
deg    = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(deg);
marker = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(marker);
rchset = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(rchset);
nbrhd  = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(nbrhd);
qsize  = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(qsize);
qlink  = (int *)MALLOC( nrow * sizeof(int) ); CHKPTR(qlink);
/* WARNING - genqmd trashes ja */    
genqmd_( &nrow, ia, ja, perm, iperm, deg, marker, rchset, nbrhd, qsize,
	 qlink, &nofsub );
FREE( deg ); FREE( marker ); FREE( rchset ); FREE( nbrhd ); FREE( qsize );
FREE( qlink );
FREE( ia );  FREE( ja );
if (!hadiperm) {
    FREE(iperm);
    }
else
    for (i=0; i<nrow; i++) iperm[i]--;

for (i=0; i<nrow; i++) perm[i]--;
return 0;
}

