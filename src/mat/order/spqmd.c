#ifndef lint
static char vcid[] = "$Id: spqmd.c,v 1.3 1995/01/13 04:36:17 bsmith Exp bsmith $";
#endif

#include "petsc.h"

#if defined(FORTRANCAPS)
#define genqmd_ GENQMD 
#elif !defined(FORTRANUNDERSCORE)
#define genqmd_ genqmd
#endif

#if defined(__cplusplus)
extern "C" {
#endif
extern void genqmd_(int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*);
#if defined(__cplusplus)
};
#endif 

/*
    SpOrderQMD - Find the Quotient Minimum Degree ordering of a given matrix.

    Input Paramter:
.    Matrix - matrix to find ordering for

    Output Parameters:
.    perm   - permutation vector (0-origin)
.    iperm  - inverse permutation vector.  If NULL, ignored.
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
genqmd_( &nrow, ia, ja, perm, iperm, deg, marker, rchset, nbrhd, qsize,
	 qlink, &nofsub );
FREE( deg ); FREE( marker ); FREE( rchset ); FREE( nbrhd ); FREE( qsize );
FREE( qlink );

    FREE(iperm);
for (i=0; i<nrow; i++) perm[i]--;
return 0;
}

