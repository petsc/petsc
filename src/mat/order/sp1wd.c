#ifndef lint
static char vcid[] = "$Id: sp1wd.c,v 1.2 1994/11/25 23:04:56 bsmith Exp bsmith $";
#endif

#include "petsc.h"

#if defined(FORTRANCAPS)
#define gen1wd_ GEN1WD
#elif !defined (FORTRANUNDERSCORE)
#define gen1wd_ gen1wd
#endif

#if defined(__cplusplus)
extern "C" {
  void gen1wd_(int*,int*,int*,int*,int*,int*,int*,int*,int*);
};
#endif 

/*
    SpOrder1WD - Find the 1-way dissection ordering of a given matrix.

    Input Paramter:
.    Matrix - matrix to find ordering for

    Output Parameters:
.    perm   - permutation vector (0-origin)
.    iperm  - inverse permutation vector.  If NULL, ignored.
*/    
int SpOrder1WD(int nrow, int *ia, int *ja, int * perm )
{
int i,   *mask, *xls, nblks, *xblk, *ls;

mask = (int *)MALLOC( nrow * sizeof(int) );     CHKPTR(mask);
xls  = (int *)MALLOC( (nrow+1) * sizeof(int) ); CHKPTR(xls);
ls   = (int *)MALLOC( nrow * sizeof(int) );     CHKPTR(ls);
xblk = (int *)MALLOC( nrow * sizeof(int) );     CHKPTR(xblk);
gen1wd_( &nrow, ia, ja, mask, &nblks, xblk, perm, xls, ls );
FREE( mask ); FREE( xls ); FREE( ls ); FREE( xblk );

for (i=0; i<nrow; i++) perm[i]--;

return 0;
}

