
#ifndef lint
static char vcid[] = "$Id: baijfact.c,v 1.6 1996/02/20 18:52:30 curfman Exp bsmith $";
#endif

/*  
              This file creating by running f2c 
            linpack. this version dated 08/14/78 
      cleve moler, university of new mexico, argonne national lab.

      Computes the inverse of a matrix given its factors and pivots
    calculated by Linpack_DGEFA().
*/

#include "petsc.h"

int Linpack_DGEDI(Scalar *a,int n,int *ipvt,Scalar *work)
{
    int     a_offset, i__2,kb, kp1, nm1,i, j, k, l, ll;
    Scalar  t, *aa,*ax,*ay,tmp;

    --work;
    --ipvt;
    a_offset = n + 1;
    a       -= a_offset;

   /*     compute inverse(u) */

    for (k = 1; k <= n; ++k) {
	a[k + k * n] = 1. / a[k + k * n];
	t = -a[k + k * n];
	i__2 = k - 1;
	/* dscal_(&i__2, &t, &a[k * n + 1], &c__1); */
        aa = &a[k * n + 1]; 
        for ( ll=0; ll<i__2; ll++ ) aa[ll] *= t;
	kp1 = k + 1;
	if (n < kp1) continue;
        ax = aa;
        for (j = kp1; j <= n; ++j) {
	    t = a[k + j * n];
	    a[k + j * n] = 0.;
	    /* daxpy_(&k, &t, &a[k * n + 1], &c__1, &a[j * n + 1], &c__1);*/
            ay = &a[j * n + 1];
            for ( ll=0; ll<k; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
    }

   /*    form inverse(u)*inverse(l) */

    nm1 = n - 1;
    if (nm1 < 1) {
	return 0;
    }
    for (kb = 1; kb <= nm1; ++kb) {
	k   = n - kb;
	kp1 = k + 1;
        aa  = a + k * n;
	for (i = kp1; i <= n; ++i) {
	    work[i] = aa[i];
	    aa[i]   = 0.;
	}
	for (j = kp1; j <= n; ++j) {
	    t = work[j];
	    /* daxpy_(n, &t, &a[j * n + 1], &c__1, &a[k * n + 1], &c__1);*/
            ax = &a[j * n + 1];
            ay = &a[k * n + 1];
            for ( ll=0; ll<n; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
	l = ipvt[k];
	if (l != k) {
	    /* dswap_(n, &a[k * n + 1], &c__1, &a[l * n + 1], &c__1); */
            ax = &a[k * n + 1]; 
            ay = &a[l * n + 1];
            for ( ll=0; ll<n; ll++ ) {
              tmp    = ax[ll];
              ax[ll] = ay[ll];
              ay[ll] = tmp;
            }
	}
    }
    return 0;
}

