
#ifndef lint
static char vcid[] = "$Id: dgedi.c,v 1.2 1996/03/04 05:16:18 bsmith Exp bsmith $";
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
    int     i__2,kb, kp1, nm1,i, j, k, l, ll,kn,knp1,jn;
    Scalar  t, *aa,*ax,*ay,tmp;

    --work;
    --ipvt;
    a       -= n + 1;

   /*     compute inverse(u) */

    for (k = 1; k <= n; ++k) {
        kn           = k*n;
        knp1         = kn + k;
	a[knp1]      = 1.0 / a[knp1];
	t            = -a[knp1];
	i__2         = k - 1;
        aa           = &a[1 + kn]; 
        for ( ll=0; ll<i__2; ll++ ) aa[ll] *= t;
	kp1 = k + 1;
	if (n < kp1) continue;
        ax = aa;
        for (j = kp1; j <= n; ++j) {
            jn = j*n;
	    t = a[k + jn];
	    a[k + jn] = 0.;
            ay = &a[1 + jn];
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
        kn  = k*n;
	kp1 = k + 1;
        aa  = a + kn;
	for (i = kp1; i <= n; ++i) {
	    work[i] = aa[i];
	    aa[i]   = 0.;
	}
	for (j = kp1; j <= n; ++j) {
	    t = work[j];
            ax = &a[j * n + 1];
            ay = &a[kn + 1];
            for ( ll=0; ll<n; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
	l = ipvt[k];
	if (l != k) {
            ax = &a[kn + 1]; 
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

