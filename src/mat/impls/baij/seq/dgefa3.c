
/*
       This routine was converted by f2c from Linpack source
             linpack. this version dated 08/14/78 
      cleve moler, university of new mexico, argonne national lab.
*/
#include "petsc.h"

int Linpack_DGEFA(Scalar *a, int n, int *ipvt)
{
    int     i__2, i__3, kp1, nm1, j, k, l,ll;
    Scalar  t,*aa,*ax,*ay;
    double  tmp,max;

/*     gaussian elimination with partial pivoting */

    /* Parameter adjustments */
    --ipvt;
    a       -= n + 1;

    /* Function Body */
    nm1 = n - 1;
    if (nm1 < 1) {
	goto L70;
    }
    for (k = 1; k <= nm1; ++k) {
	kp1 = k + 1;

/*        find l = pivot index */

	i__2 = n - k + 1;
        aa = &a[k + k * n];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for ( ll=1; ll<i__2; ll++ ) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l += k - 1;
	ipvt[k] = l;

	if (a[l + k * n] == 0.) {
	  SETERRQ(k,"Linpack_DGEFA:Zero pivot");
	}

/*           interchange if necessary */

	if (l != k) {
	  t = a[l + k * n];
	  a[l + k * n] = a[k + k * n];
	  a[k + k * n] = t;
        }

/*           compute multipliers */

	t = -1. / a[k + k * n];
	i__2 = n - k;
        aa = &a[k + 1 + k * n]; 
        for ( ll=0; ll<i__2; ll++ ) {
          aa[ll] *= t;
        }

/*           row elimination with column indexing */

	ax = &a[k+1+k*n]; 
        for (j = kp1; j <= n; ++j) {
	    t = a[l + j * n];
	    if (l != k) {
	      a[l + j * n] = a[k + j * n];
	      a[k + j * n] = t;
            }

	    i__3 = n - k;
            ay = &a[k+1+j*n];
            for ( ll=0; ll<i__3; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
    }
L70:
    ipvt[n] = n;
    if (a[n + n * n] == 0.) {
	SETERRQ(n,"Linpack_DGEFA:Zero pivot,final row");
    }
    return 0;
} 

/*
     Version for 3 by 3 matrices.

   THIS CODE HAS NOT BEEN TESTED YET!

*/
int Linpack_DGEFA_3(Scalar *a, int *ipvt)
{
    int     i__2, i__3, kp1, j, k, l,ll;
    Scalar  t,*aa,*ax,*ay;
    double  tmp,max;

/*     gaussian elimination with partial pivoting */

    /* Parameter adjustments */
    --ipvt;
    a       -= 4;

    /* Function Body */
    for (k = 1; k <= 2; ++k) {
	kp1 = k + 1;

/*        find l = pivot index */

	i__2 = 4 - k;
        aa = &a[4*k];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for ( ll=1; ll<i__2; ll++ ) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l += k - 1;
	ipvt[k] = l;

	if (a[l + 3*k] == 0.) {
	  SETERRQ(k,"Linpack_DGEFA:Zero pivot");
	}

/*           interchange if necessary */

	if (l != k) {
	  t = a[l + 3*k];
	  a[l + 3*k] = a[4*k];
	  a[4*k] = t;
        }

/*           compute multipliers */

	t = -1. / a[4*k];
	i__2 = 3 - k;
        aa = &a[1 + 4*k]; 
        for ( ll=0; ll<i__2; ll++ ) {
          aa[ll] *= t;
        }

/*           row elimination with column indexing */

	ax = &a[4*k+1]; 
        for (j = kp1; j <= 3; ++j) {
	    t = a[l + 3*j];
	    if (l != k) {
	      a[l + 3*j] = a[k + 3*j];
	      a[k + 3*j] = t;
            }

	    i__3 = 3 - k;
            ay = &a[k+1+3*j];
            for ( ll=0; ll<i__3; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
    }
    ipvt[3] = 3;
    if (a[12] == 0.) {
	SETERRQ(3,"Linpack_DGEFA:Zero pivot,final row");
    }
    return 0;
} 


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
    int     i__2,kb, kp1, nm1,i, j, k, l, ll;
    Scalar  t, *aa,*ax,*ay,tmp;

    --work;
    --ipvt;
    a       -= n + 1;

   /*     compute inverse(u) */

    for (k = 1; k <= n; ++k) {
	a[k + k * n] = 1. / a[k + k * n];
	t = -a[k + k * n];
	i__2 = k - 1;
        aa = &a[k * n + 1]; 
        for ( ll=0; ll<i__2; ll++ ) aa[ll] *= t;
	kp1 = k + 1;
	if (n < kp1) continue;
        ax = aa;
        for (j = kp1; j <= n; ++j) {
	    t = a[k + j * n];
	    a[k + j * n] = 0.;
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
            ax = &a[j * n + 1];
            ay = &a[k * n + 1];
            for ( ll=0; ll<n; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
	l = ipvt[k];
	if (l != k) {
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

