#ifndef lint
static char vcid[] = "$Id: dgefa3.c,v 1.1 1996/04/28 00:57:27 bsmith Exp bsmith $";
#endif
/*
    Inverts 3 by 3 matrix using partial pivoting.
*/
#include "petsc.h"

int Kernel_A_gets_inverse_A_3(Scalar *a)
{
    int     i__2, i__3, kp1, j, k, l,ll,i,ipvt_l[3],*ipvt = ipvt_l,kb;
    Scalar  t,*aa,*ax,*ay,work_l[9],*work = work_l,stmp;
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
	  t          = a[l + 3*k];
	  a[l + 3*k] = a[4*k];
	  a[4*k]     = t;
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

    /*
         Now form the inverse 
    */

    --work;

   /*     compute inverse(u) */

    for (k = 1; k <= 3; ++k) {
	a[k + 3*k] = 1.0 / a[k + 3*k];
	t = -a[k + 3*k];
	i__2 = k - 1;
        aa = &a[3*k + 1]; 
        for ( ll=0; ll<i__2; ll++ ) aa[ll] *= t;
	kp1 = k + 1;
	if (3 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 3; ++j) {
	    t = a[k + 3*j];
	    a[k + 3*j] = 0.;
            ay = &a[3*j + 1];
            for ( ll=0; ll<k; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
    }

   /*    form inverse(u)*inverse(l) */

    for (kb = 1; kb <= 2; ++kb) {
	k   = 3 - kb;
	kp1 = k + 1;
        aa  = a + 3*k;
	for (i = kp1; i <= 3; ++i) {
	    work[i] = aa[i];
	    aa[i]   = 0.;
	}
	for (j = kp1; j <= 3; ++j) {
	    t = work[j];
            ax = &a[3*j + 1];
            ay = &a[3*k + 1];
            for ( ll=0; ll<3; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
	l = ipvt[k];
	if (l != k) {
            ax = &a[3*k + 1]; 
            ay = &a[3*l + 1];
            for ( ll=0; ll<3; ll++ ) {
              stmp    = ax[ll];
              ax[ll] = ay[ll];
              ay[ll] = stmp;
            }
	}
    }
    return 0;
}

