#ifndef lint
static char vcid[] = "$Id: dgefa3.c,v 1.2 1996/04/28 02:24:51 bsmith Exp bsmith $";
#endif
/*
    Inverts 3 by 3 matrix using partial pivoting.
*/
#include "petsc.h"

int Kernel_A_gets_inverse_A_3(Scalar *a)
{
    int     i__2, i__3, kp1, j, k, l,ll,i,ipvt_l[3],*ipvt = ipvt_l,kb,k3;
    int     k4,j3;
    Scalar  t,*aa,*ax,*ay,work_l[9],*work = work_l,stmp;
    double  tmp,max;

/*     gaussian elimination with partial pivoting */

    /* Parameter adjustments */
    --ipvt;
    a       -= 4;

    for (k = 1; k <= 2; ++k) {
	kp1 = k + 1;
        k3  = 3*k;
        k4  = k3 + k;
/*        find l = pivot index */

	i__2 = 4 - k;
        aa = &a[k4];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for ( ll=1; ll<i__2; ll++ ) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l += k - 1;
	ipvt[k] = l;

	if (a[l + k3] == 0.) {
	  SETERRQ(k,"Linpack_DGEFA:Zero pivot");
	}

/*           interchange if necessary */

	if (l != k) {
	  t          = a[l + k3];
	  a[l + k3] = a[k4];
	  a[k4]     = t;
        }

/*           compute multipliers */

	t = -1. / a[k4];
	i__2 = 3 - k;
        aa = &a[1 + k4]; 
        for ( ll=0; ll<i__2; ll++ ) {
          aa[ll] *= t;
        }

/*           row elimination with column indexing */

	ax = &a[k4+1]; 
        for (j = kp1; j <= 3; ++j) {
            j3 = 3*j;
	    t = a[l + j3];
	    if (l != k) {
	      a[l + j3] = a[k + j3];
	      a[k + j3] = t;
            }

	    i__3 = 3 - k;
            ay = &a[k+1+j3];
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
        k3 = 3*k;
        k4 = k3 + k;
	a[k4] = 1.0 / a[k4];
	t = -a[k4];
	i__2 = k - 1;
        aa = &a[k3 + 1]; 
        for ( ll=0; ll<i__2; ll++ ) aa[ll] *= t;
	kp1 = k + 1;
	if (3 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 3; ++j) {
            j3 = 3*j;
	    t = a[k + j3];
	    a[k + j3] = 0.;
            ay = &a[j3 + 1];
            for ( ll=0; ll<k; ll++ ) {
              ay[ll] += t*ax[ll];
            }
	}
    }

   /*    form inverse(u)*inverse(l) */

    for (kb = 1; kb <= 2; ++kb) {
	k   = 3 - kb;
        k3  = 3*k;
	kp1 = k + 1;
        aa  = a + k3;
	for (i = kp1; i <= 3; ++i) {
	    work[i] = aa[i];
	    aa[i]   = 0.0;
	}
	for (j = kp1; j <= 3; ++j) {
	    t = work[j];
            ax = &a[3*j + 1];
            ay = &a[k3 + 1];
            ay[0] += t*ax[0];
            ay[1] += t*ax[1];
            ay[2] += t*ax[2];
	}
	l = ipvt[k];
	if (l != k) {
            ax = &a[k3 + 1]; 
            ay = &a[3*l + 1];
            stmp = ax[0]; ax[0] = ay[0]; ay[0] = stmp;
            stmp = ax[1]; ax[1] = ay[1]; ay[1] = stmp;
            stmp = ax[2]; ax[2] = ay[2]; ay[2] = stmp;
	}
    }
    return 0;
}

