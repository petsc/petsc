
/*
       This routine was converted by f2c from Linpack source
             linpack. this version dated 08/14/78 
      cleve moler, university of new mexico, argonne national lab.
*/
#include "petsc.h"

int Linpack_DGEFA(Scalar *a, int n, int *ipvt)
{
    int     a_offset, i__1, i__2, i__3, kp1, nm1, j, k, l,ll;
    Scalar  t,*aa,tmp,max,*ax,*ay;

/*     gaussian elimination with partial pivoting */

    /* Parameter adjustments */
    --ipvt;
    a_offset = n + 1;
    a       -= a_offset;

    /* Function Body */
    nm1 = n - 1;
    if (nm1 < 1) {
	goto L70;
    }
    i__1 = nm1;
    for (k = 1; k <= i__1; ++k) {
	kp1 = k + 1;

/*        find l = pivot index */

	i__2 = n - k + 1;
	/* l = idamax_(&i__2, &a[k + k * n], &c__1) + k - 1; */
        aa = &a[k + k * n];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for ( ll=1; ll<i__2; ll++ ) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l += k - 1;
	ipvt[k] = l;

/*        zero pivot implies this column already triangularized */

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
	/* dscal_(&i__2, &t, &a[k + 1 + k * n], &c__1); */
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
	    /* daxpy_(&i__3, &t, &a[k + 1 + k * n], &c__1, &a[k + 1 + j * 
		    n], &c__1); */
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

