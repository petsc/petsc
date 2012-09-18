
/*
       This routine was converted by f2c from Linpack source
             linpack. this version dated 08/14/78
      cleve moler, university of new mexico, argonne national lab.

        Does an LU factorization with partial pivoting of a dense
     n by n matrix.

       Used by the sparse factorization routines in
     src/mat/impls/baij/seq

*/
#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "PetscLINPACKgefa"
PetscErrorCode PetscLINPACKgefa(MatScalar *a,PetscInt n,PetscInt *ipvt)
{
    PetscInt   i__2,i__3,kp1,nm1,j,k,l,ll,kn,knp1,jn1;
    MatScalar  t,*ax,*ay,*aa;
    MatReal    tmp,max;

/*     gaussian elimination with partial pivoting */

    PetscFunctionBegin;
    /* Parameter adjustments */
    --ipvt;
    a       -= n + 1;

    /* Function Body */
    nm1 = n - 1;
    for (k = 1; k <= nm1; ++k) {
	kp1  = k + 1;
        kn   = k*n;
        knp1 = k*n + k;

/*        find l = pivot index */

	i__2 = n - k + 1;
        aa = &a[knp1];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for (ll=1; ll<i__2; ll++) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l += k - 1;
	ipvt[k] = l;

	if (a[l + kn] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",k-1);

/*           interchange if necessary */

	if (l != k) {
	  t = a[l + kn];
	  a[l + kn] = a[knp1];
	  a[knp1] = t;
        }

/*           compute multipliers */

	t = -1. / a[knp1];
	i__2 = n - k;
        aa = &a[1 + knp1];
        for (ll=0; ll<i__2; ll++) {
          aa[ll] *= t;
        }

/*           row elimination with column indexing */

	ax = aa;
        for (j = kp1; j <= n; ++j) {
            jn1 = j*n;
	    t = a[l + jn1];
	    if (l != k) {
	      a[l + jn1] = a[k + jn1];
	      a[k + jn1] = t;
            }

	    i__3 = n - k;
            ay = &a[1+k+jn1];
            for (ll=0; ll<i__3; ll++) {
              ay[ll] += t*ax[ll];
            }
	}
    }
    ipvt[n] = n;
    if (a[n + n * n] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",n-1);
    PetscFunctionReturn(0);
}

