
/*
      Inverts 5 by 5 matrix using partial pivoting.

       Used by the sparse factorization routines in
     src/mat/impls/baij/seq

       This is a combination of the Linpack routines
    dgefa() and dgedi() specialized for a size of 5.

*/
#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "PetscKernel_A_gets_inverse_A_5"
PetscErrorCode PetscKernel_A_gets_inverse_A_5(MatScalar *a,PetscInt *ipvt,MatScalar *work,PetscReal shift)
{
    PetscInt   i__2,i__3,kp1,j,k,l,ll,i,kb,k3;
    PetscInt   k4,j3;
    MatScalar  *aa,*ax,*ay,stmp;
    MatReal    tmp,max;

/*     gaussian elimination with partial pivoting */

    PetscFunctionBegin;
    shift = .25*shift*(1.e-12 + PetscAbsScalar(a[0]) + PetscAbsScalar(a[6]) + PetscAbsScalar(a[12]) + PetscAbsScalar(a[18]) + PetscAbsScalar(a[24]));
    /* Parameter adjustments */
    a       -= 6;

    for (k = 1; k <= 4; ++k) {
	kp1 = k + 1;
        k3  = 5*k;
        k4  = k3 + k;
/*        find l = pivot index */

	i__2 = 6 - k;
        aa = &a[k4];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for (ll=1; ll<i__2; ll++) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l       += k - 1;
	ipvt[k-1] = l;

        if (a[l + k3] == 0.0) {
          if (shift == 0.0) {
	    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",k-1);
  	  } else {
            /* SHIFT is applied to SINGLE diagonal entry; does this make any sense? */
  	    a[l + k3] = shift;
  	  }
        }

/*           interchange if necessary */

	if (l != k) {
	  stmp      = a[l + k3];
	  a[l + k3] = a[k4];
	  a[k4]     = stmp;
        }

/*           compute multipliers */

	stmp = -1. / a[k4];
	i__2 = 5 - k;
        aa = &a[1 + k4];
        for (ll=0; ll<i__2; ll++) {
          aa[ll] *= stmp;
        }

/*           row elimination with column indexing */

	ax = &a[k4+1];
        for (j = kp1; j <= 5; ++j) {
            j3   = 5*j;
	    stmp = a[l + j3];
	    if (l != k) {
	      a[l + j3] = a[k + j3];
	      a[k + j3] = stmp;
            }

	    i__3 = 5 - k;
            ay = &a[1+k+j3];
            for (ll=0; ll<i__3; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    }
    ipvt[4] = 5;
    if (a[30] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",4);

    /*
         Now form the inverse
    */

   /*     compute inverse(u) */

    for (k = 1; k <= 5; ++k) {
        k3    = 5*k;
        k4    = k3 + k;
	a[k4] = 1.0 / a[k4];
	stmp  = -a[k4];
	i__2  = k - 1;
        aa    = &a[k3 + 1];
        for (ll=0; ll<i__2; ll++) aa[ll] *= stmp;
	kp1 = k + 1;
	if (5 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 5; ++j) {
            j3        = 5*j;
	    stmp      = a[k + j3];
	    a[k + j3] = 0.0;
            ay        = &a[j3 + 1];
            for (ll=0; ll<k; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    }

   /*    form inverse(u)*inverse(l) */

    for (kb = 1; kb <= 4; ++kb) {
	k   = 5 - kb;
        k3  = 5*k;
	kp1 = k + 1;
        aa  = a + k3;
	for (i = kp1; i <= 5; ++i) {
            work[i-1] = aa[i];
	    aa[i]   = 0.0;
	}
	for (j = kp1; j <= 5; ++j) {
	    stmp  = work[j-1];
            ax    = &a[5*j + 1];
            ay    = &a[k3 + 1];
            ay[0] += stmp*ax[0];
            ay[1] += stmp*ax[1];
            ay[2] += stmp*ax[2];
            ay[3] += stmp*ax[3];
            ay[4] += stmp*ax[4];
	}
	l = ipvt[k-1];
	if (l != k) {
            ax = &a[k3 + 1];
            ay = &a[5*l + 1];
            stmp = ax[0]; ax[0] = ay[0]; ay[0] = stmp;
            stmp = ax[1]; ax[1] = ay[1]; ay[1] = stmp;
            stmp = ax[2]; ax[2] = ay[2]; ay[2] = stmp;
            stmp = ax[3]; ax[3] = ay[3]; ay[3] = stmp;
            stmp = ax[4]; ax[4] = ay[4]; ay[4] = stmp;
	}
    }
    PetscFunctionReturn(0);
}

