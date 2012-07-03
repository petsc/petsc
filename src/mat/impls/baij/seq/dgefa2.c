
/*
     Inverts 2 by 2 matrix using partial pivoting.

       Used by the sparse factorization routines in 
     src/mat/impls/baij/seq


       This is a combination of the Linpack routines
    dgefa() and dgedi() specialized for a size of 2.

*/
#include <petscsys.h>

#undef __FUNCT__  
#define __FUNCT__ "PetscKernel_A_gets_inverse_A_2"
PetscErrorCode PetscKernel_A_gets_inverse_A_2(MatScalar *a,PetscReal shift)
{
    PetscInt   i__2,i__3,kp1,j,k,l,ll,i,ipvt[2],k3;
    PetscInt   k4,j3;
    MatScalar  *aa,*ax,*ay,work[4],stmp;
    MatReal    tmp,max;

/*     gaussian elimination with partial pivoting */

    PetscFunctionBegin;
    shift = .25*shift*(1.e-12 + PetscAbsScalar(a[0]) + PetscAbsScalar(a[3]));
    /* Parameter adjustments */
    a       -= 3;

    /*for (k = 1; k <= 1; ++k) {*/
        k   = 1; 
	kp1 = k + 1;
        k3  = 2*k;
        k4  = k3 + k;
/*        find l = pivot index */

	i__2 = 3 - k;
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
	i__2 = 2 - k;
        aa = &a[1 + k4]; 
        for (ll=0; ll<i__2; ll++) {
          aa[ll] *= stmp;
        }

/*           row elimination with column indexing */

	ax = &a[k4+1]; 
        for (j = kp1; j <= 2; ++j) {
            j3   = 2*j;
	    stmp = a[l + j3];
	    if (l != k) {
	      a[l + j3] = a[k + j3];
	      a[k + j3] = stmp;
            }

	    i__3 = 2 - k;
            ay = &a[1+k+j3];
            for (ll=0; ll<i__3; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    /*}*/
    ipvt[1] = 2;
    if (a[6] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",1);

    /*
         Now form the inverse 
    */

   /*     compute inverse(u) */

    for (k = 1; k <= 2; ++k) {
        k3    = 2*k;
        k4    = k3 + k;
	a[k4] = 1.0 / a[k4];
	stmp  = -a[k4];
	i__2  = k - 1;
        aa    = &a[k3 + 1]; 
        for (ll=0; ll<i__2; ll++) aa[ll] *= stmp;
	kp1 = k + 1;
	if (2 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 2; ++j) {
            j3        = 2*j;
	    stmp      = a[k + j3];
	    a[k + j3] = 0.0;
            ay        = &a[j3 + 1];
            for (ll=0; ll<k; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    }

   /*    form inverse(u)*inverse(l) */

    /*for (kb = 1; kb <= 1; ++kb) {*/
        
	k   = 1;
        k3  = 2*k;
	kp1 = k + 1;
        aa  = a + k3;
	for (i = kp1; i <= 2; ++i) {
            work[i-1] = aa[i];
	    aa[i]   = 0.0;
	}
	for (j = kp1; j <= 2; ++j) {
	    stmp  = work[j-1];
            ax    = &a[2*j + 1];
            ay    = &a[k3 + 1];
            ay[0] += stmp*ax[0];
            ay[1] += stmp*ax[1];
	}
	l = ipvt[k-1];
	if (l != k) {
            ax = &a[k3 + 1]; 
            ay = &a[2*l + 1];
            stmp = ax[0]; ax[0] = ay[0]; ay[0] = stmp;
            stmp = ax[1]; ax[1] = ay[1]; ay[1] = stmp;
	}
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscKernel_A_gets_inverse_A_9"
PetscErrorCode PetscKernel_A_gets_inverse_A_9(MatScalar *a,PetscReal shift)
{
    PetscInt   i__2,i__3,kp1,j,k,l,ll,i,ipvt[9],kb,k3;
    PetscInt   k4,j3;
    MatScalar  *aa,*ax,*ay,work[81],stmp;
    MatReal    tmp,max;

/*     gaussian elimination with partial pivoting */

    PetscFunctionBegin;
    /* Parameter adjustments */
    a       -= 10;

    for (k = 1; k <= 8; ++k) {
	kp1 = k + 1;
        k3  = 9*k;
        k4  = k3 + k;
/*        find l = pivot index */

	i__2 = 10 - k;
        aa = &a[k4];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for (ll=1; ll<i__2; ll++) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l       += k - 1;
	ipvt[k-1] = l;

	if (a[l + k3] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",k-1);

/*           interchange if necessary */

	if (l != k) {
	  stmp      = a[l + k3];
	  a[l + k3] = a[k4];
	  a[k4]     = stmp;
        }

/*           compute multipliers */

	stmp = -1. / a[k4];
	i__2 = 9 - k;
        aa = &a[1 + k4]; 
        for (ll=0; ll<i__2; ll++) {
          aa[ll] *= stmp;
        }

/*           row elimination with column indexing */

	ax = &a[k4+1]; 
        for (j = kp1; j <= 9; ++j) {
            j3   = 9*j;
	    stmp = a[l + j3];
	    if (l != k) {
	      a[l + j3] = a[k + j3];
	      a[k + j3] = stmp;
            }

	    i__3 = 9 - k;
            ay = &a[1+k+j3];
            for (ll=0; ll<i__3; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    }
    ipvt[8] = 9;
    if (a[90] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",6);

    /*
         Now form the inverse 
    */

   /*     compute inverse(u) */

    for (k = 1; k <= 9; ++k) {
        k3    = 9*k;
        k4    = k3 + k;
	a[k4] = 1.0 / a[k4];
	stmp  = -a[k4];
	i__2  = k - 1;
        aa    = &a[k3 + 1]; 
        for (ll=0; ll<i__2; ll++) aa[ll] *= stmp;
	kp1 = k + 1;
	if (9 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 9; ++j) {
            j3        = 9*j;
	    stmp      = a[k + j3];
	    a[k + j3] = 0.0;
            ay        = &a[j3 + 1];
            for (ll=0; ll<k; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    }

   /*    form inverse(u)*inverse(l) */

    for (kb = 1; kb <= 8; ++kb) {
	k   = 9 - kb;
        k3  = 9*k;
	kp1 = k + 1;
        aa  = a + k3;
	for (i = kp1; i <= 9; ++i) {
            work[i-1] = aa[i];
	    aa[i]   = 0.0;
	}
	for (j = kp1; j <= 9; ++j) {
	    stmp  = work[j-1];
            ax    = &a[9*j + 1];
            ay    = &a[k3 + 1];
            ay[0] += stmp*ax[0];
            ay[1] += stmp*ax[1];
            ay[2] += stmp*ax[2];
            ay[3] += stmp*ax[3];
            ay[4] += stmp*ax[4];
            ay[5] += stmp*ax[5];
            ay[6] += stmp*ax[6];
            ay[7] += stmp*ax[7];
            ay[8] += stmp*ax[8];
	}
	l = ipvt[k-1];
	if (l != k) {
            ax = &a[k3 + 1]; 
            ay = &a[9*l + 1];
            stmp = ax[0]; ax[0] = ay[0]; ay[0] = stmp;
            stmp = ax[1]; ax[1] = ay[1]; ay[1] = stmp;
            stmp = ax[2]; ax[2] = ay[2]; ay[2] = stmp;
            stmp = ax[3]; ax[3] = ay[3]; ay[3] = stmp;
            stmp = ax[4]; ax[4] = ay[4]; ay[4] = stmp;
            stmp = ax[5]; ax[5] = ay[5]; ay[5] = stmp;
            stmp = ax[6]; ax[6] = ay[6]; ay[6] = stmp;
            stmp = ax[7]; ax[7] = ay[7]; ay[7] = stmp;
            stmp = ax[8]; ax[8] = ay[8]; ay[8] = stmp;
	}
    }
    PetscFunctionReturn(0);
}

/*
      Inverts 15 by 15 matrix using partial pivoting.

       Used by the sparse factorization routines in 
     src/mat/impls/baij/seq

       This is a combination of the Linpack routines
    dgefa() and dgedi() specialized for a size of 15.

*/

#undef __FUNCT__  
#define __FUNCT__ "PetscKernel_A_gets_inverse_A_15"
PetscErrorCode PetscKernel_A_gets_inverse_A_15(MatScalar *a,PetscInt *ipvt,MatScalar *work,PetscReal shift)
{
    PetscInt         i__2,i__3,kp1,j,k,l,ll,i,kb,k3;
    PetscInt         k4,j3;
    MatScalar        *aa,*ax,*ay,stmp;
    MatReal          tmp,max;

/*     gaussian elimination with partial pivoting */

    PetscFunctionBegin;
    /* Parameter adjustments */
    a       -= 16;

    for (k = 1; k <= 14; ++k) {
	kp1 = k + 1;
        k3  = 15*k;
        k4  = k3 + k;
/*        find l = pivot index */

	i__2 = 16 - k;
        aa = &a[k4];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for (ll=1; ll<i__2; ll++) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l       += k - 1;
	ipvt[k-1] = l;

	if (a[l + k3] == 0.0)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",k-1);

/*           interchange if necessary */

	if (l != k) {
	  stmp      = a[l + k3];
	  a[l + k3] = a[k4];
	  a[k4]     = stmp;
        }

/*           compute multipliers */

	stmp = -1. / a[k4];
	i__2 = 15 - k;
        aa = &a[1 + k4]; 
        for (ll=0; ll<i__2; ll++) {
          aa[ll] *= stmp;
        }

/*           row elimination with column indexing */

	ax = &a[k4+1]; 
        for (j = kp1; j <= 15; ++j) {
            j3   = 15*j;
	    stmp = a[l + j3];
	    if (l != k) {
	      a[l + j3] = a[k + j3];
	      a[k + j3] = stmp;
            }

	    i__3 = 15 - k;
            ay = &a[1+k+j3];
            for (ll=0; ll<i__3; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    }
    ipvt[14] = 15;
    if (a[240] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",6);

    /*
         Now form the inverse 
    */

   /*     compute inverse(u) */

    for (k = 1; k <= 15; ++k) {
        k3    = 15*k;
        k4    = k3 + k;
	a[k4] = 1.0 / a[k4];
	stmp  = -a[k4];
	i__2  = k - 1;
        aa    = &a[k3 + 1]; 
        for (ll=0; ll<i__2; ll++) aa[ll] *= stmp;
	kp1 = k + 1;
	if (15 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 15; ++j) {
            j3        = 15*j;
	    stmp      = a[k + j3];
	    a[k + j3] = 0.0;
            ay        = &a[j3 + 1];
            for (ll=0; ll<k; ll++) {
              ay[ll] += stmp*ax[ll];
            }
	}
    }

   /*    form inverse(u)*inverse(l) */

    for (kb = 1; kb <= 14; ++kb) {
	k   = 15 - kb;
        k3  = 15*k;
	kp1 = k + 1;
        aa  = a + k3;
	for (i = kp1; i <= 15; ++i) {
            work[i-1] = aa[i];
	    aa[i]   = 0.0;
	}
	for (j = kp1; j <= 15; ++j) {
	    stmp  = work[j-1];
            ax    = &a[15*j + 1];
            ay    = &a[k3 + 1];
            ay[0]  += stmp*ax[0];
            ay[1]  += stmp*ax[1];
            ay[2]  += stmp*ax[2];
            ay[3]  += stmp*ax[3];
            ay[4]  += stmp*ax[4];
            ay[5]  += stmp*ax[5];
            ay[6]  += stmp*ax[6];
	    ay[7]  += stmp*ax[7];
            ay[8]  += stmp*ax[8];
            ay[9]  += stmp*ax[9];
            ay[10] += stmp*ax[10];
            ay[11] += stmp*ax[11];
            ay[12] += stmp*ax[12];
            ay[13] += stmp*ax[13];
	    ay[14] += stmp*ax[14];
	}
	l = ipvt[k-1];
	if (l != k) {
            ax = &a[k3 + 1]; 
            ay = &a[15*l + 1];
            stmp = ax[0];  ax[0]  = ay[0];  ay[0]  = stmp;
            stmp = ax[1];  ax[1]  = ay[1];  ay[1]  = stmp;
            stmp = ax[2];  ax[2]  = ay[2];  ay[2]  = stmp;
            stmp = ax[3];  ax[3]  = ay[3];  ay[3]  = stmp;
            stmp = ax[4];  ax[4]  = ay[4];  ay[4]  = stmp;
            stmp = ax[5];  ax[5]  = ay[5];  ay[5]  = stmp;
            stmp = ax[6];  ax[6]  = ay[6];  ay[6]  = stmp;
	    stmp = ax[7];  ax[7]  = ay[7];  ay[7]  = stmp;
            stmp = ax[8];  ax[8]  = ay[8];  ay[8]  = stmp;
            stmp = ax[9];  ax[9]  = ay[9];  ay[9]  = stmp;
            stmp = ax[10]; ax[10] = ay[10]; ay[10] = stmp;
            stmp = ax[11]; ax[11] = ay[11]; ay[11] = stmp;
            stmp = ax[12]; ax[12] = ay[12]; ay[12] = stmp;
            stmp = ax[13]; ax[13] = ay[13]; ay[13] = stmp;
	    stmp = ax[14]; ax[14] = ay[14]; ay[14] = stmp;
	}
    }
    PetscFunctionReturn(0);
}
