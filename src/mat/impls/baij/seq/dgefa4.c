
/*
       Inverts 4 by 4 matrix using partial pivoting.

       Used by the sparse factorization routines in
     src/mat/impls/baij/seq

       This is a combination of the Linpack routines
    dgefa() and dgedi() specialized for a size of 4.

*/
#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "PetscKernel_A_gets_inverse_A_4"
PetscErrorCode PetscKernel_A_gets_inverse_A_4(MatScalar *a,PetscReal shift)
{
    PetscInt   i__2,i__3,kp1,j,k,l,ll,i,ipvt[4],kb,k3;
    PetscInt   k4,j3;
    MatScalar  *aa,*ax,*ay,work[16],stmp;
    MatReal    tmp,max;

/*     gaussian elimination with partial pivoting */

    PetscFunctionBegin;
    shift = .25*shift*(1.e-12 + PetscAbsScalar(a[0]) + PetscAbsScalar(a[5]) + PetscAbsScalar(a[10]) + PetscAbsScalar(a[15]));
    /* Parameter adjustments */
    a       -= 5;

    for (k = 1; k <= 3; ++k) {
        kp1 = k + 1;
        k3  = 4*k;
        k4  = k3 + k;
/*        find l = pivot index */

        i__2 = 5 - k;
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
        i__2 = 4 - k;
        aa = &a[1 + k4];
        for (ll=0; ll<i__2; ll++) {
          aa[ll] *= stmp;
        }

/*           row elimination with column indexing */

        ax = &a[k4+1];
        for (j = kp1; j <= 4; ++j) {
            j3   = 4*j;
            stmp = a[l + j3];
            if (l != k) {
              a[l + j3] = a[k + j3];
              a[k + j3] = stmp;
            }

            i__3 = 4 - k;
            ay = &a[1+k+j3];
            for (ll=0; ll<i__3; ll++) {
              ay[ll] += stmp*ax[ll];
            }
        }
    }
    ipvt[3] = 4;
    if (a[20] == 0.0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot, row %D",3);

    /*
         Now form the inverse
    */

   /*     compute inverse(u) */

    for (k = 1; k <= 4; ++k) {
        k3    = 4*k;
        k4    = k3 + k;
        a[k4] = 1.0 / a[k4];
        stmp  = -a[k4];
        i__2  = k - 1;
        aa    = &a[k3 + 1];
        for (ll=0; ll<i__2; ll++) aa[ll] *= stmp;
        kp1 = k + 1;
        if (4 < kp1) continue;
        ax = aa;
        for (j = kp1; j <= 4; ++j) {
            j3        = 4*j;
            stmp      = a[k + j3];
            a[k + j3] = 0.0;
            ay        = &a[j3 + 1];
            for (ll=0; ll<k; ll++) {
              ay[ll] += stmp*ax[ll];
            }
        }
    }

   /*    form inverse(u)*inverse(l) */

    for (kb = 1; kb <= 3; ++kb) {
        k   = 4 - kb;
        k3  = 4*k;
        kp1 = k + 1;
        aa  = a + k3;
        for (i = kp1; i <= 4; ++i) {
            work[i-1] = aa[i];
            aa[i]   = 0.0;
        }
        for (j = kp1; j <= 4; ++j) {
            stmp  = work[j-1];
            ax    = &a[4*j + 1];
            ay    = &a[k3 + 1];
            ay[0] += stmp*ax[0];
            ay[1] += stmp*ax[1];
            ay[2] += stmp*ax[2];
            ay[3] += stmp*ax[3];
        }
        l = ipvt[k-1];
        if (l != k) {
            ax = &a[k3 + 1];
            ay = &a[4*l + 1];
            stmp = ax[0]; ax[0] = ay[0]; ay[0] = stmp;
            stmp = ax[1]; ax[1] = ay[1]; ay[1] = stmp;
            stmp = ax[2]; ax[2] = ay[2]; ay[2] = stmp;
            stmp = ax[3]; ax[3] = ay[3]; ay[3] = stmp;
        }
    }
    PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_SSE)
#include PETSC_HAVE_SSE

#undef __FUNCT__
#define __FUNCT__ "PetscKernel_A_gets_inverse_A_4_SSE"
PetscErrorCode PetscKernel_A_gets_inverse_A_4_SSE(float *a)
{
  /*
     This routine is converted from Intel's Small Matrix Library.
     See: Streaming SIMD Extensions -- Inverse of 4x4 Matrix
     Order Number: 245043-001
     March 1999
     http://www.intel.com

     Inverse of a 4x4 matrix via Kramer's Rule:
     bool Invert4x4(SMLXMatrix &);
  */
  PetscFunctionBegin;

  SSE_SCOPE_BEGIN;
    SSE_INLINE_BEGIN_1(a)

/* ----------------------------------------------- */

      SSE_LOADL_PS(SSE_ARG_1,FLOAT_0,XMM0)
      SSE_LOADH_PS(SSE_ARG_1,FLOAT_4,XMM0)

      SSE_LOADL_PS(SSE_ARG_1,FLOAT_8,XMM5)
      SSE_LOADH_PS(SSE_ARG_1,FLOAT_12,XMM5)

      SSE_COPY_PS(XMM3,XMM0)
      SSE_SHUFFLE(XMM3,XMM5,0x88)

      SSE_SHUFFLE(XMM5,XMM0,0xDD)

      SSE_LOADL_PS(SSE_ARG_1,FLOAT_2,XMM0)
      SSE_LOADH_PS(SSE_ARG_1,FLOAT_6,XMM0)

      SSE_LOADL_PS(SSE_ARG_1,FLOAT_10,XMM6)
      SSE_LOADH_PS(SSE_ARG_1,FLOAT_14,XMM6)

      SSE_COPY_PS(XMM4,XMM0)
      SSE_SHUFFLE(XMM4,XMM6,0x88)

      SSE_SHUFFLE(XMM6,XMM0,0xDD)

/* ----------------------------------------------- */

      SSE_COPY_PS(XMM7,XMM4)
      SSE_MULT_PS(XMM7,XMM6)

      SSE_SHUFFLE(XMM7,XMM7,0xB1)

      SSE_COPY_PS(XMM0,XMM5)
      SSE_MULT_PS(XMM0,XMM7)

      SSE_COPY_PS(XMM2,XMM3)
      SSE_MULT_PS(XMM2,XMM7)

      SSE_SHUFFLE(XMM7,XMM7,0x4E)

      SSE_COPY_PS(XMM1,XMM5)
      SSE_MULT_PS(XMM1,XMM7)
      SSE_SUB_PS(XMM1,XMM0)

      SSE_MULT_PS(XMM7,XMM3)
      SSE_SUB_PS(XMM7,XMM2)

      SSE_SHUFFLE(XMM7,XMM7,0x4E)
      SSE_STORE_PS(SSE_ARG_1,FLOAT_4,XMM7)

/* ----------------------------------------------- */

      SSE_COPY_PS(XMM0,XMM5)
      SSE_MULT_PS(XMM0,XMM4)

      SSE_SHUFFLE(XMM0,XMM0,0xB1)

      SSE_COPY_PS(XMM2,XMM6)
      SSE_MULT_PS(XMM2,XMM0)
      SSE_ADD_PS(XMM2,XMM1)

      SSE_COPY_PS(XMM7,XMM3)
      SSE_MULT_PS(XMM7,XMM0)

      SSE_SHUFFLE(XMM0,XMM0,0x4E)

      SSE_COPY_PS(XMM1,XMM6)
      SSE_MULT_PS(XMM1,XMM0)
      SSE_SUB_PS(XMM2,XMM1)

      SSE_MULT_PS(XMM0,XMM3)
      SSE_SUB_PS(XMM0,XMM7)

      SSE_SHUFFLE(XMM0,XMM0,0x4E)
      SSE_STORE_PS(SSE_ARG_1,FLOAT_12,XMM0)

      /* ----------------------------------------------- */

      SSE_COPY_PS(XMM7,XMM5)
      SSE_SHUFFLE(XMM7,XMM5,0x4E)
      SSE_MULT_PS(XMM7,XMM6)

      SSE_SHUFFLE(XMM7,XMM7,0xB1)

      SSE_SHUFFLE(XMM4,XMM4,0x4E)

      SSE_COPY_PS(XMM0,XMM4)
      SSE_MULT_PS(XMM0,XMM7)
      SSE_ADD_PS(XMM0,XMM2)

      SSE_COPY_PS(XMM2,XMM3)
      SSE_MULT_PS(XMM2,XMM7)

      SSE_SHUFFLE(XMM7,XMM7,0x4E)

      SSE_COPY_PS(XMM1,XMM4)
      SSE_MULT_PS(XMM1,XMM7)
      SSE_SUB_PS(XMM0,XMM1)
      SSE_STORE_PS(SSE_ARG_1,FLOAT_0,XMM0)

      SSE_MULT_PS(XMM7,XMM3)
      SSE_SUB_PS(XMM7,XMM2)

      SSE_SHUFFLE(XMM7,XMM7,0x4E)

      /* ----------------------------------------------- */

      SSE_COPY_PS(XMM1,XMM3)
      SSE_MULT_PS(XMM1,XMM5)

      SSE_SHUFFLE(XMM1,XMM1,0xB1)

      SSE_COPY_PS(XMM0,XMM6)
      SSE_MULT_PS(XMM0,XMM1)
      SSE_ADD_PS(XMM0,XMM7)

      SSE_COPY_PS(XMM2,XMM4)
      SSE_MULT_PS(XMM2,XMM1)
      SSE_SUB_PS_M(XMM2,SSE_ARG_1,FLOAT_12)

      SSE_SHUFFLE(XMM1,XMM1,0x4E)

      SSE_COPY_PS(XMM7,XMM6)
      SSE_MULT_PS(XMM7,XMM1)
      SSE_SUB_PS(XMM7,XMM0)

      SSE_MULT_PS(XMM1,XMM4)
      SSE_SUB_PS(XMM2,XMM1)
      SSE_STORE_PS(SSE_ARG_1,FLOAT_12,XMM2)

      /* ----------------------------------------------- */

      SSE_COPY_PS(XMM1,XMM3)
      SSE_MULT_PS(XMM1,XMM6)

      SSE_SHUFFLE(XMM1,XMM1,0xB1)

      SSE_COPY_PS(XMM2,XMM4)
      SSE_MULT_PS(XMM2,XMM1)
      SSE_LOAD_PS(SSE_ARG_1,FLOAT_4,XMM0)
      SSE_SUB_PS(XMM0,XMM2)

      SSE_COPY_PS(XMM2,XMM5)
      SSE_MULT_PS(XMM2,XMM1)
      SSE_ADD_PS(XMM2,XMM7)

      SSE_SHUFFLE(XMM1,XMM1,0x4E)

      SSE_COPY_PS(XMM7,XMM4)
      SSE_MULT_PS(XMM7,XMM1)
      SSE_ADD_PS(XMM7,XMM0)

      SSE_MULT_PS(XMM1,XMM5)
      SSE_SUB_PS(XMM2,XMM1)

      /* ----------------------------------------------- */

      SSE_MULT_PS(XMM4,XMM3)

      SSE_SHUFFLE(XMM4,XMM4,0xB1)

      SSE_COPY_PS(XMM1,XMM6)
      SSE_MULT_PS(XMM1,XMM4)
      SSE_ADD_PS(XMM1,XMM7)

      SSE_COPY_PS(XMM0,XMM5)
      SSE_MULT_PS(XMM0,XMM4)
      SSE_LOAD_PS(SSE_ARG_1,FLOAT_12,XMM7)
      SSE_SUB_PS(XMM7,XMM0)

      SSE_SHUFFLE(XMM4,XMM4,0x4E)

      SSE_MULT_PS(XMM6,XMM4)
      SSE_SUB_PS(XMM1,XMM6)

      SSE_MULT_PS(XMM5,XMM4)
      SSE_ADD_PS(XMM5,XMM7)

      /* ----------------------------------------------- */

      SSE_LOAD_PS(SSE_ARG_1,FLOAT_0,XMM0)
      SSE_MULT_PS(XMM3,XMM0)

      SSE_COPY_PS(XMM4,XMM3)
      SSE_SHUFFLE(XMM4,XMM3,0x4E)
      SSE_ADD_PS(XMM4,XMM3)

      SSE_COPY_PS(XMM6,XMM4)
      SSE_SHUFFLE(XMM6,XMM4,0xB1)
      SSE_ADD_SS(XMM6,XMM4)

      SSE_COPY_PS(XMM3,XMM6)
      SSE_RECIP_SS(XMM3,XMM6)
      SSE_COPY_SS(XMM4,XMM3)
      SSE_ADD_SS(XMM4,XMM3)
      SSE_MULT_SS(XMM3,XMM3)
      SSE_MULT_SS(XMM6,XMM3)
      SSE_SUB_SS(XMM4,XMM6)

      SSE_SHUFFLE(XMM4,XMM4,0x00)

      SSE_MULT_PS(XMM0,XMM4)
      SSE_STOREL_PS(SSE_ARG_1,FLOAT_0,XMM0)
      SSE_STOREH_PS(SSE_ARG_1,FLOAT_2,XMM0)

      SSE_MULT_PS(XMM1,XMM4)
      SSE_STOREL_PS(SSE_ARG_1,FLOAT_4,XMM1)
      SSE_STOREH_PS(SSE_ARG_1,FLOAT_6,XMM1)

      SSE_MULT_PS(XMM2,XMM4)
      SSE_STOREL_PS(SSE_ARG_1,FLOAT_8,XMM2)
      SSE_STOREH_PS(SSE_ARG_1,FLOAT_10,XMM2)

      SSE_MULT_PS(XMM4,XMM5)
      SSE_STOREL_PS(SSE_ARG_1,FLOAT_12,XMM4)
      SSE_STOREH_PS(SSE_ARG_1,FLOAT_14,XMM4)

      /* ----------------------------------------------- */

      SSE_INLINE_END_1;
  SSE_SCOPE_END;

  PetscFunctionReturn(0);
}

#endif


