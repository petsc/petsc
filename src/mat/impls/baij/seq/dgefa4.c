/*$Id: dgefa4.c,v 1.16 2001/04/05 18:07:25 buschelm Exp bsmith $*/
/*
       Inverts 4 by 4 matrix using partial pivoting.

       Used by the sparse factorization routines in 
     src/mat/impls/baij/seq and src/mat/impls/bdiag/seq

       See also src/inline/ilu.h

       This is a combination of the Linpack routines
    dgefa() and dgedi() specialized for a size of 4.

*/
#include "petsc.h"

#undef __FUNCT__  
#define __FUNCT__ "Kernel_A_gets_inverse_A_4"
int Kernel_A_gets_inverse_A_4(MatScalar *a)
{
    int        i__2,i__3,kp1,j,k,l,ll,i,ipvt[4],kb,k3;
    int        k4,j3;
    MatScalar  *aa,*ax,*ay,work_l[16],*work = work_l-1,stmp;
    MatReal    tmp,max;

/*     gaussian elimination with partial pivoting */

    PetscFunctionBegin;
    /* Parameter adjustments */
    a       -= 5;

    for (k = 1; k <= 3; ++k) {
        kp1 = k + 1;
        k3  = 4*k;
        k4  = k3 + k;
/*        find l = pivot index */

        i__2 = 4 - k;
        aa = &a[k4];
        max = PetscAbsScalar(aa[0]);
        l = 1;
        for (ll=1; ll<i__2; ll++) {
          tmp = PetscAbsScalar(aa[ll]);
          if (tmp > max) { max = tmp; l = ll+1;}
        }
        l       += k - 1;
        ipvt[k-1] = l;

        if (a[l + k3] == 0.) {
          SETERRQ(k,"Zero pivot");
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
    if (a[20] == 0.) {
        SETERRQ(3,"Zero pivot,final row");
    }

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
            work_l[i-1] = aa[i];
            /* work[i] = aa[i]; Fix for -O3 error on Origin 2000 */ 
            aa[i]   = 0.0;
        }
        for (j = kp1; j <= 4; ++j) {
            stmp  = work[j];
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

#ifdef PETSC_HAVE_ICL_SSE
#include "xmmintrin.h"

#undef __FUNCT__
#define __FUNCT__ "Kernel_A_gets_inverse_A_4SSE"
int Kernel_A_gets_inverse_A_4SSE(float *a)
{
  /* 
     This routine is taken from Intel's Small Matrix Library.
     See: Streaming SIMD Extensions -- Inverse of 4x4 Matrix
     Order Number: 245043-001
     March 1999
     http://www.intel.com

     Note: Intel's SML uses row-wise storage for these small matrices,
     and PETSc uses column-wise storage.  However since inv(A')=(inv(A))'
     the same code can be used here.

     Inverse of a 4x4 matrix via Kramer's Rule:
     bool Invert4x4(SMLXMatrix &);
  */
  __m128 minor0, minor1, minor2, minor3;
  __m128 row0, row1, row2, row3;
  __m128 det, tmp1;

  PetscFunctionBegin;
  tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(a)), (__m64*)(a+ 4));
  row1 = _mm_loadh_pi(_mm_loadl_pi(row1, (__m64*)(a+8)), (__m64*)(a+12));
  row0 = _mm_shuffle_ps(tmp1, row1, 0x88);
  row1 = _mm_shuffle_ps(row1, tmp1, 0xDD);
  tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(a+ 2)), (__m64*)(a+ 6));
  row3 = _mm_loadh_pi(_mm_loadl_pi(row3, (__m64*)(a+10)), (__m64*)(a+14));
  row2 = _mm_shuffle_ps(tmp1, row3, 0x88);
  row3 = _mm_shuffle_ps(row3, tmp1, 0xDD);
  /* ----------------------------------------------- */
  tmp1 = _mm_mul_ps(row2, row3);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
  minor0 = _mm_mul_ps(row1, tmp1);
  minor1 = _mm_mul_ps(row0, tmp1);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
  minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
  minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
  minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);
  /* ----------------------------------------------- */
  tmp1 = _mm_mul_ps(row1, row2);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
  minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
  minor3 = _mm_mul_ps(row0, tmp1);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
  minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
  minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
  minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);
  /* ----------------------------------------------- */
  tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
  row2 = _mm_shuffle_ps(row2, row2, 0x4E);
  minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
  minor2 = _mm_mul_ps(row0, tmp1);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
  minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
  minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
  minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);
  /* ----------------------------------------------- */
  tmp1 = _mm_mul_ps(row0, row1);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
  minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
  minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
  minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
  minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));
  /* ----------------------------------------------- */
  tmp1 = _mm_mul_ps(row0, row3);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
  minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
  minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
  minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
  minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));
  /* ----------------------------------------------- */
  tmp1 = _mm_mul_ps(row0, row2);
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
  minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
  minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
  tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
  minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
  minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);
  /* ----------------------------------------------- */
  det = _mm_mul_ps(row0, minor0);
  det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
  det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
  tmp1 = _mm_rcp_ss(det);
  det = _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
  det = _mm_shuffle_ps(det, det, 0x00);
  minor0 = _mm_mul_ps(det, minor0);
  _mm_storel_pi((__m64*)(a), minor0);
  _mm_storeh_pi((__m64*)(a+2), minor0);
  minor1 = _mm_mul_ps(det, minor1);
  _mm_storel_pi((__m64*)(a+4), minor1);
  _mm_storeh_pi((__m64*)(a+6), minor1);
  minor2 = _mm_mul_ps(det, minor2);
  _mm_storel_pi((__m64*)(a+ 8), minor2);
  _mm_storeh_pi((__m64*)(a+10), minor2);
  minor3 = _mm_mul_ps(det, minor3);
  _mm_storel_pi((__m64*)(a+12), minor3);
  _mm_storeh_pi((__m64*)(a+14), minor3);
  PetscFunctionReturn(0);
}

#endif


