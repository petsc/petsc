/*$Id: baijfact12.c,v 1.5 2001/04/05 16:57:52 buschelm Exp buschelm $*/
/*
    Factorization code for BAIJ format. 
*/
#include "src/mat/impls/baij/seq/baij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/ilu.h"

#ifdef PETSC_HAVE_ICL_SSE
/*
    Version that uses Intel Compiler intrinsic functions and PentiumIII SSE registers
*/
/*
    A = A * B   A_gets_A_times_B

    A, B - 4x4 float arrays stored in column major order

    NOTE: A and B must be allocated as 16-byte aligned data

*/
int Kernel_A_gets_A_times_B_4SSE(float *a,float *b)
{ 
  __m128 A0,A1,A2,A3,C0,C1,C2,C3;
  
  PetscFunctionBegin;
/*    A0 = _mm_load_ps(a   ); */ 
/*    A1 = _mm_load_ps(a+4 ); */ 
/*    A2 = _mm_load_ps(a+8 ); */ 
/*    A3 = _mm_load_ps(a+12); */ 
  A0 = _mm_loadh_pi(_mm_loadl_pi(A0,(__m64 *)(a   )),(__m64 *)(a+2 )); 
  A1 = _mm_loadh_pi(_mm_loadl_pi(A1,(__m64 *)(a+4 )),(__m64 *)(a+6 )); 
  A2 = _mm_loadh_pi(_mm_loadl_pi(A2,(__m64 *)(a+8 )),(__m64 *)(a+10)); 
  A3 = _mm_loadh_pi(_mm_loadl_pi(A3,(__m64 *)(a+12)),(__m64 *)(a+14)); 

  C0 = _mm_mul_ps(A0,_mm_load_ps1(b)); 
  C0 = _mm_add_ps(_mm_mul_ps(A1,_mm_load_ps1(b+1)),C0); 
  C0 = _mm_add_ps(_mm_mul_ps(A2,_mm_load_ps1(b+2)),C0); 
  C0 = _mm_add_ps(_mm_mul_ps(A3,_mm_load_ps1(b+3)),C0); 
  
/*    _mm_store_ps(a,C0); */ 
  _mm_storel_pi((__m64 *)(a  ),C0); 
  _mm_storeh_pi((__m64 *)(a+2),C0); 
  
  C1 = _mm_mul_ps(A0,_mm_load_ps1(b+4)); 
  C1 = _mm_add_ps(_mm_mul_ps(A1,_mm_load_ps1(b+5)),C1); 
  C1 = _mm_add_ps(_mm_mul_ps(A2,_mm_load_ps1(b+6)),C1); 
  C1 = _mm_add_ps(_mm_mul_ps(A3,_mm_load_ps1(b+7)),C1); 
  
/*    _mm_store_ps(a+4,C1); */ 
  _mm_storel_pi((__m64 *)(a+4),C1); 
  _mm_storeh_pi((__m64 *)(a+6),C1); 
  
  C2 = _mm_mul_ps(A0,_mm_load_ps1(b+8)); 
  C2 = _mm_add_ps(_mm_mul_ps(A1,_mm_load_ps1(b+9 )),C2); 
  C2 = _mm_add_ps(_mm_mul_ps(A2,_mm_load_ps1(b+10)),C2); 
  C2 = _mm_add_ps(_mm_mul_ps(A3,_mm_load_ps1(b+11)),C2); 
  
/*    _mm_store_ps(a+8,C2); */ 
  _mm_storel_pi((__m64 *)(a+8 ),C2); 
  _mm_storeh_pi((__m64 *)(a+10),C2); 
  
  C3 = _mm_mul_ps(A0,_mm_load_ps1(b+12)); 
  C3 = _mm_add_ps(_mm_mul_ps(A1,_mm_load_ps1(b+13)),C3); 
  C3 = _mm_add_ps(_mm_mul_ps(A2,_mm_load_ps1(b+14)),C3); 
  C3 = _mm_add_ps(_mm_mul_ps(A3,_mm_load_ps1(b+15)),C3); 
  
/*    _mm_store_ps(a+12,C3); */ 
  _mm_storel_pi((__m64 *)(a+12),C3); 
  _mm_storeh_pi((__m64 *)(a+14),C3); 
  PetscFunctionReturn(0);
}

/*
    Version that uses Intel Compiler intrinsic functions and PentiumIII SSE registers
*/
/*
     Iterated matrix updates of the form:
            C_(i) = C_(i) - A * B_(i) {for i = 1, ..., N}

     Intended to be used within a (4x4) Block LU-Factorization procedure.

     Inputs: N, A, B, C, offset
     N iterations will be performed

     A, B, and C are 4x4 matrices stored columnwise
     B(i+1) is adjacent to B(i) in memory:
        &B(i+1) = &B(i) + 16;
     C(i+1) is offset from C(i) according to offset:
        &C(i) = C + 16 * offset(i);

     Output: C

     NOTE: A, B, and C must be allocated as 16-byte aligned data
    
*/ 
int Kernel_LU_Row_Update_4_SSE(int N,float *a,float *b,float *cc,int *offset) 
{ 
  __m128 A0,A1,A2,A3,C0,C1,C2,C3; 
  float *c; 
  int    i; 
  
  PetscFunctionBegin;
/*    A0 = _mm_load_ps(a   ); */ 
/*    A1 = _mm_load_ps(a+4 ); */ 
/*    A2 = _mm_load_ps(a+8 ); */ 
/*    A3 = _mm_load_ps(a+12); */ 
  A0 = _mm_loadh_pi(_mm_loadl_pi(A0,(__m64 *)(a   )),(__m64 *)(a+2 )); 
  A1 = _mm_loadh_pi(_mm_loadl_pi(A1,(__m64 *)(a+4 )),(__m64 *)(a+6 )); 
  A2 = _mm_loadh_pi(_mm_loadl_pi(A2,(__m64 *)(a+8 )),(__m64 *)(a+10)); 
  A3 = _mm_loadh_pi(_mm_loadl_pi(A3,(__m64 *)(a+12)),(__m64 *)(a+14)); 
  
  for (i=0;i<N;i++) { 
    /* Get pointer to C(i) */  
    c = cc + 16*offset[i]; 
    
    C0 = _mm_sub_ps(_mm_mul_ps(A0,_mm_load_ps1(b  )),_mm_load_ps(c)); 
    C0 = _mm_sub_ps(_mm_mul_ps(A1,_mm_load_ps1(b+1)),C0); 
    C0 = _mm_sub_ps(_mm_mul_ps(A2,_mm_load_ps1(b+2)),C0); 
    C0 = _mm_sub_ps(_mm_mul_ps(A3,_mm_load_ps1(b+3)),C0); 
    
/*      _mm_store_ps(c,C0); */ 
    _mm_storel_pi((__m64 *)(c  ),C0); 
    _mm_storeh_pi((__m64 *)(c+2),C0); 
    
    C1 = _mm_sub_ps(_mm_mul_ps(A0,_mm_load_ps1(b+4)),_mm_load_ps(c+4)); 
    C1 = _mm_sub_ps(_mm_mul_ps(A1,_mm_load_ps1(b+5)),C1); 
    C1 = _mm_sub_ps(_mm_mul_ps(A2,_mm_load_ps1(b+6)),C1); 
    C1 = _mm_sub_ps(_mm_mul_ps(A3,_mm_load_ps1(b+7)),C1); 
    
/*      _mm_store_ps(c+4,C1); */ 
    _mm_storel_pi((__m64 *)(c+4),C1); 
    _mm_storeh_pi((__m64 *)(c+6),C1); 
    
    C2 = _mm_sub_ps(_mm_mul_ps(A0,_mm_load_ps1(b+8 )),_mm_load_ps(c+8)); 
    C2 = _mm_sub_ps(_mm_mul_ps(A1,_mm_load_ps1(b+9 )),C2); 
    C2 = _mm_sub_ps(_mm_mul_ps(A2,_mm_load_ps1(b+10)),C2); 
    C2 = _mm_sub_ps(_mm_mul_ps(A3,_mm_load_ps1(b+11)),C2); 
    
/*      _mm_store_ps(c+8,C2); */ 
    _mm_storel_pi((__m64 *)(c+8 ),C2); 
    _mm_storeh_pi((__m64 *)(c+10),C2); 
    
    C3 = _mm_sub_ps(_mm_mul_ps(A0,_mm_load_ps1(b+12)),_mm_load_ps(c+12)); 
    C3 = _mm_sub_ps(_mm_mul_ps(A1,_mm_load_ps1(b+13)),C3); 
    C3 = _mm_sub_ps(_mm_mul_ps(A2,_mm_load_ps1(b+14)),C3); 
    C3 = _mm_sub_ps(_mm_mul_ps(A3,_mm_load_ps1(b+15)),C3); 
    
/*      _mm_store_ps(c+12,C3); */ 
    _mm_storel_pi((__m64 *)(c+12),C3); 
    _mm_storeh_pi((__m64 *)(c+14),C3); 
  } 
  PetscFunctionReturn(0);
}
#endif
/*
      Version for when blocks are 4 by 4 Using natural ordering
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering"
int MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat A,Mat *B)
{
  Mat         C = *B;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ *)C->data;
  int         ierr,i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  int         *ajtmpold,*ajtmp,nz,row;
  int         *diag_offset = b->diag,*ai=a->i,*aj=a->j,*pj;
  MatScalar   *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar   p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  MatScalar   p5,p6,p7,p8,p9,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;
  MatScalar   p10,p11,p12,p13,p14,p15,p16,m10,m11,m12;
  MatScalar   m13,m14,m15,m16;
  MatScalar   *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr = PetscMalloc(16*(n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x = rtmp+16*ajtmp[j]; 
      x[0]  = x[1]  = x[2]  = x[3]  = x[4]  = x[5]  = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = 0.0;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i+1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 16*ai[i];
    for (j=0; j<nz; j++) {
      x    = rtmp+16*ajtmpold[j];
      x[0]  = v[0];  x[1]  = v[1];  x[2]  = v[2];  x[3]  = v[3];
      x[4]  = v[4];  x[5]  = v[5];  x[6]  = v[6];  x[7]  = v[7];  x[8]  = v[8];
      x[9]  = v[9];  x[10] = v[10]; x[11] = v[11]; x[12] = v[12]; x[13] = v[13];
      x[14] = v[14]; x[15] = v[15]; 
      v    += 16;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 16*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];  p9  = pc[8];
      p10 = pc[9];  p11 = pc[10]; p12 = pc[11]; p13 = pc[12]; p14 = pc[13];
      p15 = pc[14]; p16 = pc[15]; 
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 ||
          p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0
          || p16 != 0.0) {
        pv = ba + 16*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
#ifdef PETSC_HAVE_ICL_SSE
        ierr = Kernel_A_gets_A_times_B_4SSE(pc,pv);CHKERRQ(ierr);
#else
        x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];  x9  = pv[8];
        x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12]; x14 = pv[13];
        x15 = pv[14]; x16 = pv[15]; 
        pc[0] = m1 = p1*x1 + p5*x2  + p9*x3  + p13*x4;
        pc[1] = m2 = p2*x1 + p6*x2  + p10*x3 + p14*x4;
        pc[2] = m3 = p3*x1 + p7*x2  + p11*x3 + p15*x4;
        pc[3] = m4 = p4*x1 + p8*x2  + p12*x3 + p16*x4;

        pc[4] = m5 = p1*x5 + p5*x6  + p9*x7  + p13*x8;
        pc[5] = m6 = p2*x5 + p6*x6  + p10*x7 + p14*x8;
        pc[6] = m7 = p3*x5 + p7*x6  + p11*x7 + p15*x8;
        pc[7] = m8 = p4*x5 + p8*x6  + p12*x7 + p16*x8;

        pc[8]  = m9  = p1*x9 + p5*x10  + p9*x11  + p13*x12;
        pc[9]  = m10 = p2*x9 + p6*x10  + p10*x11 + p14*x12;
        pc[10] = m11 = p3*x9 + p7*x10  + p11*x11 + p15*x12;
        pc[11] = m12 = p4*x9 + p8*x10  + p12*x11 + p16*x12;

        pc[12] = m13 = p1*x13 + p5*x14  + p9*x15  + p13*x16;
        pc[13] = m14 = p2*x13 + p6*x14  + p10*x15 + p14*x16;
        pc[14] = m15 = p3*x13 + p7*x14  + p11*x15 + p15*x16;
        pc[15] = m16 = p4*x13 + p8*x14  + p12*x15 + p16*x16;
#endif
        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 16;
#ifdef PETSC_HAVE_ICL_SSE
        ierr = Kernel_LU_Row_Update_4SSE(nz,pc,pv,rtmp,pj);CHKERRQ(ierr);
#else
        for (j=0; j<nz; j++) {
          x1   = pv[0];  x2  = pv[1];   x3 = pv[2];  x4  = pv[3];
          x5   = pv[4];  x6  = pv[5];   x7 = pv[6];  x8  = pv[7]; x9 = pv[8];
          x10  = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12];
          x14  = pv[13]; x15 = pv[14]; x16 = pv[15];
          x    = rtmp + 16*pj[j];
          x[0] -= m1*x1 + m5*x2  + m9*x3  + m13*x4;
          x[1] -= m2*x1 + m6*x2  + m10*x3 + m14*x4;
          x[2] -= m3*x1 + m7*x2  + m11*x3 + m15*x4;
          x[3] -= m4*x1 + m8*x2  + m12*x3 + m16*x4;

          x[4] -= m1*x5 + m5*x6  + m9*x7  + m13*x8;
          x[5] -= m2*x5 + m6*x6  + m10*x7 + m14*x8;
          x[6] -= m3*x5 + m7*x6  + m11*x7 + m15*x8;
          x[7] -= m4*x5 + m8*x6  + m12*x7 + m16*x8;

          x[8]  -= m1*x9 + m5*x10 + m9*x11  + m13*x12;
          x[9]  -= m2*x9 + m6*x10 + m10*x11 + m14*x12;
          x[10] -= m3*x9 + m7*x10 + m11*x11 + m15*x12;
          x[11] -= m4*x9 + m8*x10 + m12*x11 + m16*x12;

          x[12] -= m1*x13 + m5*x14  + m9*x15  + m13*x16;
          x[13] -= m2*x13 + m6*x14  + m10*x15 + m14*x16;
          x[14] -= m3*x13 + m7*x14  + m11*x15 + m15*x16;
          x[15] -= m4*x13 + m8*x14  + m12*x15 + m16*x16;

          pv   += 16;
        }
#endif
        PetscLogFlops(128*nz+112);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 16*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x      = rtmp+16*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7]; pv[8] = x[8];
      pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11]; pv[12] = x[12];
      pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15];
      pv   += 16;
    }
    /* invert diagonal block */
    w = ba + 16*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_4(w);CHKERRQ(ierr);
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  C->factor    = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PetscLogFlops(1.3333*64*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
