/*
    Factorization code for BAIJ format. 
*/
#include "src/mat/impls/baij/seq/baij.h"
#include "src/inline/ilu.h"

/* ------------------------------------------------------------*/
/*
      Version for when blocks are 3 by 3
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_3"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3(Mat A,MatFactorInfo *info,Mat *B)
{
  Mat            C = *B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  PetscInt       *r,*ic,i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row,*ai=a->i,*aj=a->j;
  PetscInt       *diag_offset = b->diag,idx,*pj;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  MatScalar      p5,p6,p7,p8,p9,x5,x6,x7,x8,x9;
  MatScalar      *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc(9*(n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x = rtmp + 9*ajtmp[j]; 
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 9*ai[idx];
    for (j=0; j<nz; j++) {
      x    = rtmp + 9*ic[ajtmpold[j]];
      x[0] = v[0]; x[1] = v[1]; x[2] = v[2]; x[3] = v[3];
      x[4] = v[4]; x[5] = v[5]; x[6] = v[6]; x[7] = v[7]; x[8] = v[8];
      v    += 9;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 9*row;
      p1 = pc[0]; p2 = pc[1]; p3 = pc[2]; p4 = pc[3];
      p5 = pc[4]; p6 = pc[5]; p7 = pc[6]; p8 = pc[7]; p9 = pc[8];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0) { 
        pv = ba + 9*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1 = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
        x5 = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
        pc[0] = m1 = p1*x1 + p4*x2 + p7*x3;
        pc[1] = m2 = p2*x1 + p5*x2 + p8*x3;
        pc[2] = m3 = p3*x1 + p6*x2 + p9*x3;

        pc[3] = m4 = p1*x4 + p4*x5 + p7*x6;
        pc[4] = m5 = p2*x4 + p5*x5 + p8*x6;
        pc[5] = m6 = p3*x4 + p6*x5 + p9*x6;

        pc[6] = m7 = p1*x7 + p4*x8 + p7*x9;
        pc[7] = m8 = p2*x7 + p5*x8 + p8*x9;
        pc[8] = m9 = p3*x7 + p6*x8 + p9*x9;
        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 9;
        for (j=0; j<nz; j++) {
          x1   = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
          x5   = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
          x    = rtmp + 9*pj[j];
          x[0] -= m1*x1 + m4*x2 + m7*x3;
          x[1] -= m2*x1 + m5*x2 + m8*x3;
          x[2] -= m3*x1 + m6*x2 + m9*x3;
 
          x[3] -= m1*x4 + m4*x5 + m7*x6;
          x[4] -= m2*x4 + m5*x5 + m8*x6;
          x[5] -= m3*x4 + m6*x5 + m9*x6;

          x[6] -= m1*x7 + m4*x8 + m7*x9;
          x[7] -= m2*x7 + m5*x8 + m8*x9;
          x[8] -= m3*x7 + m6*x8 + m9*x9;
          pv   += 9;
        }
        PetscLogFlops(54*nz+36);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 9*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x     = rtmp + 9*pj[j];
      pv[0] = x[0]; pv[1] = x[1]; pv[2] = x[2]; pv[3] = x[3];
      pv[4] = x[4]; pv[5] = x[5]; pv[6] = x[6]; pv[7] = x[7]; pv[8] = x[8];
      pv   += 9;
    }
    /* invert diagonal block */
    w = ba + 9*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_3(w);CHKERRQ(ierr);
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PetscLogFlops(1.3333*27*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
