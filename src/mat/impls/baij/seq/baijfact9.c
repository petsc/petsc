
/*
    Factorization code for BAIJ format.
*/
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/blockinvert.h>

/* ------------------------------------------------------------*/
/*
      Version for when blocks are 5 by 5
*/
#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_5_inplace"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS              isrow = b->row,isicol = b->icol;
  PetscErrorCode  ierr;
  const PetscInt  *r,*ic,*bi = b->i,*bj = b->j,*ajtmpold,*ajtmp;
  PetscInt        i,j,n = a->mbs,nz,row,idx,ipvt[5];
  const PetscInt  *diag_offset = b->diag,*ai=a->i,*aj=a->j,*pj;
  MatScalar       *w,*pv,*rtmp,*x,*pc;
  const MatScalar *v,*aa = a->a;
  MatScalar       p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  MatScalar       p5,p6,p7,p8,p9,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;
  MatScalar       x17,x18,x19,x20,x21,x22,x23,x24,x25,p10,p11,p12,p13,p14;
  MatScalar       p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,m10,m11,m12;
  MatScalar       m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25;
  MatScalar       *ba = b->a,work[25];
  PetscReal       shift = info->shiftamount;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc(25*(n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);

#define PETSC_USE_MEMZERO 1
#define PETSC_USE_MEMCPY 1

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
#if defined(PETSC_USE_MEMZERO)
      ierr = PetscMemzero(rtmp+25*ajtmp[j],25*sizeof(PetscScalar));CHKERRQ(ierr);
#else
      x = rtmp+25*ajtmp[j];
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = x[16] = x[17] = 0.0;
      x[18] = x[19] = x[20] = x[21] = x[22] = x[23] = x[24] = 0.0;
#endif
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 25*ai[idx];
    for (j=0; j<nz; j++) {
#if defined(PETSC_USE_MEMCPY)
      ierr = PetscMemcpy(rtmp+25*ic[ajtmpold[j]],v,25*sizeof(PetscScalar));CHKERRQ(ierr);
#else
      x    = rtmp+25*ic[ajtmpold[j]];
      x[0] = v[0]; x[1] = v[1]; x[2] = v[2]; x[3] = v[3];
      x[4] = v[4]; x[5] = v[5]; x[6] = v[6]; x[7] = v[7]; x[8] = v[8];
      x[9] = v[9]; x[10] = v[10]; x[11] = v[11]; x[12] = v[12]; x[13] = v[13];
      x[14] = v[14]; x[15] = v[15]; x[16] = v[16]; x[17] = v[17];
      x[18] = v[18]; x[19] = v[19]; x[20] = v[20]; x[21] = v[21];
      x[22] = v[22]; x[23] = v[23]; x[24] = v[24];
#endif
      v    += 25;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 25*row;
      p1 = pc[0]; p2 = pc[1]; p3 = pc[2]; p4 = pc[3];
      p5 = pc[4]; p6 = pc[5]; p7 = pc[6]; p8 = pc[7]; p9 = pc[8];
      p10 = pc[9]; p11 = pc[10]; p12 = pc[11]; p13 = pc[12]; p14 = pc[13];
      p15 = pc[14]; p16 = pc[15]; p17 = pc[16]; p18 = pc[17]; p19 = pc[18];
      p20 = pc[19]; p21 = pc[20]; p22 = pc[21]; p23 = pc[22]; p24 = pc[23];
      p25 = pc[24];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 ||
          p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0
          || p16 != 0.0 || p17 != 0.0 || p18 != 0.0 || p19 != 0.0 ||
          p20 != 0.0 || p21 != 0.0 || p22 != 0.0 || p23 != 0.0 ||
          p24 != 0.0 || p25 != 0.0) {
        pv = ba + 25*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1 = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
        x5 = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
        x10 = pv[9]; x11 = pv[10]; x12 = pv[11]; x13 = pv[12]; x14 = pv[13];
        x15 = pv[14]; x16 = pv[15]; x17 = pv[16]; x18 = pv[17];
        x19 = pv[18]; x20 = pv[19]; x21 = pv[20]; x22 = pv[21];
        x23 = pv[22]; x24 = pv[23]; x25 = pv[24];
        pc[0] = m1 = p1*x1 + p6*x2  + p11*x3 + p16*x4 + p21*x5;
        pc[1] = m2 = p2*x1 + p7*x2  + p12*x3 + p17*x4 + p22*x5;
        pc[2] = m3 = p3*x1 + p8*x2  + p13*x3 + p18*x4 + p23*x5;
        pc[3] = m4 = p4*x1 + p9*x2  + p14*x3 + p19*x4 + p24*x5;
        pc[4] = m5 = p5*x1 + p10*x2 + p15*x3 + p20*x4 + p25*x5;

        pc[5] = m6 = p1*x6 + p6*x7  + p11*x8 + p16*x9 + p21*x10;
        pc[6] = m7 = p2*x6 + p7*x7  + p12*x8 + p17*x9 + p22*x10;
        pc[7] = m8 = p3*x6 + p8*x7  + p13*x8 + p18*x9 + p23*x10;
        pc[8] = m9 = p4*x6 + p9*x7  + p14*x8 + p19*x9 + p24*x10;
        pc[9] = m10 = p5*x6 + p10*x7 + p15*x8 + p20*x9 + p25*x10;

        pc[10] = m11 = p1*x11 + p6*x12  + p11*x13 + p16*x14 + p21*x15;
        pc[11] = m12 = p2*x11 + p7*x12  + p12*x13 + p17*x14 + p22*x15;
        pc[12] = m13 = p3*x11 + p8*x12  + p13*x13 + p18*x14 + p23*x15;
        pc[13] = m14 = p4*x11 + p9*x12  + p14*x13 + p19*x14 + p24*x15;
        pc[14] = m15 = p5*x11 + p10*x12 + p15*x13 + p20*x14 + p25*x15;

        pc[15] = m16 = p1*x16 + p6*x17  + p11*x18 + p16*x19 + p21*x20;
        pc[16] = m17 = p2*x16 + p7*x17  + p12*x18 + p17*x19 + p22*x20;
        pc[17] = m18 = p3*x16 + p8*x17  + p13*x18 + p18*x19 + p23*x20;
        pc[18] = m19 = p4*x16 + p9*x17  + p14*x18 + p19*x19 + p24*x20;
        pc[19] = m20 = p5*x16 + p10*x17 + p15*x18 + p20*x19 + p25*x20;

        pc[20] = m21 = p1*x21 + p6*x22  + p11*x23 + p16*x24 + p21*x25;
        pc[21] = m22 = p2*x21 + p7*x22  + p12*x23 + p17*x24 + p22*x25;
        pc[22] = m23 = p3*x21 + p8*x22  + p13*x23 + p18*x24 + p23*x25;
        pc[23] = m24 = p4*x21 + p9*x22  + p14*x23 + p19*x24 + p24*x25;
        pc[24] = m25 = p5*x21 + p10*x22 + p15*x23 + p20*x24 + p25*x25;

        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 25;
        for (j=0; j<nz; j++) {
          x1   = pv[0];  x2 = pv[1];   x3  = pv[2];  x4  = pv[3];
          x5   = pv[4];  x6 = pv[5];   x7  = pv[6];  x8  = pv[7]; x9 = pv[8];
          x10  = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12];
          x14  = pv[13]; x15 = pv[14]; x16 = pv[15]; x17 = pv[16];
          x18  = pv[17]; x19 = pv[18]; x20 = pv[19]; x21 = pv[20];
          x22  = pv[21]; x23 = pv[22]; x24 = pv[23]; x25 = pv[24];
          x    = rtmp + 25*pj[j];
          x[0] -= m1*x1 + m6*x2  + m11*x3 + m16*x4 + m21*x5;
          x[1] -= m2*x1 + m7*x2  + m12*x3 + m17*x4 + m22*x5;
          x[2] -= m3*x1 + m8*x2  + m13*x3 + m18*x4 + m23*x5;
          x[3] -= m4*x1 + m9*x2  + m14*x3 + m19*x4 + m24*x5;
          x[4] -= m5*x1 + m10*x2 + m15*x3 + m20*x4 + m25*x5;

          x[5] -= m1*x6 + m6*x7  + m11*x8 + m16*x9 + m21*x10;
          x[6] -= m2*x6 + m7*x7  + m12*x8 + m17*x9 + m22*x10;
          x[7] -= m3*x6 + m8*x7  + m13*x8 + m18*x9 + m23*x10;
          x[8] -= m4*x6 + m9*x7  + m14*x8 + m19*x9 + m24*x10;
          x[9] -= m5*x6 + m10*x7 + m15*x8 + m20*x9 + m25*x10;

          x[10] -= m1*x11 + m6*x12  + m11*x13 + m16*x14 + m21*x15;
          x[11] -= m2*x11 + m7*x12  + m12*x13 + m17*x14 + m22*x15;
          x[12] -= m3*x11 + m8*x12  + m13*x13 + m18*x14 + m23*x15;
          x[13] -= m4*x11 + m9*x12  + m14*x13 + m19*x14 + m24*x15;
          x[14] -= m5*x11 + m10*x12 + m15*x13 + m20*x14 + m25*x15;

          x[15] -= m1*x16 + m6*x17  + m11*x18 + m16*x19 + m21*x20;
          x[16] -= m2*x16 + m7*x17  + m12*x18 + m17*x19 + m22*x20;
          x[17] -= m3*x16 + m8*x17  + m13*x18 + m18*x19 + m23*x20;
          x[18] -= m4*x16 + m9*x17  + m14*x18 + m19*x19 + m24*x20;
          x[19] -= m5*x16 + m10*x17 + m15*x18 + m20*x19 + m25*x20;

          x[20] -= m1*x21 + m6*x22  + m11*x23 + m16*x24 + m21*x25;
          x[21] -= m2*x21 + m7*x22  + m12*x23 + m17*x24 + m22*x25;
          x[22] -= m3*x21 + m8*x22  + m13*x23 + m18*x24 + m23*x25;
          x[23] -= m4*x21 + m9*x22  + m14*x23 + m19*x24 + m24*x25;
          x[24] -= m5*x21 + m10*x22 + m15*x23 + m20*x24 + m25*x25;

          pv   += 25;
        }
        ierr = PetscLogFlops(250.0*nz+225.0);CHKERRQ(ierr);
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 25*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
#if defined(PETSC_USE_MEMCPY)
      ierr = PetscMemcpy(pv,rtmp+25*pj[j],25*sizeof(PetscScalar));CHKERRQ(ierr);
#else
      x     = rtmp+25*pj[j];
      pv[0] = x[0]; pv[1] = x[1]; pv[2] = x[2]; pv[3] = x[3];
      pv[4] = x[4]; pv[5] = x[5]; pv[6] = x[6]; pv[7] = x[7]; pv[8] = x[8];
      pv[9] = x[9]; pv[10] = x[10]; pv[11] = x[11]; pv[12] = x[12];
      pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15]; pv[16] = x[16];
      pv[17] = x[17]; pv[18] = x[18]; pv[19] = x[19]; pv[20] = x[20];
      pv[21] = x[21]; pv[22] = x[22]; pv[23] = x[23]; pv[24] = x[24];
#endif
      pv   += 25;
    }
    /* invert diagonal block */
    w = ba + 25*diag_offset[i];
    ierr = PetscKernel_A_gets_inverse_A_5(w,ipvt,work,shift);CHKERRQ(ierr);
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_5_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_5_inplace;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*5*5*5*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* MatLUFactorNumeric_SeqBAIJ_5 -
     copied from MatLUFactorNumeric_SeqBAIJ_N_inplace() and manually re-implemented
       PetscKernel_A_gets_A_times_B()
       PetscKernel_A_gets_A_minus_B_times_C()
       PetscKernel_A_gets_inverse_A()
*/

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_5"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C=B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ *)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*pv,*aa=a->a,work[25];
  PetscInt       flg,ipvt[5];
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* generate work space needed by the factorization */
  ierr = PetscMalloc2(bs2*n,MatScalar,&rtmp,bs2,MatScalar,&mwork);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*n*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0; i<n; i++){
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* U part */
    nz = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* load in initial (unfactored row) */
    nz    = ai[r[i]+1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + bs2*ai[r[i]];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(rtmp+bs2*ic[ajtmp[j]],v+bs2*j,bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for (k=0;k < nzL;k++) {
      row = bjtmp[k];
      pc = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j]!=0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = PetscKernel_A_gets_A_times_B_5(pc,pv,mwork);CHKERRQ(ierr);

        pj = b->j + bdiag[row+1]+1; /* begining of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          ierr = PetscKernel_A_gets_A_minus_B_times_C_5(v,pc,pv);CHKERRQ(ierr);
          pv  += bs2;
        }
        ierr = PetscLogFlops(250*nz+225);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv   = b->a + bs2*bi[i] ;
    pj   = b->j + bi[i] ;
    nz   = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* Mark diagonal and invert diagonal for simplier triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscMemcpy(pv,rtmp+bs2*pj[0],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    /* ierr = PetscKernel_A_gets_inverse_A(bs,pv,v_pivots,v_work);CHKERRQ(ierr); */
    ierr = PetscKernel_A_gets_inverse_A_5(pv,ipvt,work,shift);CHKERRQ(ierr);

    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++){
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
  }

  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_5;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_5;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*5*5*5*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
      Version for when blocks are 5 by 5 Using natural ordering
*/
#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering_inplace"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j,ipvt[5];
  PetscInt       *ajtmpold,*ajtmp,nz,row;
  PetscInt       *diag_offset = b->diag,*ai=a->i,*aj=a->j,*pj;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15;
  MatScalar      x16,x17,x18,x19,x20,x21,x22,x23,x24,x25;
  MatScalar      p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15;
  MatScalar      p16,p17,p18,p19,p20,p21,p22,p23,p24,p25;
  MatScalar      m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15;
  MatScalar      m16,m17,m18,m19,m20,m21,m22,m23,m24,m25;
  MatScalar      *ba = b->a,*aa = a->a,work[25];
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  ierr = PetscMalloc(25*(n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x = rtmp+25*ajtmp[j];
      x[0]  = x[1]  = x[2]  = x[3]  = x[4]  = x[5]  = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = 0.0;
      x[16] = x[17] = x[18] = x[19] = x[20] = x[21] = x[22] = x[23] = x[24] = 0.0;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i+1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 25*ai[i];
    for (j=0; j<nz; j++) {
      x    = rtmp+25*ajtmpold[j];
      x[0]  = v[0];  x[1]  = v[1];  x[2]  = v[2];  x[3]  = v[3];
      x[4]  = v[4];  x[5]  = v[5];  x[6]  = v[6];  x[7]  = v[7];  x[8]  = v[8];
      x[9]  = v[9];  x[10] = v[10]; x[11] = v[11]; x[12] = v[12]; x[13] = v[13];
      x[14] = v[14]; x[15] = v[15]; x[16] = v[16]; x[17] = v[17]; x[18] = v[18];
      x[19] = v[19]; x[20] = v[20]; x[21] = v[21]; x[22] = v[22]; x[23] = v[23];
      x[24] = v[24];
      v    += 25;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 25*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];  p9  = pc[8];
      p10 = pc[9];  p11 = pc[10]; p12 = pc[11]; p13 = pc[12]; p14 = pc[13];
      p15 = pc[14]; p16 = pc[15]; p17 = pc[16]; p18 = pc[17];
      p19 = pc[18]; p20 = pc[19]; p21 = pc[20]; p22 = pc[21]; p23 = pc[22];
      p24 = pc[23]; p25 = pc[24];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 ||
          p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0
          || p16 != 0.0 || p17 != 0.0 || p18 != 0.0 || p19 != 0.0 || p20 != 0.0
          || p21 != 0.0 || p22 != 0.0 || p23 != 0.0 || p24 != 0.0 || p25 != 0.0) {
        pv = ba + 25*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];  x9  = pv[8];
        x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12]; x14 = pv[13];
        x15 = pv[14]; x16 = pv[15]; x17 = pv[16]; x18 = pv[17]; x19 = pv[18];
        x20 = pv[19]; x21 = pv[20]; x22 = pv[21]; x23 = pv[22]; x24 = pv[23];
        x25 = pv[24];
        pc[0] = m1 = p1*x1 + p6*x2  + p11*x3 + p16*x4 + p21*x5;
        pc[1] = m2 = p2*x1 + p7*x2  + p12*x3 + p17*x4 + p22*x5;
        pc[2] = m3 = p3*x1 + p8*x2  + p13*x3 + p18*x4 + p23*x5;
        pc[3] = m4 = p4*x1 + p9*x2  + p14*x3 + p19*x4 + p24*x5;
        pc[4] = m5 = p5*x1 + p10*x2 + p15*x3 + p20*x4 + p25*x5;

        pc[5]  = m6  = p1*x6 + p6*x7  + p11*x8 + p16*x9 + p21*x10;
        pc[6]  = m7  = p2*x6 + p7*x7  + p12*x8 + p17*x9 + p22*x10;
        pc[7]  = m8  = p3*x6 + p8*x7  + p13*x8 + p18*x9 + p23*x10;
        pc[8]  = m9  = p4*x6 + p9*x7  + p14*x8 + p19*x9 + p24*x10;
        pc[9]  = m10 = p5*x6 + p10*x7 + p15*x8 + p20*x9 + p25*x10;

        pc[10] = m11 = p1*x11 + p6*x12  + p11*x13 + p16*x14 + p21*x15;
        pc[11] = m12 = p2*x11 + p7*x12  + p12*x13 + p17*x14 + p22*x15;
        pc[12] = m13 = p3*x11 + p8*x12  + p13*x13 + p18*x14 + p23*x15;
        pc[13] = m14 = p4*x11 + p9*x12  + p14*x13 + p19*x14 + p24*x15;
        pc[14] = m15 = p5*x11 + p10*x12 + p15*x13 + p20*x14 + p25*x15;

        pc[15] = m16 = p1*x16 + p6*x17  + p11*x18 + p16*x19 + p21*x20;
        pc[16] = m17 = p2*x16 + p7*x17  + p12*x18 + p17*x19 + p22*x20;
        pc[17] = m18 = p3*x16 + p8*x17  + p13*x18 + p18*x19 + p23*x20;
        pc[18] = m19 = p4*x16 + p9*x17  + p14*x18 + p19*x19 + p24*x20;
        pc[19] = m20 = p5*x16 + p10*x17 + p15*x18 + p20*x19 + p25*x20;

        pc[20] = m21 = p1*x21 + p6*x22  + p11*x23 + p16*x24 + p21*x25;
        pc[21] = m22 = p2*x21 + p7*x22  + p12*x23 + p17*x24 + p22*x25;
        pc[22] = m23 = p3*x21 + p8*x22  + p13*x23 + p18*x24 + p23*x25;
        pc[23] = m24 = p4*x21 + p9*x22  + p14*x23 + p19*x24 + p24*x25;
        pc[24] = m25 = p5*x21 + p10*x22 + p15*x23 + p20*x24 + p25*x25;

        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 25;
        for (j=0; j<nz; j++) {
          x1   = pv[0];  x2  = pv[1];   x3 = pv[2];  x4  = pv[3];
          x5   = pv[4];  x6  = pv[5];   x7 = pv[6];  x8  = pv[7]; x9 = pv[8];
          x10  = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12];
          x14  = pv[13]; x15 = pv[14]; x16 = pv[15]; x17 = pv[16]; x18 = pv[17];
          x19 = pv[18];  x20 = pv[19]; x21 = pv[20]; x22 = pv[21]; x23 = pv[22];
          x24 = pv[23];  x25 = pv[24];
          x    = rtmp + 25*pj[j];
          x[0] -= m1*x1 + m6*x2   + m11*x3  + m16*x4 + m21*x5;
          x[1] -= m2*x1 + m7*x2   + m12*x3  + m17*x4 + m22*x5;
          x[2] -= m3*x1 + m8*x2   + m13*x3  + m18*x4 + m23*x5;
          x[3] -= m4*x1 + m9*x2   + m14*x3  + m19*x4 + m24*x5;
          x[4] -= m5*x1 + m10*x2  + m15*x3  + m20*x4 + m25*x5;

          x[5] -= m1*x6 + m6*x7   + m11*x8  + m16*x9 + m21*x10;
          x[6] -= m2*x6 + m7*x7   + m12*x8  + m17*x9 + m22*x10;
          x[7] -= m3*x6 + m8*x7   + m13*x8  + m18*x9 + m23*x10;
          x[8] -= m4*x6 + m9*x7   + m14*x8  + m19*x9 + m24*x10;
          x[9] -= m5*x6 + m10*x7  + m15*x8  + m20*x9 + m25*x10;

          x[10] -= m1*x11 + m6*x12  + m11*x13 + m16*x14 + m21*x15;
          x[11] -= m2*x11 + m7*x12  + m12*x13 + m17*x14 + m22*x15;
          x[12] -= m3*x11 + m8*x12  + m13*x13 + m18*x14 + m23*x15;
          x[13] -= m4*x11 + m9*x12  + m14*x13 + m19*x14 + m24*x15;
          x[14] -= m5*x11 + m10*x12 + m15*x13 + m20*x14 + m25*x15;

          x[15] -= m1*x16 + m6*x17  + m11*x18 + m16*x19 + m21*x20;
          x[16] -= m2*x16 + m7*x17  + m12*x18 + m17*x19 + m22*x20;
          x[17] -= m3*x16 + m8*x17  + m13*x18 + m18*x19 + m23*x20;
          x[18] -= m4*x16 + m9*x17  + m14*x18 + m19*x19 + m24*x20;
          x[19] -= m5*x16 + m10*x17 + m15*x18 + m20*x19 + m25*x20;

          x[20] -= m1*x21 + m6*x22  + m11*x23 + m16*x24 + m21*x25;
          x[21] -= m2*x21 + m7*x22  + m12*x23 + m17*x24 + m22*x25;
          x[22] -= m3*x21 + m8*x22  + m13*x23 + m18*x24 + m23*x25;
          x[23] -= m4*x21 + m9*x22  + m14*x23 + m19*x24 + m24*x25;
          x[24] -= m5*x21 + m10*x22 + m15*x23 + m20*x24 + m25*x25;
          pv   += 25;
        }
        ierr = PetscLogFlops(250.0*nz+225.0);CHKERRQ(ierr);
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 25*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x      = rtmp+25*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7]; pv[8] = x[8];
      pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11]; pv[12] = x[12];
      pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15]; pv[16] = x[16]; pv[17] = x[17];
      pv[18] = x[18]; pv[19] = x[19]; pv[20] = x[20]; pv[21] = x[21]; pv[22] = x[22];
      pv[23] = x[23]; pv[24] = x[24];
      pv   += 25;
    }
    /* invert diagonal block */
    w = ba + 25*diag_offset[i];
    ierr = PetscKernel_A_gets_inverse_A_5(w,ipvt,work,shift);CHKERRQ(ierr);
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_5_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_5_NaturalOrdering_inplace;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*5*5*5*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering"
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C=B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ *)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*vv,*pv,*aa=a->a,work[25];
  PetscInt       flg,ipvt[5];
  PetscReal      shift = info->shiftamount;

  PetscFunctionBegin;
  /* generate work space needed by the factorization */
  ierr = PetscMalloc2(bs2*n,MatScalar,&rtmp,bs2,MatScalar,&mwork);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,bs2*n*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0; i<n; i++){
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* U part */
    nz = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;
    for  (j=0; j<nz; j++){
      ierr = PetscMemzero(rtmp+bs2*bjtmp[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* load in initial (unfactored row) */
    nz    = ai[i+1] - ai[i];
    ajtmp = aj + ai[i];
    v     = aa + bs2*ai[i];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(rtmp+bs2*ajtmp[j],v+bs2*j,bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for (k=0;k < nzL;k++) {
      row = bjtmp[k];
      pc = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j]!=0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = PetscKernel_A_gets_A_times_B_5(pc,pv,mwork);CHKERRQ(ierr);

        pj = b->j + bdiag[row+1]+1; /* begining of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          vv    = rtmp + bs2*pj[j];
          ierr = PetscKernel_A_gets_A_minus_B_times_C_5(vv,pc,pv);CHKERRQ(ierr);
          pv  += bs2;
        }
        ierr = PetscLogFlops(250*nz+225);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv   = b->a + bs2*bi[i] ;
    pj   = b->j + bi[i] ;
    nz   = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }

    /* Mark diagonal and invert diagonal for simplier triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscMemcpy(pv,rtmp+bs2*pj[0],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    /* ierr = PetscKernel_A_gets_inverse_A(bs,pv,v_pivots,v_work);CHKERRQ(ierr); */
    ierr = PetscKernel_A_gets_inverse_A_5(pv,ipvt,work,shift);CHKERRQ(ierr);

    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++){
      ierr = PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(MatScalar));CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);
  C->ops->solve          = MatSolve_SeqBAIJ_5_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_5_NaturalOrdering;
  C->assembled = PETSC_TRUE;
  ierr = PetscLogFlops(1.333333333333*5*5*5*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
