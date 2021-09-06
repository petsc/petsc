
/*
    Factorization code for BAIJ format.
*/
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
      Version for when blocks are 3 by 3
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a    = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row,*ai=a->i,*aj=a->j;
  PetscInt       *diag_offset = b->diag,idx,*pj;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  MatScalar      p5,p6,p7,p8,p9,x5,x6,x7,x8,x9;
  MatScalar      *ba   = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc1(9*(n+1),&rtmp);CHKERRQ(ierr);
  allowzeropivot = PetscNot(A->erroriffailure);

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x    = rtmp + 9*ajtmp[j];
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
      v   += 9;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 9*row;
      p1 = pc[0]; p2 = pc[1]; p3 = pc[2]; p4 = pc[3];
      p5 = pc[4]; p6 = pc[5]; p7 = pc[6]; p8 = pc[7]; p9 = pc[8];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0) {
        pv    = ba + 9*diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
        x5    = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
        pc[0] = m1 = p1*x1 + p4*x2 + p7*x3;
        pc[1] = m2 = p2*x1 + p5*x2 + p8*x3;
        pc[2] = m3 = p3*x1 + p6*x2 + p9*x3;

        pc[3] = m4 = p1*x4 + p4*x5 + p7*x6;
        pc[4] = m5 = p2*x4 + p5*x5 + p8*x6;
        pc[5] = m6 = p3*x4 + p6*x5 + p9*x6;

        pc[6] = m7 = p1*x7 + p4*x8 + p7*x9;
        pc[7] = m8 = p2*x7 + p5*x8 + p8*x9;
        pc[8] = m9 = p3*x7 + p6*x8 + p9*x9;
        nz    = bi[row+1] - diag_offset[row] - 1;
        pv   += 9;
        for (j=0; j<nz; j++) {
          x1    = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
          x5    = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
          x     = rtmp + 9*pj[j];
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
        ierr = PetscLogFlops(54.0*nz+36.0);CHKERRQ(ierr);
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
    w    = ba + 9*diag_offset[i];
    ierr = PetscKernel_A_gets_inverse_A_3(w,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_3_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_3_inplace;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*3*3*3*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* MatLUFactorNumeric_SeqBAIJ_3 -
     copied from MatLUFactorNumeric_SeqBAIJ_N_inplace() and manually re-implemented
       PetscKernel_A_gets_A_times_B()
       PetscKernel_A_gets_A_minus_B_times_C()
       PetscKernel_A_gets_inverse_A()
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C     =B;
  Mat_SeqBAIJ    *a    =(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*pv,*aa=a->a;
  PetscInt       flg;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  allowzeropivot = PetscNot(A->erroriffailure);

  /* generate work space needed by the factorization */
  ierr = PetscMalloc2(bs2*n,&rtmp,bs2,&mwork);CHKERRQ(ierr);
  ierr = PetscArrayzero(rtmp,bs2*n);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      ierr = PetscArrayzero(rtmp+bs2*bjtmp[j],bs2);CHKERRQ(ierr);
    }

    /* U part */
    nz    = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;
    for  (j=0; j<nz; j++) {
      ierr = PetscArrayzero(rtmp+bs2*bjtmp[j],bs2);CHKERRQ(ierr);
    }

    /* load in initial (unfactored row) */
    nz    = ai[r[i]+1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + bs2*ai[r[i]];
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(rtmp+bs2*ic[ajtmp[j]],v+bs2*j,bs2);CHKERRQ(ierr);
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for (k = 0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j]!=0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = PetscKernel_A_gets_A_times_B_3(pc,pv,mwork);CHKERRQ(ierr);

        pj = b->j + bdiag[row+1] + 1; /* beginning of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries in U(row,:) excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          ierr = PetscKernel_A_gets_A_minus_B_times_C_3(v,pc,pv);CHKERRQ(ierr);
          pv  += bs2;
        }
        ierr = PetscLogFlops(54.0*nz+45);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2*bi[i];
    pj = b->j + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2);CHKERRQ(ierr);
    }

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscArraycpy(pv,rtmp+bs2*pj[0],bs2);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_3(pv,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) B->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pj = b->j + bdiag[i+1] + 1;
    pv = b->a + bs2*(bdiag[i+1]+1);
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_3;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_3;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*3*3*3*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row;
  PetscInt       *diag_offset = b->diag,*ai=a->i,*aj=a->j,*pj;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  MatScalar      p5,p6,p7,p8,p9,x5,x6,x7,x8,x9;
  MatScalar      *ba   = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  ierr = PetscMalloc1(9*(n+1),&rtmp);CHKERRQ(ierr);
  allowzeropivot = PetscNot(A->erroriffailure);

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x    = rtmp+9*ajtmp[j];
      x[0] = x[1]  = x[2]  = x[3]  = x[4]  = x[5]  = x[6] = x[7] = x[8] = 0.0;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i+1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 9*ai[i];
    for (j=0; j<nz; j++) {
      x    = rtmp+9*ajtmpold[j];
      x[0] = v[0];  x[1]  = v[1];  x[2]  = v[2];  x[3]  = v[3];
      x[4] = v[4];  x[5]  = v[5];  x[6]  = v[6];  x[7]  = v[7];  x[8]  = v[8];
      v   += 9;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 9*row;
      p1 = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5 = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];  p9  = pc[8];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0) {
        pv    = ba + 9*diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        x5    = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];  x9  = pv[8];
        pc[0] = m1 = p1*x1 + p4*x2 + p7*x3;
        pc[1] = m2 = p2*x1 + p5*x2 + p8*x3;
        pc[2] = m3 = p3*x1 + p6*x2 + p9*x3;

        pc[3] = m4 = p1*x4 + p4*x5 + p7*x6;
        pc[4] = m5 = p2*x4 + p5*x5 + p8*x6;
        pc[5] = m6 = p3*x4 + p6*x5 + p9*x6;

        pc[6] = m7 = p1*x7 + p4*x8 + p7*x9;
        pc[7] = m8 = p2*x7 + p5*x8 + p8*x9;
        pc[8] = m9 = p3*x7 + p6*x8 + p9*x9;

        nz  = bi[row+1] - diag_offset[row] - 1;
        pv += 9;
        for (j=0; j<nz; j++) {
          x1    = pv[0];  x2  = pv[1];   x3 = pv[2];  x4  = pv[3];
          x5    = pv[4];  x6  = pv[5];   x7 = pv[6];  x8  = pv[7]; x9 = pv[8];
          x     = rtmp + 9*pj[j];
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
        ierr = PetscLogFlops(54.0*nz+36.0);CHKERRQ(ierr);
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 9*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x     = rtmp+9*pj[j];
      pv[0] = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4] = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7]; pv[8] = x[8];
      pv   += 9;
    }
    /* invert diagonal block */
    w    = ba + 9*diag_offset[i];
    ierr = PetscKernel_A_gets_inverse_A_3(w,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_3_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_3_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*3*3*3*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/*
  MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering -
    copied from MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering_inplace()
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C =B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,nz,nzL,row;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag,*pj,bs2=a->bs2;
  MatScalar      *rtmp,*pc,*mwork,*v,*pv,*aa=a->a;
  PetscInt       flg;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);

  /* generate work space needed by the factorization */
  ierr = PetscMalloc2(bs2*n,&rtmp,bs2,&mwork);CHKERRQ(ierr);
  ierr = PetscArrayzero(rtmp,bs2*n);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      ierr = PetscArrayzero(rtmp+bs2*bjtmp[j],bs2);CHKERRQ(ierr);
    }

    /* U part */
    nz    = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1] + 1;
    for  (j=0; j<nz; j++) {
      ierr = PetscArrayzero(rtmp+bs2*bjtmp[j],bs2);CHKERRQ(ierr);
    }

    /* load in initial (unfactored row) */
    nz    = ai[i+1] - ai[i];
    ajtmp = aj + ai[i];
    v     = aa + bs2*ai[i];
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(rtmp+bs2*ajtmp[j],v+bs2*j,bs2);CHKERRQ(ierr);
    }

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i+1] - bi[i];
    for (k=0; k<nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j]!=0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = PetscKernel_A_gets_A_times_B_3(pc,pv,mwork);CHKERRQ(ierr);

        pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries in U(row,:) excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          ierr = PetscKernel_A_gets_A_minus_B_times_C_3(v,pc,pv);CHKERRQ(ierr);
          pv  += bs2;
        }
        ierr = PetscLogFlops(54.0*nz+45);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2*bi[i];
    pj = b->j + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2);CHKERRQ(ierr);
    }

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscArraycpy(pv,rtmp+bs2*pj[0],bs2);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_3(pv,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) B->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_3_NaturalOrdering;
  C->ops->forwardsolve   = MatForwardSolve_SeqBAIJ_3_NaturalOrdering;
  C->ops->backwardsolve  = MatBackwardSolve_SeqBAIJ_3_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_3_NaturalOrdering;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*3*3*3*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

