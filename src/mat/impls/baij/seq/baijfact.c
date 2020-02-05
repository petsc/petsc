
/*
    Factorization code for BAIJ format.
*/
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C     =B;
  Mat_SeqBAIJ    *a    =(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic;
  PetscInt       i,j,k,nz,nzL,row,*pj;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,bs2=a->bs2;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag;
  MatScalar      *rtmp,*pc,*mwork,*pv;
  MatScalar      *aa=a->a,*v;
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
    for (k=0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j] != (PetscScalar)0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = PetscKernel_A_gets_A_times_B_2(pc,pv,mwork);CHKERRQ(ierr);

        pj = b->j + bdiag[row+1]+1; /* begining of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + 4*pj[j];
          ierr = PetscKernel_A_gets_A_minus_B_times_C_2(v,pc,pv);CHKERRQ(ierr);
          pv  += 4;
        }
        ierr = PetscLogFlops(16*nz+12);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
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

    /* Mark diagonal and invert diagonal for simplier triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscArraycpy(pv,rtmp+bs2*pj[0],bs2);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_2(pv,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
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
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_2;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*2*2*2*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C =B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,k,nz,nzL,row,*pj;
  const PetscInt n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,bs2=a->bs2;
  const PetscInt *ajtmp,*bjtmp,*bdiag=b->diag;
  MatScalar      *rtmp,*pc,*mwork,*pv;
  MatScalar      *aa=a->a,*v;
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
    bjtmp = bj + bdiag[i+1]+1;
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
    for (k=0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2*row;
      for (flg=0,j=0; j<bs2; j++) {
        if (pc[j]!=(PetscScalar)0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = PetscKernel_A_gets_A_times_B_2(pc,pv,mwork);CHKERRQ(ierr);

        pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row]-bdiag[row+1] - 1; /* num of entries in U(row,:) excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + 4*pj[j];
          ierr = PetscKernel_A_gets_A_minus_B_times_C_2(v,pc,pv);CHKERRQ(ierr);
          pv  += 4;
        }
        ierr = PetscLogFlops(16*nz+12);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
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

    /* Mark diagonal and invert diagonal for simplier triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    ierr = PetscArraycpy(pv,rtmp+bs2*pj[0],bs2);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A_2(pv,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) B->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    /*
    pv = b->a + bs2*bi[2*n-i];
    pj = b->j + bi[2*n-i];
    nz = bi[2*n-i+1] - bi[2*n-i] - 1;
    */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_2_NaturalOrdering;
  C->ops->forwardsolve   = MatForwardSolve_SeqBAIJ_2_NaturalOrdering;
  C->ops->backwardsolve  = MatBackwardSolve_SeqBAIJ_2_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2_NaturalOrdering;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*2*2*2*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_inplace(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C     = B;
  Mat_SeqBAIJ    *a    = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row;
  PetscInt       *diag_offset=b->diag,idx,*ai=a->i,*aj=a->j,*pj;
  MatScalar      *pv,*v,*rtmp,m1,m2,m3,m4,*pc,*w,*x,x1,x2,x3,x4;
  MatScalar      p1,p2,p3,p4;
  MatScalar      *ba   = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc1(4*(n+1),&rtmp);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x = rtmp+4*ajtmp[j]; x[0] = x[1] = x[2] = x[3] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 4*ai[idx];
    for (j=0; j<nz; j++) {
      x    = rtmp+4*ic[ajtmpold[j]];
      x[0] = v[0]; x[1] = v[1]; x[2] = v[2]; x[3] = v[3];
      v   += 4;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 4*row;
      p1 = pc[0]; p2 = pc[1]; p3 = pc[2]; p4 = pc[3];
      if (p1 != (PetscScalar)0.0 || p2 != (PetscScalar)0.0 || p3 != (PetscScalar)0.0 || p4 != (PetscScalar)0.0) {
        pv    = ba + 4*diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
        pc[0] = m1 = p1*x1 + p3*x2;
        pc[1] = m2 = p2*x1 + p4*x2;
        pc[2] = m3 = p1*x3 + p3*x4;
        pc[3] = m4 = p2*x3 + p4*x4;
        nz    = bi[row+1] - diag_offset[row] - 1;
        pv   += 4;
        for (j=0; j<nz; j++) {
          x1    = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
          x     = rtmp + 4*pj[j];
          x[0] -= m1*x1 + m3*x2;
          x[1] -= m2*x1 + m4*x2;
          x[2] -= m1*x3 + m3*x4;
          x[3] -= m2*x3 + m4*x4;
          pv   += 4;
        }
        ierr = PetscLogFlops(16.0*nz+12.0);CHKERRQ(ierr);
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 4*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x     = rtmp+4*pj[j];
      pv[0] = x[0]; pv[1] = x[1]; pv[2] = x[2]; pv[3] = x[3];
      pv   += 4;
    }
    /* invert diagonal block */
    w    = ba + 4*diag_offset[i];
    ierr = PetscKernel_A_gets_inverse_A_2(w,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_2_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2_inplace;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*8*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
/*
      Version for when blocks are 2 by 2 Using natural ordering
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row;
  PetscInt       *diag_offset = b->diag,*ai=a->i,*aj=a->j,*pj;
  MatScalar      *pv,*v,*rtmp,*pc,*w,*x;
  MatScalar      p1,p2,p3,p4,m1,m2,m3,m4,x1,x2,x3,x4;
  MatScalar      *ba   = b->a,*aa = a->a;
  PetscReal      shift = info->shiftamount;
  PetscBool      allowzeropivot,zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  ierr = PetscMalloc1(4*(n+1),&rtmp);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      x    = rtmp+4*ajtmp[j];
      x[0] = x[1]  = x[2]  = x[3]  = 0.0;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i+1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 4*ai[i];
    for (j=0; j<nz; j++) {
      x    = rtmp+4*ajtmpold[j];
      x[0] = v[0];  x[1]  = v[1];  x[2]  = v[2];  x[3]  = v[3];
      v   += 4;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 4*row;
      p1 = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      if (p1 != (PetscScalar)0.0 || p2 != (PetscScalar)0.0 || p3 != (PetscScalar)0.0 || p4 != (PetscScalar)0.0) {
        pv    = ba + 4*diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        pc[0] = m1 = p1*x1 + p3*x2;
        pc[1] = m2 = p2*x1 + p4*x2;
        pc[2] = m3 = p1*x3 + p3*x4;
        pc[3] = m4 = p2*x3 + p4*x4;
        nz    = bi[row+1] - diag_offset[row] - 1;
        pv   += 4;
        for (j=0; j<nz; j++) {
          x1    = pv[0];  x2  = pv[1];   x3 = pv[2];  x4  = pv[3];
          x     = rtmp + 4*pj[j];
          x[0] -= m1*x1 + m3*x2;
          x[1] -= m2*x1 + m4*x2;
          x[2] -= m1*x3 + m3*x4;
          x[3] -= m2*x3 + m4*x4;
          pv   += 4;
        }
        ierr = PetscLogFlops(16.0*nz+12.0);CHKERRQ(ierr);
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 4*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      x     = rtmp+4*pj[j];
      pv[0] = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      /*
      printf(" col %d:",pj[j]);
      PetscInt j1;
      for (j1=0; j1<4; j1++) printf(" %g,",*(pv+j1));
      printf("\n");
      */
      pv += 4;
    }
    /* invert diagonal block */
    w = ba + 4*diag_offset[i];
    ierr = PetscKernel_A_gets_inverse_A_2(w,shift, allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_2_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*8*b->mbs);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
/*
     Version for when blocks are 1 by 1.
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_1(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat             C     =B;
  Mat_SeqBAIJ     *a    =(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
  IS              isrow = b->row,isicol = b->icol;
  PetscErrorCode  ierr;
  const PetscInt  *r,*ic,*ics;
  const PetscInt  n=a->mbs,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bdiag=b->diag;
  PetscInt        i,j,k,nz,nzL,row,*pj;
  const PetscInt  *ajtmp,*bjtmp;
  MatScalar       *rtmp,*pc,multiplier,*pv;
  const MatScalar *aa=a->a,*v;
  PetscBool       row_identity,col_identity;
  FactorShiftCtx  sctx;
  const PetscInt  *ddiag;
  PetscReal       rs;
  MatScalar       d;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    ddiag          = a->diag;
    sctx.shift_top = info->zeropivot;
    for (i=0; i<n; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (aa)[ddiag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      v  = aa+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) rs += PetscAbsScalar(v[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    sctx.shift_top *= 1.1;
    sctx.nshift_max = 5;
    sctx.shift_lo   = 0.;
    sctx.shift_hi   = 1.;
  }

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+1,&rtmp);CHKERRQ(ierr);
  ics  = ic;

  do {
    sctx.newshift = PETSC_FALSE;
    for (i=0; i<n; i++) {
      /* zero rtmp */
      /* L part */
      nz    = bi[i+1] - bi[i];
      bjtmp = bj + bi[i];
      for  (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;

      /* U part */
      nz    = bdiag[i]-bdiag[i+1];
      bjtmp = bj + bdiag[i+1]+1;
      for  (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;

      /* load in initial (unfactored row) */
      nz    = ai[r[i]+1] - ai[r[i]];
      ajtmp = aj + ai[r[i]];
      v     = aa + ai[r[i]];
      for (j=0; j<nz; j++) rtmp[ics[ajtmp[j]]] = v[j];

      /* ZeropivotApply() */
      rtmp[i] += sctx.shift_amount;  /* shift the diagonal of the matrix */

      /* elimination */
      bjtmp = bj + bi[i];
      row   = *bjtmp++;
      nzL   = bi[i+1] - bi[i];
      for (k=0; k < nzL; k++) {
        pc = rtmp + row;
        if (*pc != (PetscScalar)0.0) {
          pv         = b->a + bdiag[row];
          multiplier = *pc * (*pv);
          *pc        = multiplier;

          pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
          pv = b->a + bdiag[row+1]+1;
          nz = bdiag[row]-bdiag[row+1]-1; /* num of entries in U(row,:) excluding diag */
          for (j=0; j<nz; j++) rtmp[pj[j]] -= multiplier * pv[j];
          ierr = PetscLogFlops(2.0*nz);CHKERRQ(ierr);
        }
        row = *bjtmp++;
      }

      /* finished row so stick it into b->a */
      rs = 0.0;
      /* L part */
      pv = b->a + bi[i];
      pj = b->j + bi[i];
      nz = bi[i+1] - bi[i];
      for (j=0; j<nz; j++) {
        pv[j] = rtmp[pj[j]]; rs += PetscAbsScalar(pv[j]);
      }

      /* U part */
      pv = b->a + bdiag[i+1]+1;
      pj = b->j + bdiag[i+1]+1;
      nz = bdiag[i] - bdiag[i+1]-1;
      for (j=0; j<nz; j++) {
        pv[j] = rtmp[pj[j]]; rs += PetscAbsScalar(pv[j]);
      }

      sctx.rs = rs;
      sctx.pv = rtmp[i];
      ierr    = MatPivotCheck(B,A,info,&sctx,i);CHKERRQ(ierr);
      if (sctx.newshift) break; /* break for-loop */
      rtmp[i] = sctx.pv; /* sctx.pv might be updated in the case of MAT_SHIFT_INBLOCKS */

      /* Mark diagonal and invert diagonal for simplier triangular solves */
      pv  = b->a + bdiag[i];
      *pv = (PetscScalar)1.0/rtmp[i];

    } /* endof for (i=0; i<n; i++) { */

    /* MatPivotRefine() */
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE && !sctx.newshift && sctx.shift_fraction>0 && sctx.nshift<sctx.nshift_max) {
      /*
       * if no shift in this attempt & shifting & started shifting & can refine,
       * then try lower shift
       */
      sctx.shift_hi       = sctx.shift_fraction;
      sctx.shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;
      sctx.shift_amount   = sctx.shift_fraction * sctx.shift_top;
      sctx.newshift       = PETSC_TRUE;
      sctx.nshift++;
    }
  } while (sctx.newshift);

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    C->ops->solve          = MatSolve_SeqBAIJ_1_NaturalOrdering;
    C->ops->forwardsolve   = MatForwardSolve_SeqBAIJ_1_NaturalOrdering;
    C->ops->backwardsolve  = MatBackwardSolve_SeqBAIJ_1_NaturalOrdering;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_1_NaturalOrdering;
  } else {
    C->ops->solve          = MatSolve_SeqBAIJ_1;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_1;
  }
  C->assembled = PETSC_TRUE;
  ierr         = PetscLogFlops(C->cmap->n);CHKERRQ(ierr);

  /* MatShiftView(A,info,&sctx) */
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %g, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,(double)sctx.shift_amount,(double)sctx.shift_fraction,(double)sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS) {
      ierr = PetscInfo2(A,"number of shift_inblocks applied %D, each shift_amount %g\n",sctx.nshift,(double)info->shiftamount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_1_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a    = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row,*ai = a->i,*aj = a->j;
  PetscInt       *diag_offset = b->diag,diag,*pj;
  MatScalar      *pv,*v,*rtmp,multiplier,*pc;
  MatScalar      *ba = b->a,*aa = a->a;
  PetscBool      row_identity, col_identity;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+1,&rtmp);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  (j=0; j<nz; j++) rtmp[ajtmp[j]] = 0.0;

    /* load in initial (unfactored row) */
    nz       = ai[r[i]+1] - ai[r[i]];
    ajtmpold = aj + ai[r[i]];
    v        = aa + ai[r[i]];
    for (j=0; j<nz; j++) rtmp[ic[ajtmpold[j]]] =  v[j];

    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + row;
      if (*pc != 0.0) {
        pv         = ba + diag_offset[row];
        pj         = bj + diag_offset[row] + 1;
        multiplier = *pc * *pv++;
        *pc        = multiplier;
        nz         = bi[row+1] - diag_offset[row] - 1;
        for (j=0; j<nz; j++) rtmp[pj[j]] -= multiplier * pv[j];
        ierr = PetscLogFlops(1.0+2.0*nz);CHKERRQ(ierr);
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) pv[j] = rtmp[pj[j]];
    diag = diag_offset[i] - bi[i];
    /* check pivot entry for current row */
    if (pv[diag] == 0.0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot: row in original ordering %D in permuted ordering %D",r[i],i);
    pv[diag] = 1.0/pv[diag];
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    C->ops->solve          = MatSolve_SeqBAIJ_1_NaturalOrdering_inplace;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_1_NaturalOrdering_inplace;
  } else {
    C->ops->solve          = MatSolve_SeqBAIJ_1_inplace;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_1_inplace;
  }
  C->assembled = PETSC_TRUE;
  ierr         = PetscLogFlops(C->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqbaij_petsc(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (A->hermitian && (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian Factor is not supported");
#endif
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    ierr = MatSetType(*B,MATSEQBAIJ);CHKERRQ(ierr);

    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqBAIJ;
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqBAIJ;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    ierr = MatSetType(*B,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*B,A->rmap->bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqBAIJ;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqBAIJ;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  (*B)->factortype = ftype;

  ierr = PetscFree((*B)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&(*B)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
PetscErrorCode MatLUFactor_SeqBAIJ(Mat A,IS row,IS col,const MatFactorInfo *info)
{
  PetscErrorCode ierr;
  Mat            C;

  PetscFunctionBegin;
  ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&C);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(C,A,row,col,info);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(C,A,info);CHKERRQ(ierr);

  A->ops->solve          = C->ops->solve;
  A->ops->solvetranspose = C->ops->solvetranspose;

  ierr = MatHeaderMerge(A,&C);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)((Mat_SeqBAIJ*)(A->data))->icol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/sbaij/seq/sbaij.h>
PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N(Mat C,Mat A,const MatFactorInfo *info)
{
  PetscErrorCode ierr;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data;
  Mat_SeqSBAIJ   *b=(Mat_SeqSBAIJ*)C->data;
  IS             ip=b->row;
  const PetscInt *rip;
  PetscInt       i,j,mbs=a->mbs,bs=A->rmap->bs,*bi=b->i,*bj=b->j,*bcol;
  PetscInt       *ai=a->i,*aj=a->j;
  PetscInt       k,jmin,jmax,*jl,*il,col,nexti,ili,nz;
  MatScalar      *rtmp,*ba=b->a,*bval,*aa=a->a,dk,uikdi;
  PetscReal      rs;
  FactorShiftCtx sctx;

  PetscFunctionBegin;
  if (bs > 1) { /* convert A to a SBAIJ matrix and apply Cholesky factorization from it */
    if (!a->sbaijMat) {
      ierr = MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&a->sbaijMat);CHKERRQ(ierr);
    }
    ierr = (a->sbaijMat)->ops->choleskyfactornumeric(C,a->sbaijMat,info);CHKERRQ(ierr);
    ierr = MatDestroy(&a->sbaijMat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  ierr = PetscMalloc3(mbs,&rtmp,mbs,&il,mbs,&jl);CHKERRQ(ierr);

  sctx.shift_amount = 0.;
  sctx.nshift       = 0;
  do {
    sctx.newshift = PETSC_FALSE;
    for (i=0; i<mbs; i++) {
      rtmp[i] = 0.0; jl[i] = mbs; il[0] = 0;
    }

    for (k = 0; k<mbs; k++) {
      bval = ba + bi[k];
      /* initialize k-th row by the perm[k]-th row of A */
      jmin = ai[rip[k]]; jmax = ai[rip[k]+1];
      for (j = jmin; j < jmax; j++) {
        col = rip[aj[j]];
        if (col >= k) { /* only take upper triangular entry */
          rtmp[col] = aa[j];
          *bval++   = 0.0; /* for in-place factorization */
        }
      }

      /* shift the diagonal of the matrix */
      if (sctx.nshift) rtmp[k] += sctx.shift_amount;

      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i  = jl[k]; /* first row to be added to k_th row  */

      while (i < k) {
        nexti = jl[i]; /* next row to be added to k_th row */

        /* compute multiplier, update diag(k) and U(i,k) */
        ili     = il[i]; /* index of first nonzero element in U(i,k:bms-1) */
        uikdi   = -ba[ili]*ba[bi[i]]; /* diagonal(k) */
        dk     += uikdi*ba[ili];
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row */
        jmin = ili + 1; jmax = bi[i+1];
        if (jmin < jmax) {
          for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];
          /* update il and jl for row i */
          il[i] = jmin;
          j     = bj[jmin]; jl[i] = jl[j]; jl[j] = i;
        }
        i = nexti;
      }

      /* shift the diagonals when zero pivot is detected */
      /* compute rs=sum of abs(off-diagonal) */
      rs   = 0.0;
      jmin = bi[k]+1;
      nz   = bi[k+1] - jmin;
      if (nz) {
        bcol = bj + jmin;
        while (nz--) {
          rs += PetscAbsScalar(rtmp[*bcol]);
          bcol++;
        }
      }

      sctx.rs = rs;
      sctx.pv = dk;
      ierr    = MatPivotCheck(C,A,info,&sctx,k);CHKERRQ(ierr);
      if (sctx.newshift) break;
      dk = sctx.pv;

      /* copy data into U(k,:) */
      ba[bi[k]] = 1.0/dk; /* U(k,k) */
      jmin      = bi[k]+1; jmax = bi[k+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++) {
          col = bj[j]; ba[j] = rtmp[col]; rtmp[col] = 0.0;
        }
        /* add the k-th row into il and jl */
        il[k] = jmin;
        i     = bj[jmin]; jl[k] = jl[i]; jl[i] = k;
      }
    }
  } while (sctx.newshift);
  ierr = PetscFree3(rtmp,il,jl);CHKERRQ(ierr);

  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);

  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;

  ierr = PetscLogFlops(C->rmap->N);CHKERRQ(ierr);
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo2(A,"number of shiftpd tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shiftnz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data;
  Mat_SeqSBAIJ   *b=(Mat_SeqSBAIJ*)C->data;
  PetscErrorCode ierr;
  PetscInt       i,j,am=a->mbs;
  PetscInt       *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscInt       k,jmin,*jl,*il,nexti,ili,*acol,*bcol,nz;
  MatScalar      *rtmp,*ba=b->a,*aa=a->a,dk,uikdi,*aval,*bval;
  PetscReal      rs;
  FactorShiftCtx sctx;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  ierr = PetscMalloc3(am,&rtmp,am,&il,am,&jl);CHKERRQ(ierr);

  do {
    sctx.newshift = PETSC_FALSE;
    for (i=0; i<am; i++) {
      rtmp[i] = 0.0; jl[i] = am; il[0] = 0;
    }

    for (k = 0; k<am; k++) {
      /* initialize k-th row with elements nonzero in row perm(k) of A */
      nz   = ai[k+1] - ai[k];
      acol = aj + ai[k];
      aval = aa + ai[k];
      bval = ba + bi[k];
      while (nz--) {
        if (*acol < k) { /* skip lower triangular entries */
          acol++; aval++;
        } else {
          rtmp[*acol++] = *aval++;
          *bval++       = 0.0; /* for in-place factorization */
        }
      }

      /* shift the diagonal of the matrix */
      if (sctx.nshift) rtmp[k] += sctx.shift_amount;

      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i  = jl[k]; /* first row to be added to k_th row  */

      while (i < k) {
        nexti = jl[i]; /* next row to be added to k_th row */
        /* compute multiplier, update D(k) and U(i,k) */
        ili     = il[i]; /* index of first nonzero element in U(i,k:bms-1) */
        uikdi   = -ba[ili]*ba[bi[i]];
        dk     += uikdi*ba[ili];
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row ... */
        jmin = ili + 1;
        nz   = bi[i+1] - jmin;
        if (nz > 0) {
          bcol = bj + jmin;
          bval = ba + jmin;
          while (nz--) rtmp[*bcol++] += uikdi*(*bval++);
          /* update il and jl for i-th row */
          il[i] = jmin;
          j     = bj[jmin]; jl[i] = jl[j]; jl[j] = i;
        }
        i = nexti;
      }

      /* shift the diagonals when zero pivot is detected */
      /* compute rs=sum of abs(off-diagonal) */
      rs   = 0.0;
      jmin = bi[k]+1;
      nz   = bi[k+1] - jmin;
      if (nz) {
        bcol = bj + jmin;
        while (nz--) {
          rs += PetscAbsScalar(rtmp[*bcol]);
          bcol++;
        }
      }

      sctx.rs = rs;
      sctx.pv = dk;
      ierr    = MatPivotCheck(C,A,info,&sctx,k);CHKERRQ(ierr);
      if (sctx.newshift) break;    /* sctx.shift_amount is updated */
      dk = sctx.pv;

      /* copy data into U(k,:) */
      ba[bi[k]] = 1.0/dk;
      jmin      = bi[k]+1;
      nz        = bi[k+1] - jmin;
      if (nz) {
        bcol = bj + jmin;
        bval = ba + jmin;
        while (nz--) {
          *bval++       = rtmp[*bcol];
          rtmp[*bcol++] = 0.0;
        }
        /* add k-th row into il and jl */
        il[k] = jmin;
        i     = bj[jmin]; jl[k] = jl[i]; jl[i] = k;
      }
    }
  } while (sctx.newshift);
  ierr = PetscFree3(rtmp,il,jl);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  ierr = PetscLogFlops(C->rmap->N);CHKERRQ(ierr);
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shiftnz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo2(A,"number of shiftpd tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#include <petscbt.h>
#include <../src/mat/utils/freespace.h>
PetscErrorCode MatICCFactorSymbolic_SeqBAIJ(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  Mat                B;
  PetscErrorCode     ierr;
  PetscBool          perm_identity,missing;
  PetscInt           reallocs=0,i,*ai=a->i,*aj=a->j,am=a->mbs,bs=A->rmap->bs,*ui;
  const PetscInt     *rip;
  PetscInt           jmin,jmax,nzk,k,j,*jl,prow,*il,nextprow;
  PetscInt           nlnk,*lnk,*lnk_lvl=NULL,ncols,ncols_upper,*cols,*cols_lvl,*uj,**uj_ptr,**uj_lvl_ptr;
  PetscReal          fill          =info->fill,levels=info->levels;
  PetscFreeSpaceList free_space    =NULL,current_space=NULL;
  PetscFreeSpaceList free_space_lvl=NULL,current_space_lvl=NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  ierr = MatMissingDiagonal(A,&missing,&i);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",i);

  if (bs > 1) {
    if (!a->sbaijMat) {
      ierr = MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&a->sbaijMat);CHKERRQ(ierr);
    }
    (fact)->ops->iccfactorsymbolic = MatICCFactorSymbolic_SeqSBAIJ;  /* undue the change made in MatGetFactor_seqbaij_petsc */

    ierr = MatICCFactorSymbolic(fact,a->sbaijMat,perm,info);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  /* special case that simply copies fill pattern */
  if (!levels && perm_identity) {
    ierr = PetscMalloc1(am+1,&ui);CHKERRQ(ierr);
    for (i=0; i<am; i++) ui[i] = ai[i+1] - a->diag[i]; /* ui: rowlengths - changes when !perm_identity */
    B    = fact;
    ierr = MatSeqSBAIJSetPreallocation(B,1,0,ui);CHKERRQ(ierr);


    b  = (Mat_SeqSBAIJ*)B->data;
    uj = b->j;
    for (i=0; i<am; i++) {
      aj = a->j + a->diag[i];
      for (j=0; j<ui[i]; j++) *uj++ = *aj++;
      b->ilen[i] = ui[i];
    }
    ierr = PetscFree(ui);CHKERRQ(ierr);

    B->factortype = MAT_FACTOR_NONE;

    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    B->factortype = MAT_FACTOR_ICC;

    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering;
    PetscFunctionReturn(0);
  }

  /* initialization */
  ierr  = PetscMalloc1(am+1,&ui);CHKERRQ(ierr);
  ui[0] = 0;
  ierr  = PetscMalloc1(2*am+1,&cols_lvl);CHKERRQ(ierr);

  /* jl: linked list for storing indices of the pivot rows
     il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
  ierr = PetscMalloc4(am,&uj_ptr,am,&uj_lvl_ptr,am,&il,am,&jl);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    jl[i] = am; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = am + 1;
  ierr = PetscIncompleteLLCreate(am,am,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is fill*(ai[am]+am)/2 */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am]/2,am/2)),&free_space);CHKERRQ(ierr);

  current_space = free_space;

  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am]/2,am/2)),&free_space_lvl);CHKERRQ(ierr);
  current_space_lvl = free_space_lvl;

  for (k=0; k<am; k++) {  /* for each active row k */
    /* initialize lnk by the column indices of row rip[k] of A */
    nzk         = 0;
    ncols       = ai[rip[k]+1] - ai[rip[k]];
    ncols_upper = 0;
    cols        = cols_lvl + am;
    for (j=0; j<ncols; j++) {
      i = rip[*(aj + ai[rip[k]] + j)];
      if (i >= k) { /* only take upper triangular entry */
        cols[ncols_upper]     = i;
        cols_lvl[ncols_upper] = -1;  /* initialize level for nonzero entries */
        ncols_upper++;
      }
    }
    ierr = PetscIncompleteLLAdd(ncols_upper,cols,levels,cols_lvl,am,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);
    nzk += nlnk;

    /* update lnk by computing fill-in for each pivot row to be merged in */
    prow = jl[k]; /* 1st pivot row */

    while (prow < k) {
      nextprow = jl[prow];

      /* merge prow into k-th row */
      jmin  = il[prow] + 1; /* index of the 2nd nzero entry in U(prow,k:am-1) */
      jmax  = ui[prow+1];
      ncols = jmax-jmin;
      i     = jmin - ui[prow];
      cols  = uj_ptr[prow] + i; /* points to the 2nd nzero entry in U(prow,k:am-1) */
      for (j=0; j<ncols; j++) cols_lvl[j] = *(uj_lvl_ptr[prow] + i + j);
      ierr = PetscIncompleteLLAddSorted(ncols,cols,levels,cols_lvl,am,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);
      nzk += nlnk;

      /* update il and jl for prow */
      if (jmin < jmax) {
        il[prow] = jmin;

        j = *cols; jl[prow] = jl[j]; jl[j] = prow;
      }
      prow = nextprow;
    }

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzk) {
      i    = am - k + 1; /* num of unfactored rows */
      i    = PetscMin(PetscIntMultTruncate(i,nzk), PetscIntMultTruncate(i,i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
      ierr = PetscFreeSpaceGet(i,&current_space_lvl);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free_space and free_space_lvl, then initialize lnk */
    ierr = PetscIncompleteLLClean(am,am,nzk,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt);CHKERRQ(ierr);

    /* add the k-th row into il and jl */
    if (nzk-1 > 0) {
      i     = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:am-1) */
      jl[k] = jl[i]; jl[i] = k;
      il[k] = ui[k] + 1;
    }
    uj_ptr[k]     = current_space->array;
    uj_lvl_ptr[k] = current_space_lvl->array;

    current_space->array           += nzk;
    current_space->local_used      += nzk;
    current_space->local_remaining -= nzk;

    current_space_lvl->array           += nzk;
    current_space_lvl->local_used      += nzk;
    current_space_lvl->local_remaining -= nzk;

    ui[k+1] = ui[k] + nzk;
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree4(uj_ptr,uj_lvl_ptr,il,jl);CHKERRQ(ierr);
  ierr = PetscFree(cols_lvl);CHKERRQ(ierr);

  /* copy free_space into uj and free free_space; set uj in new datastructure; */
  ierr = PetscMalloc1(ui[am]+1,&uj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,uj);CHKERRQ(ierr);
  ierr = PetscIncompleteLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space_lvl);CHKERRQ(ierr);

  /* put together the new matrix in MATSEQSBAIJ format */
  B    = fact;
  ierr = MatSeqSBAIJSetPreallocation(B,1,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

  b                = (Mat_SeqSBAIJ*)B->data;
  b->singlemalloc  = PETSC_FALSE;
  b->free_a        = PETSC_TRUE;
  b->free_ij       = PETSC_TRUE;

  ierr = PetscMalloc1(ui[am]+1,&b->a);CHKERRQ(ierr);

  b->j             = uj;
  b->i             = ui;
  b->diag          = 0;
  b->ilen          = 0;
  b->imax          = 0;
  b->row           = perm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);

  b->icol = perm;

  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr    = PetscMalloc1(am+1,&b->solve_work);CHKERRQ(ierr);
  ierr    = PetscLogObjectMemory((PetscObject)B,(ui[am]-am)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);

  b->maxnz = b->nz = ui[am];

  B->info.factor_mallocs   = reallocs;
  B->info.fill_ratio_given = fill;
  if (ai[am] != 0.) {
    /* nonzeros in lower triangular part of A (includign diagonals)= (ai[am]+am)/2 */
    B->info.fill_ratio_needed = ((PetscReal)2*ui[am])/(ai[am]+am);
  } else {
    B->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[am] != 0) {
    PetscReal af = B->info.fill_ratio_needed;
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %g or use \n",(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix\n");CHKERRQ(ierr);
  }
#endif
  if (perm_identity) {
    B->ops->solve                 = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    B->ops->solvetranspose        = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering;
  } else {
    (fact)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJ_N;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_SeqBAIJ(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  Mat                B;
  PetscErrorCode     ierr;
  PetscBool          perm_identity,missing;
  PetscReal          fill = info->fill;
  const PetscInt     *rip;
  PetscInt           i,mbs=a->mbs,bs=A->rmap->bs,*ai=a->i,*aj=a->j,reallocs=0,prow;
  PetscInt           *jl,jmin,jmax,nzk,*ui,k,j,*il,nextprow;
  PetscInt           nlnk,*lnk,ncols,ncols_upper,*cols,*uj,**ui_ptr,*uj_ptr;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  if (bs > 1) { /* convert to seqsbaij */
    if (!a->sbaijMat) {
      ierr = MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&a->sbaijMat);CHKERRQ(ierr);
    }
    (fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJ; /* undue the change made in MatGetFactor_seqbaij_petsc */

    ierr = MatCholeskyFactorSymbolic(fact,a->sbaijMat,perm,info);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = MatMissingDiagonal(A,&missing,&i);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",i);

  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  if (!perm_identity) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported");
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  /* initialization */
  ierr  = PetscMalloc1(mbs+1,&ui);CHKERRQ(ierr);
  ui[0] = 0;

  /* jl: linked list for storing indices of the pivot rows
     il: il[i] points to the 1st nonzero entry of U(i,k:mbs-1) */
  ierr = PetscMalloc4(mbs,&ui_ptr,mbs,&il,mbs,&jl,mbs,&cols);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = mbs + 1;
  ierr = PetscLLCreate(mbs,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is fill* (ai[mbs]+mbs)/2 */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[mbs]/2,mbs/2)),&free_space);CHKERRQ(ierr);

  current_space = free_space;

  for (k=0; k<mbs; k++) {  /* for each active row k */
    /* initialize lnk by the column indices of row rip[k] of A */
    nzk         = 0;
    ncols       = ai[rip[k]+1] - ai[rip[k]];
    ncols_upper = 0;
    for (j=0; j<ncols; j++) {
      i = rip[*(aj + ai[rip[k]] + j)];
      if (i >= k) { /* only take upper triangular entry */
        cols[ncols_upper] = i;
        ncols_upper++;
      }
    }
    ierr = PetscLLAdd(ncols_upper,cols,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzk += nlnk;

    /* update lnk by computing fill-in for each pivot row to be merged in */
    prow = jl[k]; /* 1st pivot row */

    while (prow < k) {
      nextprow = jl[prow];
      /* merge prow into k-th row */
      jmin   = il[prow] + 1; /* index of the 2nd nzero entry in U(prow,k:mbs-1) */
      jmax   = ui[prow+1];
      ncols  = jmax-jmin;
      uj_ptr = ui_ptr[prow] + jmin - ui[prow]; /* points to the 2nd nzero entry in U(prow,k:mbs-1) */
      ierr   = PetscLLAddSorted(ncols,uj_ptr,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      nzk   += nlnk;

      /* update il and jl for prow */
      if (jmin < jmax) {
        il[prow] = jmin;
        j        = *uj_ptr;
        jl[prow] = jl[j];
        jl[j]    = prow;
      }
      prow = nextprow;
    }

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzk) {
      i    = mbs - k + 1; /* num of unfactored rows */
      i    = PetscMin(PetscIntMultTruncate(i,nzk), PetscIntMultTruncate(i,i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(mbs,mbs,nzk,lnk,current_space->array,lnkbt);CHKERRQ(ierr);

    /* add the k-th row into il and jl */
    if (nzk-1 > 0) {
      i     = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:mbs-1) */
      jl[k] = jl[i]; jl[i] = k;
      il[k] = ui[k] + 1;
    }
    ui_ptr[k]                       = current_space->array;
    current_space->array           += nzk;
    current_space->local_used      += nzk;
    current_space->local_remaining -= nzk;

    ui[k+1] = ui[k] + nzk;
  }

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = PetscFree4(ui_ptr,il,jl,cols);CHKERRQ(ierr);

  /* copy free_space into uj and free free_space; set uj in new datastructure; */
  ierr = PetscMalloc1(ui[mbs]+1,&uj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,uj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* put together the new matrix in MATSEQSBAIJ format */
  B    = fact;
  ierr = MatSeqSBAIJSetPreallocation(B,bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

  b               = (Mat_SeqSBAIJ*)B->data;
  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;

  ierr = PetscMalloc1(ui[mbs]+1,&b->a);CHKERRQ(ierr);

  b->j             = uj;
  b->i             = ui;
  b->diag          = 0;
  b->ilen          = 0;
  b->imax          = 0;
  b->row           = perm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  ierr     = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  b->icol  = perm;
  ierr     = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr     = PetscMalloc1(mbs+1,&b->solve_work);CHKERRQ(ierr);
  ierr     = PetscLogObjectMemory((PetscObject)B,(ui[mbs]-mbs)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = ui[mbs];

  B->info.factor_mallocs   = reallocs;
  B->info.fill_ratio_given = fill;
  if (ai[mbs] != 0.) {
    /* nonzeros in lower triangular part of A = (ai[mbs]+mbs)/2 */
    B->info.fill_ratio_needed = ((PetscReal)2*ui[mbs])/(ai[mbs]+mbs);
  } else {
    B->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[mbs] != 0.) {
    PetscReal af = B->info.fill_ratio_needed;
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %g or use \n",(double)af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix\n");CHKERRQ(ierr);
  }
#endif
  if (perm_identity) {
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering;
  } else {
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJ_N;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_N_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  const PetscInt    *ai=a->i,*aj=a->j,*adiag=a->diag,*vi;
  PetscInt          i,k,n=a->mbs;
  PetscInt          nz,bs=A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,*s,*t,*ls;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  /* forward solve the lower triangular */
  ierr = PetscArraycpy(t,b,bs);CHKERRQ(ierr); /* copy 1st block of b to t */

  for (i=1; i<n; i++) {
    v    = aa + bs2*ai[i];
    vi   = aj + ai[i];
    nz   = ai[i+1] - ai[i];
    s    = t + bs*i;
    ierr = PetscArraycpy(s,b+bs*i,bs);CHKERRQ(ierr); /* copy i_th block of b to t */
    for (k=0;k<nz;k++) {
      PetscKernel_v_gets_v_minus_A_times_w(bs,s,v,t+bs*vi[k]);
      v += bs2;
    }
  }

  /* backward solve the upper triangular */
  ls = a->solve_work + A->cmap->n;
  for (i=n-1; i>=0; i--) {
    v    = aa + bs2*(adiag[i+1]+1);
    vi   = aj + adiag[i+1]+1;
    nz   = adiag[i] - adiag[i+1]-1;
    ierr = PetscArraycpy(ls,t+i*bs,bs);CHKERRQ(ierr);
    for (k=0; k<nz; k++) {
      PetscKernel_v_gets_v_minus_A_times_w(bs,ls,v,t+bs*vi[k]);
      v += bs2;
    }
    PetscKernel_w_gets_A_times_v(bs,ls,aa+bs2*adiag[i],t+i*bs); /* *inv(diagonal[i]) */
    ierr = PetscArraycpy(x+i*bs,t+i*bs,bs);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(a->bs2)*(a->nz) - A->rmap->bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_N(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ        *a   =(Mat_SeqBAIJ*)A->data;
  IS                 iscol=a->col,isrow=a->row;
  PetscErrorCode     ierr;
  const PetscInt     *r,*c,*rout,*cout,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi;
  PetscInt           i,m,n=a->mbs;
  PetscInt           nz,bs=A->rmap->bs,bs2=a->bs2;
  const MatScalar    *aa=a->a,*v;
  PetscScalar        *x,*s,*t,*ls;
  const PetscScalar  *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* forward solve the lower triangular */
  ierr = PetscArraycpy(t,b+bs*r[0],bs);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    v    = aa + bs2*ai[i];
    vi   = aj + ai[i];
    nz   = ai[i+1] - ai[i];
    s    = t + bs*i;
    ierr = PetscArraycpy(s,b+bs*r[i],bs);CHKERRQ(ierr);
    for (m=0; m<nz; m++) {
      PetscKernel_v_gets_v_minus_A_times_w(bs,s,v,t+bs*vi[m]);
      v += bs2;
    }
  }

  /* backward solve the upper triangular */
  ls = a->solve_work + A->cmap->n;
  for (i=n-1; i>=0; i--) {
    v    = aa + bs2*(adiag[i+1]+1);
    vi   = aj + adiag[i+1]+1;
    nz   = adiag[i] - adiag[i+1] - 1;
    ierr = PetscArraycpy(ls,t+i*bs,bs);CHKERRQ(ierr);
    for (m=0; m<nz; m++) {
      PetscKernel_v_gets_v_minus_A_times_w(bs,ls,v,t+bs*vi[m]);
      v += bs2;
    }
    PetscKernel_w_gets_A_times_v(bs,ls,v,t+i*bs); /* *inv(diagonal[i]) */
    ierr = PetscArraycpy(x + bs*c[i],t+i*bs,bs);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(a->bs2)*(a->nz) - A->rmap->bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    For each block in an block array saves the largest absolute value in the block into another array
*/
static PetscErrorCode MatBlockAbs_private(PetscInt nbs,PetscInt bs2,PetscScalar *blockarray,PetscReal *absarray)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscArrayzero(absarray,nbs+1);CHKERRQ(ierr);
  for (i=0; i<nbs; i++) {
    for (j=0; j<bs2; j++) {
      if (absarray[i] < PetscAbsScalar(blockarray[i*nbs+j])) absarray[i] = PetscAbsScalar(blockarray[i*nbs+j]);
    }
  }
  PetscFunctionReturn(0);
}

/*
     This needs to be renamed and called by the regular MatILUFactor_SeqBAIJ when drop tolerance is used
*/
PetscErrorCode MatILUDTFactor_SeqBAIJ(Mat A,IS isrow,IS iscol,const MatFactorInfo *info,Mat *fact)
{
  Mat            B = *fact;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b;
  IS             isicol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic;
  PetscInt       i,mbs=a->mbs,bs=A->rmap->bs,bs2=a->bs2,*ai=a->i,*aj=a->j,*ajtmp,*adiag;
  PetscInt       *bi,*bj,*bdiag;

  PetscInt  row,nzi,nzi_bl,nzi_bu,*im,dtcount,nzi_al,nzi_au;
  PetscInt  nlnk,*lnk;
  PetscBT   lnkbt;
  PetscBool row_identity,icol_identity;
  MatScalar *aatmp,*pv,*batmp,*ba,*rtmp,*pc,*multiplier,*vtmp;
  PetscInt  j,nz,*pj,*bjtmp,k,ncut,*jtmp;

  PetscReal dt=info->dt;          /* shift=info->shiftamount; */
  PetscInt  nnz_max;
  PetscBool missing;
  PetscReal *vtmp_abs;
  MatScalar *v_work;
  PetscInt  *v_pivots;
  PetscBool allowzeropivot,zeropivotdetected=PETSC_FALSE;

  PetscFunctionBegin;
  /* ------- symbolic factorization, can be reused ---------*/
  ierr = MatMissingDiagonal(A,&missing,&i);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",i);
  adiag=a->diag;

  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);

  /* bdiag is location of diagonal in factor */
  ierr = PetscMalloc1(mbs+1,&bdiag);CHKERRQ(ierr);

  /* allocate row pointers bi */
  ierr = PetscMalloc1(2*mbs+2,&bi);CHKERRQ(ierr);

  /* allocate bj and ba; max num of nonzero entries is (ai[n]+2*n*dtcount+2) */
  dtcount = (PetscInt)info->dtcount;
  if (dtcount > mbs-1) dtcount = mbs-1;
  nnz_max = ai[mbs]+2*mbs*dtcount +2;
  /* printf("MatILUDTFactor_SeqBAIJ, bs %d, ai[mbs] %d, nnz_max  %d, dtcount %d\n",bs,ai[mbs],nnz_max,dtcount); */
  ierr    = PetscMalloc1(nnz_max,&bj);CHKERRQ(ierr);
  nnz_max = nnz_max*bs2;
  ierr    = PetscMalloc1(nnz_max,&ba);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSeqBAIJSetPreallocation(B,bs,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)isicol);CHKERRQ(ierr);

  b               = (Mat_SeqBAIJ*)(B)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;

  b->a    = ba;
  b->j    = bj;
  b->i    = bi;
  b->diag = bdiag;
  b->ilen = 0;
  b->imax = 0;
  b->row  = isrow;
  b->col  = iscol;

  ierr = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);

  b->icol  = isicol;
  ierr     = PetscMalloc1(bs*(mbs+1),&b->solve_work);CHKERRQ(ierr);
  ierr     = PetscLogObjectMemory((PetscObject)B,nnz_max*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = nnz_max/bs2;

  (B)->factortype            = MAT_FACTOR_ILUDT;
  (B)->info.factor_mallocs   = 0;
  (B)->info.fill_ratio_given = ((PetscReal)nnz_max)/((PetscReal)(ai[mbs]*bs2));
  /* ------- end of symbolic factorization ---------*/
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* linked list for storing column indices of the active row */
  nlnk = mbs + 1;
  ierr = PetscLLCreate(mbs,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* im: used by PetscLLAddSortedLU(); jtmp: working array for column indices of active row */
  ierr = PetscMalloc2(mbs,&im,mbs,&jtmp);CHKERRQ(ierr);
  /* rtmp, vtmp: working arrays for sparse and contiguous row entries of active row */
  ierr = PetscMalloc2(mbs*bs2,&rtmp,mbs*bs2,&vtmp);CHKERRQ(ierr);
  ierr = PetscMalloc1(mbs+1,&vtmp_abs);CHKERRQ(ierr);
  ierr = PetscMalloc3(bs,&v_work,bs2,&multiplier,bs,&v_pivots);CHKERRQ(ierr);

  allowzeropivot = PetscNot(A->erroriffailure);
  bi[0]       = 0;
  bdiag[0]    = (nnz_max/bs2)-1; /* location of diagonal in factor B */
  bi[2*mbs+1] = bdiag[0]+1; /* endof bj and ba array */
  for (i=0; i<mbs; i++) {
    /* copy initial fill into linked list */
    nzi = ai[r[i]+1] - ai[r[i]];
    if (!nzi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",r[i],i);
    nzi_al = adiag[r[i]] - ai[r[i]];
    nzi_au = ai[r[i]+1] - adiag[r[i]] -1;

    /* load in initial unfactored row */
    ajtmp = aj + ai[r[i]];
    ierr  = PetscLLAddPerm(nzi,ajtmp,ic,mbs,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    ierr  = PetscArrayzero(rtmp,mbs*bs2);CHKERRQ(ierr);
    aatmp = a->a + bs2*ai[r[i]];
    for (j=0; j<nzi; j++) {
      ierr = PetscArraycpy(rtmp+bs2*ic[ajtmp[j]],aatmp+bs2*j,bs2);CHKERRQ(ierr);
    }

    /* add pivot rows into linked list */
    row = lnk[mbs];
    while (row < i) {
      nzi_bl = bi[row+1] - bi[row] + 1;
      bjtmp  = bj + bdiag[row+1]+1; /* points to 1st column next to the diagonal in U */
      ierr   = PetscLLAddSortedLU(bjtmp,row,nlnk,lnk,lnkbt,i,nzi_bl,im);CHKERRQ(ierr);
      nzi   += nlnk;
      row    = lnk[row];
    }

    /* copy data from lnk into jtmp, then initialize lnk */
    ierr = PetscLLClean(mbs,mbs,nzi,lnk,jtmp,lnkbt);CHKERRQ(ierr);

    /* numerical factorization */
    bjtmp = jtmp;
    row   = *bjtmp++; /* 1st pivot row */

    while  (row < i) {
      pc = rtmp + bs2*row;
      pv = ba + bs2*bdiag[row]; /* inv(diag) of the pivot row */
      PetscKernel_A_gets_A_times_B(bs,pc,pv,multiplier); /* pc= multiplier = pc*inv(diag[row]) */
      ierr = MatBlockAbs_private(1,bs2,pc,vtmp_abs);CHKERRQ(ierr);
      if (vtmp_abs[0] > dt) { /* apply tolerance dropping rule */
        pj = bj + bdiag[row+1] + 1;         /* point to 1st entry of U(row,:) */
        pv = ba + bs2*(bdiag[row+1] + 1);
        nz = bdiag[row] - bdiag[row+1] - 1;         /* num of entries in U(row,:), excluding diagonal */
        for (j=0; j<nz; j++) {
          PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j);
        }
        /* ierr = PetscLogFlops(bslog*(nz+1.0)-bs);CHKERRQ(ierr); */
      }
      row = *bjtmp++;
    }

    /* copy sparse rtmp into contiguous vtmp; separate L and U part */
    nzi_bl = 0; j = 0;
    while (jtmp[j] < i) { /* L-part. Note: jtmp is sorted */
      ierr = PetscArraycpy(vtmp+bs2*j,rtmp+bs2*jtmp[j],bs2);CHKERRQ(ierr);
      nzi_bl++; j++;
    }
    nzi_bu = nzi - nzi_bl -1;

    while (j < nzi) { /* U-part */
      ierr = PetscArraycpy(vtmp+bs2*j,rtmp+bs2*jtmp[j],bs2);CHKERRQ(ierr);
      j++;
    }

    ierr = MatBlockAbs_private(nzi,bs2,vtmp,vtmp_abs);CHKERRQ(ierr);

    bjtmp = bj + bi[i];
    batmp = ba + bs2*bi[i];
    /* apply level dropping rule to L part */
    ncut = nzi_al + dtcount;
    if (ncut < nzi_bl) {
      ierr = PetscSortSplitReal(ncut,nzi_bl,vtmp_abs,jtmp);CHKERRQ(ierr);
      ierr = PetscSortIntWithScalarArray(ncut,jtmp,vtmp);CHKERRQ(ierr);
    } else {
      ncut = nzi_bl;
    }
    for (j=0; j<ncut; j++) {
      bjtmp[j] = jtmp[j];
      ierr     = PetscArraycpy(batmp+bs2*j,rtmp+bs2*bjtmp[j],bs2);CHKERRQ(ierr);
    }
    bi[i+1] = bi[i] + ncut;
    nzi     = ncut + 1;

    /* apply level dropping rule to U part */
    ncut = nzi_au + dtcount;
    if (ncut < nzi_bu) {
      ierr = PetscSortSplitReal(ncut,nzi_bu,vtmp_abs+nzi_bl+1,jtmp+nzi_bl+1);CHKERRQ(ierr);
      ierr = PetscSortIntWithScalarArray(ncut,jtmp+nzi_bl+1,vtmp+nzi_bl+1);CHKERRQ(ierr);
    } else {
      ncut = nzi_bu;
    }
    nzi += ncut;

    /* mark bdiagonal */
    bdiag[i+1]    = bdiag[i] - (ncut + 1);
    bi[2*mbs - i] = bi[2*mbs - i +1] - (ncut + 1);

    bjtmp  = bj + bdiag[i];
    batmp  = ba + bs2*bdiag[i];
    ierr   = PetscArraycpy(batmp,rtmp+bs2*i,bs2);CHKERRQ(ierr);
    *bjtmp = i;

    bjtmp = bj + bdiag[i+1]+1;
    batmp = ba + (bdiag[i+1]+1)*bs2;

    for (k=0; k<ncut; k++) {
      bjtmp[k] = jtmp[nzi_bl+1+k];
      ierr     = PetscArraycpy(batmp+bs2*k,rtmp+bs2*bjtmp[k],bs2);CHKERRQ(ierr);
    }

    im[i] = nzi; /* used by PetscLLAddSortedLU() */

    /* invert diagonal block for simplier triangular solves - add shift??? */
    batmp = ba + bs2*bdiag[i];

    ierr  = PetscKernel_A_gets_inverse_A(bs,batmp,v_pivots,v_work,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) B->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  } /* for (i=0; i<mbs; i++) */
  ierr = PetscFree3(v_work,multiplier,v_pivots);CHKERRQ(ierr);

  /* printf("end of L %d, beginning of U %d\n",bi[mbs],bdiag[mbs]); */
  if (bi[mbs] >= bdiag[mbs]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"end of L array %d cannot >= the beginning of U array %d",bi[mbs],bdiag[mbs]);

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  ierr = PetscFree2(im,jtmp);CHKERRQ(ierr);
  ierr = PetscFree2(rtmp,vtmp);CHKERRQ(ierr);

  ierr     = PetscLogFlops(bs2*B->cmap->n);CHKERRQ(ierr);
  b->maxnz = b->nz = bi[mbs] + bdiag[0] - bdiag[mbs];

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&icol_identity);CHKERRQ(ierr);
  if (row_identity && icol_identity) {
    B->ops->solve = MatSolve_SeqBAIJ_N_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqBAIJ_N;
  }

  B->ops->solveadd          = 0;
  B->ops->solvetranspose    = 0;
  B->ops->solvetransposeadd = 0;
  B->ops->matsolve          = 0;
  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;
  PetscFunctionReturn(0);
}
