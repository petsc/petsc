
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
  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(ISGetIndices(isicol,&ic));
  allowzeropivot = PetscNot(A->erroriffailure);

  /* generate work space needed by the factorization */
  PetscCall(PetscMalloc2(bs2*n,&rtmp,bs2,&mwork));
  PetscCall(PetscArrayzero(rtmp,bs2*n));

  for (i=0; i<n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* U part */
    nz    = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* load in initial (unfactored row) */
    nz    = ai[r[i]+1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + bs2*ai[r[i]];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(rtmp+bs2*ic[ajtmp[j]],v+bs2*j,bs2));
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
        PetscCall(PetscKernel_A_gets_A_times_B_2(pc,pv,mwork));

        pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + 4*pj[j];
          PetscCall(PetscKernel_A_gets_A_minus_B_times_C_2(v,pc,pv));
          pv  += 4;
        }
        PetscCall(PetscLogFlops(16.0*nz+12)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2*bi[i];
    pj = b->j + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv,rtmp+bs2*pj[0],bs2));
    PetscCall(PetscKernel_A_gets_inverse_A_2(pv,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) B->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }
  }

  PetscCall(PetscFree2(rtmp,mwork));
  PetscCall(ISRestoreIndices(isicol,&ic));
  PetscCall(ISRestoreIndices(isrow,&r));

  C->ops->solve          = MatSolve_SeqBAIJ_2;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*2*2*2*n)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C =B;
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data,*b=(Mat_SeqBAIJ*)C->data;
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
  PetscCall(PetscMalloc2(bs2*n,&rtmp,bs2,&mwork));
  PetscCall(PetscArrayzero(rtmp,bs2*n));

  for (i=0; i<n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i+1] - bi[i];
    bjtmp = bj + bi[i];
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* U part */
    nz    = bdiag[i] - bdiag[i+1];
    bjtmp = bj + bdiag[i+1]+1;
    for  (j=0; j<nz; j++) {
      PetscCall(PetscArrayzero(rtmp+bs2*bjtmp[j],bs2));
    }

    /* load in initial (unfactored row) */
    nz    = ai[i+1] - ai[i];
    ajtmp = aj + ai[i];
    v     = aa + bs2*ai[i];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(rtmp+bs2*ajtmp[j],v+bs2*j,bs2));
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
        PetscCall(PetscKernel_A_gets_A_times_B_2(pc,pv,mwork));

        pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row]-bdiag[row+1] - 1; /* num of entries in U(row,:) excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + 4*pj[j];
          PetscCall(PetscKernel_A_gets_A_minus_B_times_C_2(v,pc,pv));
          pv  += 4;
        }
        PetscCall(PetscLogFlops(16.0*nz+12)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2*bi[i];
    pj = b->j + bi[i];
    nz = bi[i+1] - bi[i];
    for (j=0; j<nz; j++) {
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv   = b->a + bs2*bdiag[i];
    pj   = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv,rtmp+bs2*pj[0],bs2));
    PetscCall(PetscKernel_A_gets_inverse_A_2(pv,shift,allowzeropivot,&zeropivotdetected));
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
      PetscCall(PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2));
    }
  }
  PetscCall(PetscFree2(rtmp,mwork));

  C->ops->solve          = MatSolve_SeqBAIJ_2_NaturalOrdering;
  C->ops->forwardsolve   = MatForwardSolve_SeqBAIJ_2_NaturalOrdering;
  C->ops->backwardsolve  = MatBackwardSolve_SeqBAIJ_2_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2_NaturalOrdering;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*2*2*2*n)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_inplace(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C     = B;
  Mat_SeqBAIJ    *a    = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
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
  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(ISGetIndices(isicol,&ic));
  PetscCall(PetscMalloc1(4*(n+1),&rtmp));

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
        PetscCall(PetscLogFlops(16.0*nz+12.0));
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
    PetscCall(PetscKernel_A_gets_inverse_A_2(w,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  PetscCall(PetscFree(rtmp));
  PetscCall(ISRestoreIndices(isicol,&ic));
  PetscCall(ISRestoreIndices(isrow,&r));

  C->ops->solve          = MatSolve_SeqBAIJ_2_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2_inplace;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*8*b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
/*
      Version for when blocks are 2 by 2 Using natural ordering
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
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
  PetscCall(PetscMalloc1(4*(n+1),&rtmp));
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
        PetscCall(PetscLogFlops(16.0*nz+12.0));
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
    PetscCall(PetscKernel_A_gets_inverse_A_2(w,shift, allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }

  PetscCall(PetscFree(rtmp));

  C->ops->solve          = MatSolve_SeqBAIJ_2_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_2_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333*8*b->mbs)); /* from inverting diagonal blocks */
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
  PetscCall(PetscMemzero(&sctx,sizeof(FactorShiftCtx)));

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

  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(ISGetIndices(isicol,&ic));
  PetscCall(PetscMalloc1(n+1,&rtmp));
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
          PetscCall(PetscLogFlops(2.0*nz));
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
      PetscCall(MatPivotCheck(B,A,info,&sctx,i));
      if (sctx.newshift) break; /* break for-loop */
      rtmp[i] = sctx.pv; /* sctx.pv might be updated in the case of MAT_SHIFT_INBLOCKS */

      /* Mark diagonal and invert diagonal for simpler triangular solves */
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

  PetscCall(PetscFree(rtmp));
  PetscCall(ISRestoreIndices(isicol,&ic));
  PetscCall(ISRestoreIndices(isrow,&r));

  PetscCall(ISIdentity(isrow,&row_identity));
  PetscCall(ISIdentity(isicol,&col_identity));
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
  PetscCall(PetscLogFlops(C->cmap->n));

  /* MatShiftView(A,info,&sctx) */
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      PetscCall(PetscInfo(A,"number of shift_pd tries %" PetscInt_FMT ", shift_amount %g, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,(double)sctx.shift_amount,(double)sctx.shift_fraction,(double)sctx.shift_top));
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      PetscCall(PetscInfo(A,"number of shift_nz tries %" PetscInt_FMT ", shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount));
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS) {
      PetscCall(PetscInfo(A,"number of shift_inblocks applied %" PetscInt_FMT ", each shift_amount %g\n",sctx.nshift,(double)info->shiftamount));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_1_inplace(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a    = (Mat_SeqBAIJ*)A->data,*b = (Mat_SeqBAIJ*)C->data;
  IS             isrow = b->row,isicol = b->icol;
  const PetscInt *r,*ic;
  PetscInt       i,j,n = a->mbs,*bi = b->i,*bj = b->j;
  PetscInt       *ajtmpold,*ajtmp,nz,row,*ai = a->i,*aj = a->j;
  PetscInt       *diag_offset = b->diag,diag,*pj;
  MatScalar      *pv,*v,*rtmp,multiplier,*pc;
  MatScalar      *ba = b->a,*aa = a->a;
  PetscBool      row_identity, col_identity;

  PetscFunctionBegin;
  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(ISGetIndices(isicol,&ic));
  PetscCall(PetscMalloc1(n+1,&rtmp));

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
        PetscCall(PetscLogFlops(1.0+2.0*nz));
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
    PetscCheckFalse(pv[diag] == 0.0,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot: row in original ordering %" PetscInt_FMT " in permuted ordering %" PetscInt_FMT,r[i],i);
    pv[diag] = 1.0/pv[diag];
  }

  PetscCall(PetscFree(rtmp));
  PetscCall(ISRestoreIndices(isicol,&ic));
  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(ISIdentity(isrow,&row_identity));
  PetscCall(ISIdentity(isicol,&col_identity));
  if (row_identity && col_identity) {
    C->ops->solve          = MatSolve_SeqBAIJ_1_NaturalOrdering_inplace;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_1_NaturalOrdering_inplace;
  } else {
    C->ops->solve          = MatSolve_SeqBAIJ_1_inplace;
    C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_1_inplace;
  }
  C->assembled = PETSC_TRUE;
  PetscCall(PetscLogFlops(C->cmap->n));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_petsc(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERPETSC;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqbaij_petsc(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCheckFalse(A->hermitian && (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC),PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian Factor is not supported");
#endif
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(MatSetSizes(*B,n,n,n,n));
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    PetscCall(MatSetType(*B,MATSEQBAIJ));

    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqBAIJ;
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqBAIJ;
    PetscCall(PetscStrallocpy(MATORDERINGND,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ILU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ILUDT]));
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    PetscCall(MatSetType(*B,MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(*B,A->rmap->bs,MAT_SKIP_ALLOCATION,NULL));

    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqBAIJ;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqBAIJ;
    /*  Future optimization would be direct symbolic and numerical factorization for BAIJ to support orderings and Cholesky, instead of first converting to SBAIJ */
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_CHOLESKY]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL,(char**)&(*B)->preferredordering[MAT_FACTOR_ICC]));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  (*B)->factortype = ftype;
  (*B)->canuseordering = PETSC_TRUE;

  PetscCall(PetscFree((*B)->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC,&(*B)->solvertype));
  PetscCall(PetscObjectComposeFunction((PetscObject)*B,"MatFactorGetSolverType_C",MatFactorGetSolverType_petsc));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
PetscErrorCode MatLUFactor_SeqBAIJ(Mat A,IS row,IS col,const MatFactorInfo *info)
{
  Mat            C;

  PetscFunctionBegin;
  PetscCall(MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&C));
  PetscCall(MatLUFactorSymbolic(C,A,row,col,info));
  PetscCall(MatLUFactorNumeric(C,A,info));

  A->ops->solve          = C->ops->solve;
  A->ops->solvetranspose = C->ops->solvetranspose;

  PetscCall(MatHeaderMerge(A,&C));
  PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)((Mat_SeqBAIJ*)(A->data))->icol));
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/sbaij/seq/sbaij.h>
PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N(Mat C,Mat A,const MatFactorInfo *info)
{
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
      PetscCall(MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&a->sbaijMat));
    }
    PetscCall((a->sbaijMat)->ops->choleskyfactornumeric(C,a->sbaijMat,info));
    PetscCall(MatDestroy(&a->sbaijMat));
    PetscFunctionReturn(0);
  }

  /* MatPivotSetUp(): initialize shift context sctx */
  PetscCall(PetscMemzero(&sctx,sizeof(FactorShiftCtx)));

  PetscCall(ISGetIndices(ip,&rip));
  PetscCall(PetscMalloc3(mbs,&rtmp,mbs,&il,mbs,&jl));

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
      PetscCall(MatPivotCheck(C,A,info,&sctx,k));
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
  PetscCall(PetscFree3(rtmp,il,jl));

  PetscCall(ISRestoreIndices(ip,&rip));

  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;

  PetscCall(PetscLogFlops(C->rmap->N));
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      PetscCall(PetscInfo(A,"number of shiftpd tries %" PetscInt_FMT ", shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount));
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      PetscCall(PetscInfo(A,"number of shiftnz tries %" PetscInt_FMT ", shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering(Mat C,Mat A,const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data;
  Mat_SeqSBAIJ   *b=(Mat_SeqSBAIJ*)C->data;
  PetscInt       i,j,am=a->mbs;
  PetscInt       *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscInt       k,jmin,*jl,*il,nexti,ili,*acol,*bcol,nz;
  MatScalar      *rtmp,*ba=b->a,*aa=a->a,dk,uikdi,*aval,*bval;
  PetscReal      rs;
  FactorShiftCtx sctx;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  PetscCall(PetscMemzero(&sctx,sizeof(FactorShiftCtx)));

  PetscCall(PetscMalloc3(am,&rtmp,am,&il,am,&jl));

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
      PetscCall(MatPivotCheck(C,A,info,&sctx,k));
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
  PetscCall(PetscFree3(rtmp,il,jl));

  C->ops->solve          = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;
  C->preallocated        = PETSC_TRUE;

  PetscCall(PetscLogFlops(C->rmap->N));
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      PetscCall(PetscInfo(A,"number of shiftnz tries %" PetscInt_FMT ", shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount));
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      PetscCall(PetscInfo(A,"number of shiftpd tries %" PetscInt_FMT ", shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount));
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
  PetscCall(MatMissingDiagonal(A,&missing,&i));
  PetscCheck(!missing,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %" PetscInt_FMT,i);

  if (bs > 1) {
    if (!a->sbaijMat) {
      PetscCall(MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&a->sbaijMat));
    }
    (fact)->ops->iccfactorsymbolic = MatICCFactorSymbolic_SeqSBAIJ;  /* undue the change made in MatGetFactor_seqbaij_petsc */

    PetscCall(MatICCFactorSymbolic(fact,a->sbaijMat,perm,info));
    PetscFunctionReturn(0);
  }

  PetscCall(ISIdentity(perm,&perm_identity));
  PetscCall(ISGetIndices(perm,&rip));

  /* special case that simply copies fill pattern */
  if (!levels && perm_identity) {
    PetscCall(PetscMalloc1(am+1,&ui));
    for (i=0; i<am; i++) ui[i] = ai[i+1] - a->diag[i]; /* ui: rowlengths - changes when !perm_identity */
    B    = fact;
    PetscCall(MatSeqSBAIJSetPreallocation(B,1,0,ui));

    b  = (Mat_SeqSBAIJ*)B->data;
    uj = b->j;
    for (i=0; i<am; i++) {
      aj = a->j + a->diag[i];
      for (j=0; j<ui[i]; j++) *uj++ = *aj++;
      b->ilen[i] = ui[i];
    }
    PetscCall(PetscFree(ui));

    B->factortype = MAT_FACTOR_NONE;

    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
    B->factortype = MAT_FACTOR_ICC;

    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering;
    PetscFunctionReturn(0);
  }

  /* initialization */
  PetscCall(PetscMalloc1(am+1,&ui));
  ui[0] = 0;
  PetscCall(PetscMalloc1(2*am+1,&cols_lvl));

  /* jl: linked list for storing indices of the pivot rows
     il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
  PetscCall(PetscMalloc4(am,&uj_ptr,am,&uj_lvl_ptr,am,&il,am,&jl));
  for (i=0; i<am; i++) {
    jl[i] = am; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = am + 1;
  PetscCall(PetscIncompleteLLCreate(am,am,nlnk,lnk,lnk_lvl,lnkbt));

  /* initial FreeSpace size is fill*(ai[am]+am)/2 */
  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am]/2,am/2)),&free_space));

  current_space = free_space;

  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am]/2,am/2)),&free_space_lvl));
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
    PetscCall(PetscIncompleteLLAdd(ncols_upper,cols,levels,cols_lvl,am,&nlnk,lnk,lnk_lvl,lnkbt));
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
      PetscCall(PetscIncompleteLLAddSorted(ncols,cols,levels,cols_lvl,am,&nlnk,lnk,lnk_lvl,lnkbt));
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
      PetscCall(PetscFreeSpaceGet(i,&current_space));
      PetscCall(PetscFreeSpaceGet(i,&current_space_lvl));
      reallocs++;
    }

    /* copy data into free_space and free_space_lvl, then initialize lnk */
    PetscCall(PetscIncompleteLLClean(am,am,nzk,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt));

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

  PetscCall(ISRestoreIndices(perm,&rip));
  PetscCall(PetscFree4(uj_ptr,uj_lvl_ptr,il,jl));
  PetscCall(PetscFree(cols_lvl));

  /* copy free_space into uj and free free_space; set uj in new datastructure; */
  PetscCall(PetscMalloc1(ui[am]+1,&uj));
  PetscCall(PetscFreeSpaceContiguous(&free_space,uj));
  PetscCall(PetscIncompleteLLDestroy(lnk,lnkbt));
  PetscCall(PetscFreeSpaceDestroy(free_space_lvl));

  /* put together the new matrix in MATSEQSBAIJ format */
  B    = fact;
  PetscCall(MatSeqSBAIJSetPreallocation(B,1,MAT_SKIP_ALLOCATION,NULL));

  b                = (Mat_SeqSBAIJ*)B->data;
  b->singlemalloc  = PETSC_FALSE;
  b->free_a        = PETSC_TRUE;
  b->free_ij       = PETSC_TRUE;

  PetscCall(PetscMalloc1(ui[am]+1,&b->a));

  b->j             = uj;
  b->i             = ui;
  b->diag          = NULL;
  b->ilen          = NULL;
  b->imax          = NULL;
  b->row           = perm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  PetscCall(PetscObjectReference((PetscObject)perm));

  b->icol = perm;

  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(PetscMalloc1(am+1,&b->solve_work));
  PetscCall(PetscLogObjectMemory((PetscObject)B,(ui[am]-am)*(sizeof(PetscInt)+sizeof(MatScalar))));

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
    PetscCall(PetscInfo(A,"Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af));
    PetscCall(PetscInfo(A,"Run with -pc_factor_fill %g or use \n",(double)af));
    PetscCall(PetscInfo(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af));
  } else {
    PetscCall(PetscInfo(A,"Empty matrix\n"));
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
      PetscCall(MatConvert(A,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&a->sbaijMat));
    }
    (fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqSBAIJ; /* undue the change made in MatGetFactor_seqbaij_petsc */

    PetscCall(MatCholeskyFactorSymbolic(fact,a->sbaijMat,perm,info));
    PetscFunctionReturn(0);
  }

  PetscCall(MatMissingDiagonal(A,&missing,&i));
  PetscCheck(!missing,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %" PetscInt_FMT,i);

  /* check whether perm is the identity mapping */
  PetscCall(ISIdentity(perm,&perm_identity));
  PetscCheck(perm_identity,PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported");
  PetscCall(ISGetIndices(perm,&rip));

  /* initialization */
  PetscCall(PetscMalloc1(mbs+1,&ui));
  ui[0] = 0;

  /* jl: linked list for storing indices of the pivot rows
     il: il[i] points to the 1st nonzero entry of U(i,k:mbs-1) */
  PetscCall(PetscMalloc4(mbs,&ui_ptr,mbs,&il,mbs,&jl,mbs,&cols));
  for (i=0; i<mbs; i++) {
    jl[i] = mbs; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = mbs + 1;
  PetscCall(PetscLLCreate(mbs,mbs,nlnk,lnk,lnkbt));

  /* initial FreeSpace size is fill* (ai[mbs]+mbs)/2 */
  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[mbs]/2,mbs/2)),&free_space));

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
    PetscCall(PetscLLAdd(ncols_upper,cols,mbs,&nlnk,lnk,lnkbt));
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
      PetscCall(PetscLLAddSorted(ncols,uj_ptr,mbs,&nlnk,lnk,lnkbt));
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
      PetscCall(PetscFreeSpaceGet(i,&current_space));
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    PetscCall(PetscLLClean(mbs,mbs,nzk,lnk,current_space->array,lnkbt));

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

  PetscCall(ISRestoreIndices(perm,&rip));
  PetscCall(PetscFree4(ui_ptr,il,jl,cols));

  /* copy free_space into uj and free free_space; set uj in new datastructure; */
  PetscCall(PetscMalloc1(ui[mbs]+1,&uj));
  PetscCall(PetscFreeSpaceContiguous(&free_space,uj));
  PetscCall(PetscLLDestroy(lnk,lnkbt));

  /* put together the new matrix in MATSEQSBAIJ format */
  B    = fact;
  PetscCall(MatSeqSBAIJSetPreallocation(B,bs,MAT_SKIP_ALLOCATION,NULL));

  b               = (Mat_SeqSBAIJ*)B->data;
  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;

  PetscCall(PetscMalloc1(ui[mbs]+1,&b->a));

  b->j             = uj;
  b->i             = ui;
  b->diag          = NULL;
  b->ilen          = NULL;
  b->imax          = NULL;
  b->row           = perm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  PetscCall(PetscObjectReference((PetscObject)perm));
  b->icol  = perm;
  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(PetscMalloc1(mbs+1,&b->solve_work));
  PetscCall(PetscLogObjectMemory((PetscObject)B,(ui[mbs]-mbs)*(sizeof(PetscInt)+sizeof(MatScalar))));
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
    PetscCall(PetscInfo(A,"Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af));
    PetscCall(PetscInfo(A,"Run with -pc_factor_fill %g or use \n",(double)af));
    PetscCall(PetscInfo(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af));
  } else {
    PetscCall(PetscInfo(A,"Empty matrix\n"));
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
  const PetscInt    *ai=a->i,*aj=a->j,*adiag=a->diag,*vi;
  PetscInt          i,k,n=a->mbs;
  PetscInt          nz,bs=A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,*s,*t,*ls;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;

  /* forward solve the lower triangular */
  PetscCall(PetscArraycpy(t,b,bs)); /* copy 1st block of b to t */

  for (i=1; i<n; i++) {
    v    = aa + bs2*ai[i];
    vi   = aj + ai[i];
    nz   = ai[i+1] - ai[i];
    s    = t + bs*i;
    PetscCall(PetscArraycpy(s,b+bs*i,bs)); /* copy i_th block of b to t */
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
    PetscCall(PetscArraycpy(ls,t+i*bs,bs));
    for (k=0; k<nz; k++) {
      PetscKernel_v_gets_v_minus_A_times_w(bs,ls,v,t+bs*vi[k]);
      v += bs2;
    }
    PetscKernel_w_gets_A_times_v(bs,ls,aa+bs2*adiag[i],t+i*bs); /* *inv(diagonal[i]) */
    PetscCall(PetscArraycpy(x+i*bs,t+i*bs,bs));
  }

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*(a->bs2)*(a->nz) - A->rmap->bs*A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_N(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ        *a   =(Mat_SeqBAIJ*)A->data;
  IS                 iscol=a->col,isrow=a->row;
  const PetscInt     *r,*c,*rout,*cout,*ai=a->i,*aj=a->j,*adiag=a->diag,*vi;
  PetscInt           i,m,n=a->mbs;
  PetscInt           nz,bs=A->rmap->bs,bs2=a->bs2;
  const MatScalar    *aa=a->a,*v;
  PetscScalar        *x,*s,*t,*ls;
  const PetscScalar  *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;

  PetscCall(ISGetIndices(isrow,&rout)); r = rout;
  PetscCall(ISGetIndices(iscol,&cout)); c = cout;

  /* forward solve the lower triangular */
  PetscCall(PetscArraycpy(t,b+bs*r[0],bs));
  for (i=1; i<n; i++) {
    v    = aa + bs2*ai[i];
    vi   = aj + ai[i];
    nz   = ai[i+1] - ai[i];
    s    = t + bs*i;
    PetscCall(PetscArraycpy(s,b+bs*r[i],bs));
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
    PetscCall(PetscArraycpy(ls,t+i*bs,bs));
    for (m=0; m<nz; m++) {
      PetscKernel_v_gets_v_minus_A_times_w(bs,ls,v,t+bs*vi[m]);
      v += bs2;
    }
    PetscKernel_w_gets_A_times_v(bs,ls,v,t+i*bs); /* *inv(diagonal[i]) */
    PetscCall(PetscArraycpy(x + bs*c[i],t+i*bs,bs));
  }
  PetscCall(ISRestoreIndices(isrow,&rout));
  PetscCall(ISRestoreIndices(iscol,&cout));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*(a->bs2)*(a->nz) - A->rmap->bs*A->cmap->n));
  PetscFunctionReturn(0);
}

/*
    For each block in an block array saves the largest absolute value in the block into another array
*/
static PetscErrorCode MatBlockAbs_private(PetscInt nbs,PetscInt bs2,PetscScalar *blockarray,PetscReal *absarray)
{
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscCall(PetscArrayzero(absarray,nbs+1));
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
  PetscCall(MatMissingDiagonal(A,&missing,&i));
  PetscCheck(!missing,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %" PetscInt_FMT,i);
  adiag=a->diag;

  PetscCall(ISInvertPermutation(iscol,PETSC_DECIDE,&isicol));

  /* bdiag is location of diagonal in factor */
  PetscCall(PetscMalloc1(mbs+1,&bdiag));

  /* allocate row pointers bi */
  PetscCall(PetscMalloc1(2*mbs+2,&bi));

  /* allocate bj and ba; max num of nonzero entries is (ai[n]+2*n*dtcount+2) */
  dtcount = (PetscInt)info->dtcount;
  if (dtcount > mbs-1) dtcount = mbs-1;
  nnz_max = ai[mbs]+2*mbs*dtcount +2;
  /* printf("MatILUDTFactor_SeqBAIJ, bs %d, ai[mbs] %d, nnz_max  %d, dtcount %d\n",bs,ai[mbs],nnz_max,dtcount); */
  PetscCall(PetscMalloc1(nnz_max,&bj));
  nnz_max = nnz_max*bs2;
  PetscCall(PetscMalloc1(nnz_max,&ba));

  /* put together the new matrix */
  PetscCall(MatSeqBAIJSetPreallocation(B,bs,MAT_SKIP_ALLOCATION,NULL));
  PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)isicol));

  b               = (Mat_SeqBAIJ*)(B)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;

  b->a    = ba;
  b->j    = bj;
  b->i    = bi;
  b->diag = bdiag;
  b->ilen = NULL;
  b->imax = NULL;
  b->row  = isrow;
  b->col  = iscol;

  PetscCall(PetscObjectReference((PetscObject)isrow));
  PetscCall(PetscObjectReference((PetscObject)iscol));

  b->icol  = isicol;
  PetscCall(PetscMalloc1(bs*(mbs+1),&b->solve_work));
  PetscCall(PetscLogObjectMemory((PetscObject)B,nnz_max*(sizeof(PetscInt)+sizeof(MatScalar))));
  b->maxnz = nnz_max/bs2;

  (B)->factortype            = MAT_FACTOR_ILUDT;
  (B)->info.factor_mallocs   = 0;
  (B)->info.fill_ratio_given = ((PetscReal)nnz_max)/((PetscReal)(ai[mbs]*bs2));
  /* ------- end of symbolic factorization ---------*/
  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(ISGetIndices(isicol,&ic));

  /* linked list for storing column indices of the active row */
  nlnk = mbs + 1;
  PetscCall(PetscLLCreate(mbs,mbs,nlnk,lnk,lnkbt));

  /* im: used by PetscLLAddSortedLU(); jtmp: working array for column indices of active row */
  PetscCall(PetscMalloc2(mbs,&im,mbs,&jtmp));
  /* rtmp, vtmp: working arrays for sparse and contiguous row entries of active row */
  PetscCall(PetscMalloc2(mbs*bs2,&rtmp,mbs*bs2,&vtmp));
  PetscCall(PetscMalloc1(mbs+1,&vtmp_abs));
  PetscCall(PetscMalloc3(bs,&v_work,bs2,&multiplier,bs,&v_pivots));

  allowzeropivot = PetscNot(A->erroriffailure);
  bi[0]       = 0;
  bdiag[0]    = (nnz_max/bs2)-1; /* location of diagonal in factor B */
  bi[2*mbs+1] = bdiag[0]+1; /* endof bj and ba array */
  for (i=0; i<mbs; i++) {
    /* copy initial fill into linked list */
    nzi = ai[r[i]+1] - ai[r[i]];
    PetscCheck(nzi,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix: row in original ordering %" PetscInt_FMT " in permuted ordering %" PetscInt_FMT,r[i],i);
    nzi_al = adiag[r[i]] - ai[r[i]];
    nzi_au = ai[r[i]+1] - adiag[r[i]] -1;

    /* load in initial unfactored row */
    ajtmp = aj + ai[r[i]];
    PetscCall(PetscLLAddPerm(nzi,ajtmp,ic,mbs,&nlnk,lnk,lnkbt));
    PetscCall(PetscArrayzero(rtmp,mbs*bs2));
    aatmp = a->a + bs2*ai[r[i]];
    for (j=0; j<nzi; j++) PetscCall(PetscArraycpy(rtmp+bs2*ic[ajtmp[j]],aatmp+bs2*j,bs2));

    /* add pivot rows into linked list */
    row = lnk[mbs];
    while (row < i) {
      nzi_bl = bi[row+1] - bi[row] + 1;
      bjtmp  = bj + bdiag[row+1]+1; /* points to 1st column next to the diagonal in U */
      PetscCall(PetscLLAddSortedLU(bjtmp,row,&nlnk,lnk,lnkbt,i,nzi_bl,im));
      nzi   += nlnk;
      row    = lnk[row];
    }

    /* copy data from lnk into jtmp, then initialize lnk */
    PetscCall(PetscLLClean(mbs,mbs,nzi,lnk,jtmp,lnkbt));

    /* numerical factorization */
    bjtmp = jtmp;
    row   = *bjtmp++; /* 1st pivot row */

    while  (row < i) {
      pc = rtmp + bs2*row;
      pv = ba + bs2*bdiag[row]; /* inv(diag) of the pivot row */
      PetscKernel_A_gets_A_times_B(bs,pc,pv,multiplier); /* pc= multiplier = pc*inv(diag[row]) */
      PetscCall(MatBlockAbs_private(1,bs2,pc,vtmp_abs));
      if (vtmp_abs[0] > dt) { /* apply tolerance dropping rule */
        pj = bj + bdiag[row+1] + 1;         /* point to 1st entry of U(row,:) */
        pv = ba + bs2*(bdiag[row+1] + 1);
        nz = bdiag[row] - bdiag[row+1] - 1;         /* num of entries in U(row,:), excluding diagonal */
        for (j=0; j<nz; j++) {
          PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j);
        }
        /* PetscCall(PetscLogFlops(bslog*(nz+1.0)-bs)); */
      }
      row = *bjtmp++;
    }

    /* copy sparse rtmp into contiguous vtmp; separate L and U part */
    nzi_bl = 0; j = 0;
    while (jtmp[j] < i) { /* L-part. Note: jtmp is sorted */
      PetscCall(PetscArraycpy(vtmp+bs2*j,rtmp+bs2*jtmp[j],bs2));
      nzi_bl++; j++;
    }
    nzi_bu = nzi - nzi_bl -1;

    while (j < nzi) { /* U-part */
      PetscCall(PetscArraycpy(vtmp+bs2*j,rtmp+bs2*jtmp[j],bs2));
      j++;
    }

    PetscCall(MatBlockAbs_private(nzi,bs2,vtmp,vtmp_abs));

    bjtmp = bj + bi[i];
    batmp = ba + bs2*bi[i];
    /* apply level dropping rule to L part */
    ncut = nzi_al + dtcount;
    if (ncut < nzi_bl) {
      PetscCall(PetscSortSplitReal(ncut,nzi_bl,vtmp_abs,jtmp));
      PetscCall(PetscSortIntWithScalarArray(ncut,jtmp,vtmp));
    } else {
      ncut = nzi_bl;
    }
    for (j=0; j<ncut; j++) {
      bjtmp[j] = jtmp[j];
      PetscCall(PetscArraycpy(batmp+bs2*j,rtmp+bs2*bjtmp[j],bs2));
    }
    bi[i+1] = bi[i] + ncut;
    nzi     = ncut + 1;

    /* apply level dropping rule to U part */
    ncut = nzi_au + dtcount;
    if (ncut < nzi_bu) {
      PetscCall(PetscSortSplitReal(ncut,nzi_bu,vtmp_abs+nzi_bl+1,jtmp+nzi_bl+1));
      PetscCall(PetscSortIntWithScalarArray(ncut,jtmp+nzi_bl+1,vtmp+nzi_bl+1));
    } else {
      ncut = nzi_bu;
    }
    nzi += ncut;

    /* mark bdiagonal */
    bdiag[i+1]    = bdiag[i] - (ncut + 1);
    bi[2*mbs - i] = bi[2*mbs - i +1] - (ncut + 1);

    bjtmp  = bj + bdiag[i];
    batmp  = ba + bs2*bdiag[i];
    PetscCall(PetscArraycpy(batmp,rtmp+bs2*i,bs2));
    *bjtmp = i;

    bjtmp = bj + bdiag[i+1]+1;
    batmp = ba + (bdiag[i+1]+1)*bs2;

    for (k=0; k<ncut; k++) {
      bjtmp[k] = jtmp[nzi_bl+1+k];
      PetscCall(PetscArraycpy(batmp+bs2*k,rtmp+bs2*bjtmp[k],bs2));
    }

    im[i] = nzi; /* used by PetscLLAddSortedLU() */

    /* invert diagonal block for simpler triangular solves - add shift??? */
    batmp = ba + bs2*bdiag[i];

    PetscCall(PetscKernel_A_gets_inverse_A(bs,batmp,v_pivots,v_work,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) B->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  } /* for (i=0; i<mbs; i++) */
  PetscCall(PetscFree3(v_work,multiplier,v_pivots));

  /* printf("end of L %d, beginning of U %d\n",bi[mbs],bdiag[mbs]); */
  PetscCheck(bi[mbs] < bdiag[mbs],PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"end of L array %" PetscInt_FMT " cannot >= the beginning of U array %" PetscInt_FMT,bi[mbs],bdiag[mbs]);

  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(ISRestoreIndices(isicol,&ic));

  PetscCall(PetscLLDestroy(lnk,lnkbt));

  PetscCall(PetscFree2(im,jtmp));
  PetscCall(PetscFree2(rtmp,vtmp));

  PetscCall(PetscLogFlops(bs2*B->cmap->n));
  b->maxnz = b->nz = bi[mbs] + bdiag[0] - bdiag[mbs];

  PetscCall(ISIdentity(isrow,&row_identity));
  PetscCall(ISIdentity(isicol,&icol_identity));
  if (row_identity && icol_identity) {
    B->ops->solve = MatSolve_SeqBAIJ_N_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqBAIJ_N;
  }

  B->ops->solveadd          = NULL;
  B->ops->solvetranspose    = NULL;
  B->ops->solvetransposeadd = NULL;
  B->ops->matsolve          = NULL;
  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;
  PetscFunctionReturn(0);
}
