/*
    Factorization code for BAIJ format.
*/
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
      Version for when blocks are 4 by 4
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_inplace(Mat C, Mat A, const MatFactorInfo *info)
{
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
  IS              isrow = b->row, isicol = b->icol;
  const PetscInt *r, *ic;
  PetscInt        i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  PetscInt       *ajtmpold, *ajtmp, nz, row;
  PetscInt       *diag_offset = b->diag, idx, *ai = a->i, *aj = a->j, *pj;
  MatScalar      *pv, *v, *rtmp, *pc, *w, *x;
  MatScalar       p1, p2, p3, p4, m1, m2, m3, m4, m5, m6, m7, m8, m9, x1, x2, x3, x4;
  MatScalar       p5, p6, p7, p8, p9, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
  MatScalar       p10, p11, p12, p13, p14, p15, p16, m10, m11, m12;
  MatScalar       m13, m14, m15, m16;
  MatScalar      *ba = b->a, *aa = a->a;
  PetscBool       pivotinblocks = b->pivotinblocks;
  PetscReal       shift         = info->shiftamount;
  PetscBool       allowzeropivot, zeropivotdetected = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(isicol, &ic));
  PetscCall(PetscMalloc1(16 * (n + 1), &rtmp));
  allowzeropivot = PetscNot(A->erroriffailure);

  for (i = 0; i < n; i++) {
    nz    = bi[i + 1] - bi[i];
    ajtmp = bj + bi[i];
    for (j = 0; j < nz; j++) {
      x    = rtmp + 16 * ajtmp[j];
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx + 1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 16 * ai[idx];
    for (j = 0; j < nz; j++) {
      x     = rtmp + 16 * ic[ajtmpold[j]];
      x[0]  = v[0];
      x[1]  = v[1];
      x[2]  = v[2];
      x[3]  = v[3];
      x[4]  = v[4];
      x[5]  = v[5];
      x[6]  = v[6];
      x[7]  = v[7];
      x[8]  = v[8];
      x[9]  = v[9];
      x[10] = v[10];
      x[11] = v[11];
      x[12] = v[12];
      x[13] = v[13];
      x[14] = v[14];
      x[15] = v[15];
      v += 16;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 16 * row;
      p1  = pc[0];
      p2  = pc[1];
      p3  = pc[2];
      p4  = pc[3];
      p5  = pc[4];
      p6  = pc[5];
      p7  = pc[6];
      p8  = pc[7];
      p9  = pc[8];
      p10 = pc[9];
      p11 = pc[10];
      p12 = pc[11];
      p13 = pc[12];
      p14 = pc[13];
      p15 = pc[14];
      p16 = pc[15];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 || p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 || p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0 || p16 != 0.0) {
        pv    = ba + 16 * diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0];
        x2    = pv[1];
        x3    = pv[2];
        x4    = pv[3];
        x5    = pv[4];
        x6    = pv[5];
        x7    = pv[6];
        x8    = pv[7];
        x9    = pv[8];
        x10   = pv[9];
        x11   = pv[10];
        x12   = pv[11];
        x13   = pv[12];
        x14   = pv[13];
        x15   = pv[14];
        x16   = pv[15];
        pc[0] = m1 = p1 * x1 + p5 * x2 + p9 * x3 + p13 * x4;
        pc[1] = m2 = p2 * x1 + p6 * x2 + p10 * x3 + p14 * x4;
        pc[2] = m3 = p3 * x1 + p7 * x2 + p11 * x3 + p15 * x4;
        pc[3] = m4 = p4 * x1 + p8 * x2 + p12 * x3 + p16 * x4;

        pc[4] = m5 = p1 * x5 + p5 * x6 + p9 * x7 + p13 * x8;
        pc[5] = m6 = p2 * x5 + p6 * x6 + p10 * x7 + p14 * x8;
        pc[6] = m7 = p3 * x5 + p7 * x6 + p11 * x7 + p15 * x8;
        pc[7] = m8 = p4 * x5 + p8 * x6 + p12 * x7 + p16 * x8;

        pc[8] = m9 = p1 * x9 + p5 * x10 + p9 * x11 + p13 * x12;
        pc[9] = m10 = p2 * x9 + p6 * x10 + p10 * x11 + p14 * x12;
        pc[10] = m11 = p3 * x9 + p7 * x10 + p11 * x11 + p15 * x12;
        pc[11] = m12 = p4 * x9 + p8 * x10 + p12 * x11 + p16 * x12;

        pc[12] = m13 = p1 * x13 + p5 * x14 + p9 * x15 + p13 * x16;
        pc[13] = m14 = p2 * x13 + p6 * x14 + p10 * x15 + p14 * x16;
        pc[14] = m15 = p3 * x13 + p7 * x14 + p11 * x15 + p15 * x16;
        pc[15] = m16 = p4 * x13 + p8 * x14 + p12 * x15 + p16 * x16;

        nz = bi[row + 1] - diag_offset[row] - 1;
        pv += 16;
        for (j = 0; j < nz; j++) {
          x1  = pv[0];
          x2  = pv[1];
          x3  = pv[2];
          x4  = pv[3];
          x5  = pv[4];
          x6  = pv[5];
          x7  = pv[6];
          x8  = pv[7];
          x9  = pv[8];
          x10 = pv[9];
          x11 = pv[10];
          x12 = pv[11];
          x13 = pv[12];
          x14 = pv[13];
          x15 = pv[14];
          x16 = pv[15];
          x   = rtmp + 16 * pj[j];
          x[0] -= m1 * x1 + m5 * x2 + m9 * x3 + m13 * x4;
          x[1] -= m2 * x1 + m6 * x2 + m10 * x3 + m14 * x4;
          x[2] -= m3 * x1 + m7 * x2 + m11 * x3 + m15 * x4;
          x[3] -= m4 * x1 + m8 * x2 + m12 * x3 + m16 * x4;

          x[4] -= m1 * x5 + m5 * x6 + m9 * x7 + m13 * x8;
          x[5] -= m2 * x5 + m6 * x6 + m10 * x7 + m14 * x8;
          x[6] -= m3 * x5 + m7 * x6 + m11 * x7 + m15 * x8;
          x[7] -= m4 * x5 + m8 * x6 + m12 * x7 + m16 * x8;

          x[8] -= m1 * x9 + m5 * x10 + m9 * x11 + m13 * x12;
          x[9] -= m2 * x9 + m6 * x10 + m10 * x11 + m14 * x12;
          x[10] -= m3 * x9 + m7 * x10 + m11 * x11 + m15 * x12;
          x[11] -= m4 * x9 + m8 * x10 + m12 * x11 + m16 * x12;

          x[12] -= m1 * x13 + m5 * x14 + m9 * x15 + m13 * x16;
          x[13] -= m2 * x13 + m6 * x14 + m10 * x15 + m14 * x16;
          x[14] -= m3 * x13 + m7 * x14 + m11 * x15 + m15 * x16;
          x[15] -= m4 * x13 + m8 * x14 + m12 * x15 + m16 * x16;

          pv += 16;
        }
        PetscCall(PetscLogFlops(128.0 * nz + 112.0));
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 16 * bi[i];
    pj = bj + bi[i];
    nz = bi[i + 1] - bi[i];
    for (j = 0; j < nz; j++) {
      x      = rtmp + 16 * pj[j];
      pv[0]  = x[0];
      pv[1]  = x[1];
      pv[2]  = x[2];
      pv[3]  = x[3];
      pv[4]  = x[4];
      pv[5]  = x[5];
      pv[6]  = x[6];
      pv[7]  = x[7];
      pv[8]  = x[8];
      pv[9]  = x[9];
      pv[10] = x[10];
      pv[11] = x[11];
      pv[12] = x[12];
      pv[13] = x[13];
      pv[14] = x[14];
      pv[15] = x[15];
      pv += 16;
    }
    /* invert diagonal block */
    w = ba + 16 * diag_offset[i];
    if (pivotinblocks) {
      PetscCall(PetscKernel_A_gets_inverse_A_4(w, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    } else {
      PetscCall(PetscKernel_A_gets_inverse_A_4_nopivot(w));
    }
  }

  PetscCall(PetscFree(rtmp));
  PetscCall(ISRestoreIndices(isicol, &ic));
  PetscCall(ISRestoreIndices(isrow, &r));

  C->ops->solve          = MatSolve_SeqBAIJ_4_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_4_inplace;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333 * 4 * 4 * 4 * b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatLUFactorNumeric_SeqBAIJ_4 -
     copied from MatLUFactorNumeric_SeqBAIJ_N_inplace() and manually re-implemented
       PetscKernel_A_gets_A_times_B()
       PetscKernel_A_gets_A_minus_B_times_C()
       PetscKernel_A_gets_inverse_A()
*/

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat             C = B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
  IS              isrow = b->row, isicol = b->icol;
  const PetscInt *r, *ic;
  PetscInt        i, j, k, nz, nzL, row;
  const PetscInt  n = a->mbs, *ai = a->i, *aj = a->j, *bi = b->i, *bj = b->j;
  const PetscInt *ajtmp, *bjtmp, *bdiag = b->diag, *pj, bs2 = a->bs2;
  MatScalar      *rtmp, *pc, *mwork, *v, *pv, *aa = a->a;
  PetscInt        flg;
  PetscReal       shift;
  PetscBool       allowzeropivot, zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(ISGetIndices(isrow, &r));
  PetscCall(ISGetIndices(isicol, &ic));

  if (info->shifttype == (PetscReal)MAT_SHIFT_NONE) {
    shift = 0;
  } else { /* info->shifttype == MAT_SHIFT_INBLOCKS */
    shift = info->shiftamount;
  }

  /* generate work space needed by the factorization */
  PetscCall(PetscMalloc2(bs2 * n, &rtmp, bs2, &mwork));
  PetscCall(PetscArrayzero(rtmp, bs2 * n));

  for (i = 0; i < n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i + 1] - bi[i];
    bjtmp = bj + bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* U part */
    nz    = bdiag[i] - bdiag[i + 1];
    bjtmp = bj + bdiag[i + 1] + 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* load in initial (unfactored row) */
    nz    = ai[r[i] + 1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + bs2 * ai[r[i]];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(rtmp + bs2 * ic[ajtmp[j]], v + bs2 * j, bs2));

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i + 1] - bi[i];
    for (k = 0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2 * row;
      for (flg = 0, j = 0; j < bs2; j++) {
        if (pc[j] != 0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2 * bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        PetscCall(PetscKernel_A_gets_A_times_B_4(pc, pv, mwork));

        pj = b->j + bdiag[row + 1] + 1; /* beginning of U(row,:) */
        pv = b->a + bs2 * (bdiag[row + 1] + 1);
        nz = bdiag[row] - bdiag[row + 1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j = 0; j < nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v = rtmp + bs2 * pj[j];
          PetscCall(PetscKernel_A_gets_A_minus_B_times_C_4(v, pc, pv));
          pv += bs2;
        }
        PetscCall(PetscLogFlops(128.0 * nz + 112)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2 * bi[i];
    pj = b->j + bi[i];
    nz = bi[i + 1] - bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv = b->a + bs2 * bdiag[i];
    pj = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv, rtmp + bs2 * pj[0], bs2));
    PetscCall(PetscKernel_A_gets_inverse_A_4(pv, shift, allowzeropivot, &zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2 * (bdiag[i + 1] + 1);
    pj = b->j + bdiag[i + 1] + 1;
    nz = bdiag[i] - bdiag[i + 1] - 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));
  }

  PetscCall(PetscFree2(rtmp, mwork));
  PetscCall(ISRestoreIndices(isicol, &ic));
  PetscCall(ISRestoreIndices(isrow, &r));

  C->ops->solve          = MatSolve_SeqBAIJ_4;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_4;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333 * 4 * 4 * 4 * n)); /* from inverting diagonal blocks */
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_inplace(Mat C, Mat A, const MatFactorInfo *info)
{
  /*
    Default Version for when blocks are 4 by 4 Using natural ordering
*/
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
  PetscInt     i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  PetscInt    *ajtmpold, *ajtmp, nz, row;
  PetscInt    *diag_offset = b->diag, *ai = a->i, *aj = a->j, *pj;
  MatScalar   *pv, *v, *rtmp, *pc, *w, *x;
  MatScalar    p1, p2, p3, p4, m1, m2, m3, m4, m5, m6, m7, m8, m9, x1, x2, x3, x4;
  MatScalar    p5, p6, p7, p8, p9, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
  MatScalar    p10, p11, p12, p13, p14, p15, p16, m10, m11, m12;
  MatScalar    m13, m14, m15, m16;
  MatScalar   *ba = b->a, *aa = a->a;
  PetscBool    pivotinblocks = b->pivotinblocks;
  PetscReal    shift         = info->shiftamount;
  PetscBool    allowzeropivot, zeropivotdetected = PETSC_FALSE;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  PetscCall(PetscMalloc1(16 * (n + 1), &rtmp));

  for (i = 0; i < n; i++) {
    nz    = bi[i + 1] - bi[i];
    ajtmp = bj + bi[i];
    for (j = 0; j < nz; j++) {
      x    = rtmp + 16 * ajtmp[j];
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = 0.0;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i + 1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 16 * ai[i];
    for (j = 0; j < nz; j++) {
      x     = rtmp + 16 * ajtmpold[j];
      x[0]  = v[0];
      x[1]  = v[1];
      x[2]  = v[2];
      x[3]  = v[3];
      x[4]  = v[4];
      x[5]  = v[5];
      x[6]  = v[6];
      x[7]  = v[7];
      x[8]  = v[8];
      x[9]  = v[9];
      x[10] = v[10];
      x[11] = v[11];
      x[12] = v[12];
      x[13] = v[13];
      x[14] = v[14];
      x[15] = v[15];
      v += 16;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 16 * row;
      p1  = pc[0];
      p2  = pc[1];
      p3  = pc[2];
      p4  = pc[3];
      p5  = pc[4];
      p6  = pc[5];
      p7  = pc[6];
      p8  = pc[7];
      p9  = pc[8];
      p10 = pc[9];
      p11 = pc[10];
      p12 = pc[11];
      p13 = pc[12];
      p14 = pc[13];
      p15 = pc[14];
      p16 = pc[15];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 || p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 || p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0 || p16 != 0.0) {
        pv    = ba + 16 * diag_offset[row];
        pj    = bj + diag_offset[row] + 1;
        x1    = pv[0];
        x2    = pv[1];
        x3    = pv[2];
        x4    = pv[3];
        x5    = pv[4];
        x6    = pv[5];
        x7    = pv[6];
        x8    = pv[7];
        x9    = pv[8];
        x10   = pv[9];
        x11   = pv[10];
        x12   = pv[11];
        x13   = pv[12];
        x14   = pv[13];
        x15   = pv[14];
        x16   = pv[15];
        pc[0] = m1 = p1 * x1 + p5 * x2 + p9 * x3 + p13 * x4;
        pc[1] = m2 = p2 * x1 + p6 * x2 + p10 * x3 + p14 * x4;
        pc[2] = m3 = p3 * x1 + p7 * x2 + p11 * x3 + p15 * x4;
        pc[3] = m4 = p4 * x1 + p8 * x2 + p12 * x3 + p16 * x4;

        pc[4] = m5 = p1 * x5 + p5 * x6 + p9 * x7 + p13 * x8;
        pc[5] = m6 = p2 * x5 + p6 * x6 + p10 * x7 + p14 * x8;
        pc[6] = m7 = p3 * x5 + p7 * x6 + p11 * x7 + p15 * x8;
        pc[7] = m8 = p4 * x5 + p8 * x6 + p12 * x7 + p16 * x8;

        pc[8] = m9 = p1 * x9 + p5 * x10 + p9 * x11 + p13 * x12;
        pc[9] = m10 = p2 * x9 + p6 * x10 + p10 * x11 + p14 * x12;
        pc[10] = m11 = p3 * x9 + p7 * x10 + p11 * x11 + p15 * x12;
        pc[11] = m12 = p4 * x9 + p8 * x10 + p12 * x11 + p16 * x12;

        pc[12] = m13 = p1 * x13 + p5 * x14 + p9 * x15 + p13 * x16;
        pc[13] = m14 = p2 * x13 + p6 * x14 + p10 * x15 + p14 * x16;
        pc[14] = m15 = p3 * x13 + p7 * x14 + p11 * x15 + p15 * x16;
        pc[15] = m16 = p4 * x13 + p8 * x14 + p12 * x15 + p16 * x16;
        nz           = bi[row + 1] - diag_offset[row] - 1;
        pv += 16;
        for (j = 0; j < nz; j++) {
          x1  = pv[0];
          x2  = pv[1];
          x3  = pv[2];
          x4  = pv[3];
          x5  = pv[4];
          x6  = pv[5];
          x7  = pv[6];
          x8  = pv[7];
          x9  = pv[8];
          x10 = pv[9];
          x11 = pv[10];
          x12 = pv[11];
          x13 = pv[12];
          x14 = pv[13];
          x15 = pv[14];
          x16 = pv[15];
          x   = rtmp + 16 * pj[j];
          x[0] -= m1 * x1 + m5 * x2 + m9 * x3 + m13 * x4;
          x[1] -= m2 * x1 + m6 * x2 + m10 * x3 + m14 * x4;
          x[2] -= m3 * x1 + m7 * x2 + m11 * x3 + m15 * x4;
          x[3] -= m4 * x1 + m8 * x2 + m12 * x3 + m16 * x4;

          x[4] -= m1 * x5 + m5 * x6 + m9 * x7 + m13 * x8;
          x[5] -= m2 * x5 + m6 * x6 + m10 * x7 + m14 * x8;
          x[6] -= m3 * x5 + m7 * x6 + m11 * x7 + m15 * x8;
          x[7] -= m4 * x5 + m8 * x6 + m12 * x7 + m16 * x8;

          x[8] -= m1 * x9 + m5 * x10 + m9 * x11 + m13 * x12;
          x[9] -= m2 * x9 + m6 * x10 + m10 * x11 + m14 * x12;
          x[10] -= m3 * x9 + m7 * x10 + m11 * x11 + m15 * x12;
          x[11] -= m4 * x9 + m8 * x10 + m12 * x11 + m16 * x12;

          x[12] -= m1 * x13 + m5 * x14 + m9 * x15 + m13 * x16;
          x[13] -= m2 * x13 + m6 * x14 + m10 * x15 + m14 * x16;
          x[14] -= m3 * x13 + m7 * x14 + m11 * x15 + m15 * x16;
          x[15] -= m4 * x13 + m8 * x14 + m12 * x15 + m16 * x16;

          pv += 16;
        }
        PetscCall(PetscLogFlops(128.0 * nz + 112.0));
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 16 * bi[i];
    pj = bj + bi[i];
    nz = bi[i + 1] - bi[i];
    for (j = 0; j < nz; j++) {
      x      = rtmp + 16 * pj[j];
      pv[0]  = x[0];
      pv[1]  = x[1];
      pv[2]  = x[2];
      pv[3]  = x[3];
      pv[4]  = x[4];
      pv[5]  = x[5];
      pv[6]  = x[6];
      pv[7]  = x[7];
      pv[8]  = x[8];
      pv[9]  = x[9];
      pv[10] = x[10];
      pv[11] = x[11];
      pv[12] = x[12];
      pv[13] = x[13];
      pv[14] = x[14];
      pv[15] = x[15];
      pv += 16;
    }
    /* invert diagonal block */
    w = ba + 16 * diag_offset[i];
    if (pivotinblocks) {
      PetscCall(PetscKernel_A_gets_inverse_A_4(w, shift, allowzeropivot, &zeropivotdetected));
      if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    } else {
      PetscCall(PetscKernel_A_gets_inverse_A_4_nopivot(w));
    }
  }

  PetscCall(PetscFree(rtmp));

  C->ops->solve          = MatSolve_SeqBAIJ_4_NaturalOrdering_inplace;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_4_NaturalOrdering_inplace;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333 * 4 * 4 * 4 * b->mbs)); /* from inverting diagonal blocks */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering -
    copied from MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering_inplace()
*/
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat             C = B;
  Mat_SeqBAIJ    *a = (Mat_SeqBAIJ *)A->data, *b = (Mat_SeqBAIJ *)C->data;
  PetscInt        i, j, k, nz, nzL, row;
  const PetscInt  n = a->mbs, *ai = a->i, *aj = a->j, *bi = b->i, *bj = b->j;
  const PetscInt *ajtmp, *bjtmp, *bdiag = b->diag, *pj, bs2 = a->bs2;
  MatScalar      *rtmp, *pc, *mwork, *v, *pv, *aa = a->a;
  PetscInt        flg;
  PetscReal       shift;
  PetscBool       allowzeropivot, zeropivotdetected;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);

  /* generate work space needed by the factorization */
  PetscCall(PetscMalloc2(bs2 * n, &rtmp, bs2, &mwork));
  PetscCall(PetscArrayzero(rtmp, bs2 * n));

  if (info->shifttype == (PetscReal)MAT_SHIFT_NONE) {
    shift = 0;
  } else { /* info->shifttype == MAT_SHIFT_INBLOCKS */
    shift = info->shiftamount;
  }

  for (i = 0; i < n; i++) {
    /* zero rtmp */
    /* L part */
    nz    = bi[i + 1] - bi[i];
    bjtmp = bj + bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* U part */
    nz    = bdiag[i] - bdiag[i + 1];
    bjtmp = bj + bdiag[i + 1] + 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArrayzero(rtmp + bs2 * bjtmp[j], bs2));

    /* load in initial (unfactored row) */
    nz    = ai[i + 1] - ai[i];
    ajtmp = aj + ai[i];
    v     = aa + bs2 * ai[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(rtmp + bs2 * ajtmp[j], v + bs2 * j, bs2));

    /* elimination */
    bjtmp = bj + bi[i];
    nzL   = bi[i + 1] - bi[i];
    for (k = 0; k < nzL; k++) {
      row = bjtmp[k];
      pc  = rtmp + bs2 * row;
      for (flg = 0, j = 0; j < bs2; j++) {
        if (pc[j] != 0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2 * bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        PetscCall(PetscKernel_A_gets_A_times_B_4(pc, pv, mwork));

        pj = b->j + bdiag[row + 1] + 1; /* beginning of U(row,:) */
        pv = b->a + bs2 * (bdiag[row + 1] + 1);
        nz = bdiag[row] - bdiag[row + 1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j = 0; j < nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v = rtmp + bs2 * pj[j];
          PetscCall(PetscKernel_A_gets_A_minus_B_times_C_4(v, pc, pv));
          pv += bs2;
        }
        PetscCall(PetscLogFlops(128.0 * nz + 112)); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
      }
    }

    /* finished row so stick it into b->a */
    /* L part */
    pv = b->a + bs2 * bi[i];
    pj = b->j + bi[i];
    nz = bi[i + 1] - bi[i];
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));

    /* Mark diagonal and invert diagonal for simpler triangular solves */
    pv = b->a + bs2 * bdiag[i];
    pj = b->j + bdiag[i];
    PetscCall(PetscArraycpy(pv, rtmp + bs2 * pj[0], bs2));
    PetscCall(PetscKernel_A_gets_inverse_A_4(pv, shift, allowzeropivot, &zeropivotdetected));
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2 * (bdiag[i + 1] + 1);
    pj = b->j + bdiag[i + 1] + 1;
    nz = bdiag[i] - bdiag[i + 1] - 1;
    for (j = 0; j < nz; j++) PetscCall(PetscArraycpy(pv + bs2 * j, rtmp + bs2 * pj[j], bs2));
  }
  PetscCall(PetscFree2(rtmp, mwork));

  C->ops->solve          = MatSolve_SeqBAIJ_4_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_4_NaturalOrdering;
  C->assembled           = PETSC_TRUE;

  PetscCall(PetscLogFlops(1.333333333333 * 4 * 4 * 4 * n)); /* from inverting diagonal blocks */
  PetscFunctionReturn(PETSC_SUCCESS);
}
