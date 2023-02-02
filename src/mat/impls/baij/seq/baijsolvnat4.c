#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_inplace(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a  = (Mat_SeqBAIJ *)A->data;
  PetscInt           n  = a->mbs;
  const PetscInt    *ai = a->i, *aj = a->j;
  const PetscInt    *diag = a->diag;
  const MatScalar   *aa   = a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJ)
  {
    static PetscScalar w[2000]; /* very BAD need to fix */
    fortransolvebaij4_(&n, x, ai, aj, diag, aa, b, w);
  }
#elif defined(PETSC_USE_FORTRAN_KERNEL_SOLVEBAIJUNROLL)
  fortransolvebaij4unroll_(&n, x, ai, aj, diag, aa, b);
#else
  {
    PetscScalar      s1, s2, s3, s4, x1, x2, x3, x4;
    const MatScalar *v;
    PetscInt         jdx, idt, idx, nz, i, ai16;
    const PetscInt  *vi;

    /* forward solve the lower triangular */
    idx  = 0;
    x[0] = b[0];
    x[1] = b[1];
    x[2] = b[2];
    x[3] = b[3];
    for (i = 1; i < n; i++) {
      v  = aa + 16 * ai[i];
      vi = aj + ai[i];
      nz = diag[i] - ai[i];
      idx += 4;
      s1 = b[idx];
      s2 = b[1 + idx];
      s3 = b[2 + idx];
      s4 = b[3 + idx];
      while (nz--) {
        jdx = 4 * (*vi++);
        x1  = x[jdx];
        x2  = x[1 + jdx];
        x3  = x[2 + jdx];
        x4  = x[3 + jdx];
        s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
        s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
        s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
        s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
        v += 16;
      }
      x[idx]     = s1;
      x[1 + idx] = s2;
      x[2 + idx] = s3;
      x[3 + idx] = s4;
    }
    /* backward solve the upper triangular */
    idt = 4 * (n - 1);
    for (i = n - 1; i >= 0; i--) {
      ai16 = 16 * diag[i];
      v    = aa + ai16 + 16;
      vi   = aj + diag[i] + 1;
      nz   = ai[i + 1] - diag[i] - 1;
      s1   = x[idt];
      s2   = x[1 + idt];
      s3   = x[2 + idt];
      s4   = x[3 + idt];
      while (nz--) {
        idx = 4 * (*vi++);
        x1  = x[idx];
        x2  = x[1 + idx];
        x3  = x[2 + idx];
        x4  = x[3 + idx];
        s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
        s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
        s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
        s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
        v += 16;
      }
      v          = aa + ai16;
      x[idt]     = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
      x[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
      x[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
      x[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
      idt -= 4;
    }
  }
#endif

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  const PetscInt     n = a->mbs, *vi, *ai = a->i, *aj = a->j, *adiag = a->diag;
  PetscInt           i, k, nz, idx, jdx, idt;
  const PetscInt     bs = A->rmap->bs, bs2 = a->bs2;
  const MatScalar   *aa = a->a, *v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar        s1, s2, s3, s4, x1, x2, x3, x4;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx];
  x[1] = b[1 + idx];
  x[2] = b[2 + idx];
  x[3] = b[3 + idx];
  for (i = 1; i < n; i++) {
    v   = aa + bs2 * ai[i];
    vi  = aj + ai[i];
    nz  = ai[i + 1] - ai[i];
    idx = bs * i;
    s1  = b[idx];
    s2  = b[1 + idx];
    s3  = b[2 + idx];
    s4  = b[3 + idx];
    for (k = 0; k < nz; k++) {
      jdx = bs * vi[k];
      x1  = x[jdx];
      x2  = x[1 + jdx];
      x3  = x[2 + jdx];
      x4  = x[3 + jdx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;

      v += bs2;
    }

    x[idx]     = s1;
    x[1 + idx] = s2;
    x[2 + idx] = s3;
    x[3 + idx] = s4;
  }

  /* backward solve the upper triangular */
  for (i = n - 1; i >= 0; i--) {
    v   = aa + bs2 * (adiag[i + 1] + 1);
    vi  = aj + adiag[i + 1] + 1;
    nz  = adiag[i] - adiag[i + 1] - 1;
    idt = bs * i;
    s1  = x[idt];
    s2  = x[1 + idt];
    s3  = x[2 + idt];
    s4  = x[3 + idt];

    for (k = 0; k < nz; k++) {
      idx = bs * vi[k];
      x1  = x[idx];
      x2  = x[1 + idx];
      x3  = x[2 + idx];
      x4  = x[3 + idx];
      s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
      s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
      s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
      s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;

      v += bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]     = v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4;
    x[1 + idt] = v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4;
    x[2 + idt] = v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4;
    x[3 + idt] = v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4;
  }

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * bs2 * (a->nz) - bs * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_Demotion(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ *)A->data;
  const PetscInt     n = a->mbs, *ai = a->i, *aj = a->j, *diag = a->diag;
  const MatScalar   *aa = a->a;
  const PetscScalar *b;
  PetscScalar       *x;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb, &b));
  PetscCall(VecGetArray(xx, &x));

  {
    MatScalar        s1, s2, s3, s4, x1, x2, x3, x4;
    const MatScalar *v;
    MatScalar       *t = (MatScalar *)x;
    PetscInt         jdx, idt, idx, nz, i, ai16;
    const PetscInt  *vi;

    /* forward solve the lower triangular */
    idx  = 0;
    t[0] = (MatScalar)b[0];
    t[1] = (MatScalar)b[1];
    t[2] = (MatScalar)b[2];
    t[3] = (MatScalar)b[3];
    for (i = 1; i < n; i++) {
      v  = aa + 16 * ai[i];
      vi = aj + ai[i];
      nz = diag[i] - ai[i];
      idx += 4;
      s1 = (MatScalar)b[idx];
      s2 = (MatScalar)b[1 + idx];
      s3 = (MatScalar)b[2 + idx];
      s4 = (MatScalar)b[3 + idx];
      while (nz--) {
        jdx = 4 * (*vi++);
        x1  = t[jdx];
        x2  = t[1 + jdx];
        x3  = t[2 + jdx];
        x4  = t[3 + jdx];
        s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
        s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
        s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
        s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
        v += 16;
      }
      t[idx]     = s1;
      t[1 + idx] = s2;
      t[2 + idx] = s3;
      t[3 + idx] = s4;
    }
    /* backward solve the upper triangular */
    idt = 4 * (n - 1);
    for (i = n - 1; i >= 0; i--) {
      ai16 = 16 * diag[i];
      v    = aa + ai16 + 16;
      vi   = aj + diag[i] + 1;
      nz   = ai[i + 1] - diag[i] - 1;
      s1   = t[idt];
      s2   = t[1 + idt];
      s3   = t[2 + idt];
      s4   = t[3 + idt];
      while (nz--) {
        idx = 4 * (*vi++);
        x1  = (MatScalar)x[idx];
        x2  = (MatScalar)x[1 + idx];
        x3  = (MatScalar)x[2 + idx];
        x4  = (MatScalar)x[3 + idx];
        s1 -= v[0] * x1 + v[4] * x2 + v[8] * x3 + v[12] * x4;
        s2 -= v[1] * x1 + v[5] * x2 + v[9] * x3 + v[13] * x4;
        s3 -= v[2] * x1 + v[6] * x2 + v[10] * x3 + v[14] * x4;
        s4 -= v[3] * x1 + v[7] * x2 + v[11] * x3 + v[15] * x4;
        v += 16;
      }
      v          = aa + ai16;
      x[idt]     = (PetscScalar)(v[0] * s1 + v[4] * s2 + v[8] * s3 + v[12] * s4);
      x[1 + idt] = (PetscScalar)(v[1] * s1 + v[5] * s2 + v[9] * s3 + v[13] * s4);
      x[2 + idt] = (PetscScalar)(v[2] * s1 + v[6] * s2 + v[10] * s3 + v[14] * s4);
      x[3 + idt] = (PetscScalar)(v[3] * s1 + v[7] * s2 + v[11] * s3 + v[15] * s4);
      idt -= 4;
    }
  }

  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_SSE)

  #include PETSC_HAVE_SSE
PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion_usj(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ    *a  = (Mat_SeqBAIJ *)A->data;
  unsigned short *aj = (unsigned short *)a->j;
  int            *ai = a->i, n = a->mbs, *diag = a->diag;
  MatScalar      *aa = a->a;
  PetscScalar    *x, *b;

  PetscFunctionBegin;
  SSE_SCOPE_BEGIN;
  /*
     Note: This code currently uses demotion of double
     to float when performing the mixed-mode computation.
     This may not be numerically reasonable for all applications.
  */
  PREFETCH_NTA(aa + 16 * ai[1]);

  PetscCall(VecGetArray(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  {
    /* x will first be computed in single precision then promoted inplace to double */
    MatScalar      *v, *t = (MatScalar *)x;
    int             nz, i, idt, ai16;
    unsigned int    jdx, idx;
    unsigned short *vi;
    /* Forward solve the lower triangular factor. */

    /* First block is the identity. */
    idx = 0;
    CONVERT_DOUBLE4_FLOAT4(t, b);
    v = aa + 16 * ((unsigned int)ai[1]);

    for (i = 1; i < n;) {
      PREFETCH_NTA(&v[8]);
      vi = aj + ai[i];
      nz = diag[i] - ai[i];
      idx += 4;

      /* Demote RHS from double to float. */
      CONVERT_DOUBLE4_FLOAT4(&t[idx], &b[idx]);
      LOAD_PS(&t[idx], XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        jdx = 4 * ((unsigned int)(*vi++));

        /* 4x4 Matrix-Vector product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[jdx], v)
        SSE_LOAD_PS(SSE_ARG_1, FLOAT_0, XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0, XMM6)
        SSE_SHUFFLE(XMM0, XMM0, 0x00)
        SSE_MULT_PS_M(XMM0, SSE_ARG_2, FLOAT_0)
        SSE_SUB_PS(XMM7, XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1, XMM6)
        SSE_SHUFFLE(XMM1, XMM1, 0x55)
        SSE_MULT_PS_M(XMM1, SSE_ARG_2, FLOAT_4)
        SSE_SUB_PS(XMM7, XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2, FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2, XMM6)
        SSE_SHUFFLE(XMM2, XMM2, 0xAA)
        SSE_MULT_PS_M(XMM2, SSE_ARG_2, FLOAT_8)
        SSE_SUB_PS(XMM7, XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3, XMM6)
        SSE_SHUFFLE(XMM3, XMM3, 0xFF)
        SSE_MULT_PS_M(XMM3, SSE_ARG_2, FLOAT_12)
        SSE_SUB_PS(XMM7, XMM3)
        SSE_INLINE_END_2

        v += 16;
      }
      v = aa + 16 * ai[++i];
      PREFETCH_NTA(v);
      STORE_PS(&t[idx], XMM7);
    }

    /* Backward solve the upper triangular factor.*/

    idt  = 4 * (n - 1);
    ai16 = 16 * diag[n - 1];
    v    = aa + ai16 + 16;
    for (i = n - 1; i >= 0;) {
      PREFETCH_NTA(&v[8]);
      vi = aj + diag[i] + 1;
      nz = ai[i + 1] - diag[i] - 1;

      LOAD_PS(&t[idt], XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        idx = 4 * ((unsigned int)(*vi++));

        /* 4x4 Matrix-Vector Product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[idx], v)
        SSE_LOAD_PS(SSE_ARG_1, FLOAT_0, XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0, XMM6)
        SSE_SHUFFLE(XMM0, XMM0, 0x00)
        SSE_MULT_PS_M(XMM0, SSE_ARG_2, FLOAT_0)
        SSE_SUB_PS(XMM7, XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1, XMM6)
        SSE_SHUFFLE(XMM1, XMM1, 0x55)
        SSE_MULT_PS_M(XMM1, SSE_ARG_2, FLOAT_4)
        SSE_SUB_PS(XMM7, XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2, FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2, XMM6)
        SSE_SHUFFLE(XMM2, XMM2, 0xAA)
        SSE_MULT_PS_M(XMM2, SSE_ARG_2, FLOAT_8)
        SSE_SUB_PS(XMM7, XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3, XMM6)
        SSE_SHUFFLE(XMM3, XMM3, 0xFF)
        SSE_MULT_PS_M(XMM3, SSE_ARG_2, FLOAT_12)
        SSE_SUB_PS(XMM7, XMM3)
        SSE_INLINE_END_2
        v += 16;
      }
      v    = aa + ai16;
      ai16 = 16 * diag[--i];
      PREFETCH_NTA(aa + ai16 + 16);
      /*
         Scale the result by the diagonal 4x4 block,
         which was inverted as part of the factorization
      */
      SSE_INLINE_BEGIN_3(v, &t[idt], aa + ai16)
      /* First Column */
      SSE_COPY_PS(XMM0, XMM7)
      SSE_SHUFFLE(XMM0, XMM0, 0x00)
      SSE_MULT_PS_M(XMM0, SSE_ARG_1, FLOAT_0)

      /* Second Column */
      SSE_COPY_PS(XMM1, XMM7)
      SSE_SHUFFLE(XMM1, XMM1, 0x55)
      SSE_MULT_PS_M(XMM1, SSE_ARG_1, FLOAT_4)
      SSE_ADD_PS(XMM0, XMM1)

      SSE_PREFETCH_NTA(SSE_ARG_3, FLOAT_24)

      /* Third Column */
      SSE_COPY_PS(XMM2, XMM7)
      SSE_SHUFFLE(XMM2, XMM2, 0xAA)
      SSE_MULT_PS_M(XMM2, SSE_ARG_1, FLOAT_8)
      SSE_ADD_PS(XMM0, XMM2)

      /* Fourth Column */
      SSE_COPY_PS(XMM3, XMM7)
      SSE_SHUFFLE(XMM3, XMM3, 0xFF)
      SSE_MULT_PS_M(XMM3, SSE_ARG_1, FLOAT_12)
      SSE_ADD_PS(XMM0, XMM3)

      SSE_STORE_PS(SSE_ARG_2, FLOAT_0, XMM0)
      SSE_INLINE_END_3

      v = aa + ai16 + 16;
      idt -= 4;
    }

    /* Convert t from single precision back to double precision (inplace)*/
    idt = 4 * (n - 1);
    for (i = n - 1; i >= 0; i--) {
      /*     CONVERT_FLOAT4_DOUBLE4(&x[idt],&t[idt]); */
      /* Unfortunately, CONVERT_ will count from 0 to 3 which doesn't work here. */
      PetscScalar *xtemp = &x[idt];
      MatScalar   *ttemp = &t[idt];
      xtemp[3]           = (PetscScalar)ttemp[3];
      xtemp[2]           = (PetscScalar)ttemp[2];
      xtemp[1]           = (PetscScalar)ttemp[1];
      xtemp[0]           = (PetscScalar)ttemp[0];
      idt -= 4;
    }

  } /* End of artificial scope. */
  PetscCall(VecRestoreArray(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  SSE_SCOPE_END;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion(Mat A, Vec bb, Vec xx)
{
  Mat_SeqBAIJ *a  = (Mat_SeqBAIJ *)A->data;
  int         *aj = a->j;
  int         *ai = a->i, n = a->mbs, *diag = a->diag;
  MatScalar   *aa = a->a;
  PetscScalar *x, *b;

  PetscFunctionBegin;
  SSE_SCOPE_BEGIN;
  /*
     Note: This code currently uses demotion of double
     to float when performing the mixed-mode computation.
     This may not be numerically reasonable for all applications.
  */
  PREFETCH_NTA(aa + 16 * ai[1]);

  PetscCall(VecGetArray(bb, &b));
  PetscCall(VecGetArray(xx, &x));
  {
    /* x will first be computed in single precision then promoted inplace to double */
    MatScalar *v, *t = (MatScalar *)x;
    int        nz, i, idt, ai16;
    int        jdx, idx;
    int       *vi;
    /* Forward solve the lower triangular factor. */

    /* First block is the identity. */
    idx = 0;
    CONVERT_DOUBLE4_FLOAT4(t, b);
    v = aa + 16 * ai[1];

    for (i = 1; i < n;) {
      PREFETCH_NTA(&v[8]);
      vi = aj + ai[i];
      nz = diag[i] - ai[i];
      idx += 4;

      /* Demote RHS from double to float. */
      CONVERT_DOUBLE4_FLOAT4(&t[idx], &b[idx]);
      LOAD_PS(&t[idx], XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        jdx = 4 * (*vi++);
        /*          jdx = *vi++; */

        /* 4x4 Matrix-Vector product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[jdx], v)
        SSE_LOAD_PS(SSE_ARG_1, FLOAT_0, XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0, XMM6)
        SSE_SHUFFLE(XMM0, XMM0, 0x00)
        SSE_MULT_PS_M(XMM0, SSE_ARG_2, FLOAT_0)
        SSE_SUB_PS(XMM7, XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1, XMM6)
        SSE_SHUFFLE(XMM1, XMM1, 0x55)
        SSE_MULT_PS_M(XMM1, SSE_ARG_2, FLOAT_4)
        SSE_SUB_PS(XMM7, XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2, FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2, XMM6)
        SSE_SHUFFLE(XMM2, XMM2, 0xAA)
        SSE_MULT_PS_M(XMM2, SSE_ARG_2, FLOAT_8)
        SSE_SUB_PS(XMM7, XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3, XMM6)
        SSE_SHUFFLE(XMM3, XMM3, 0xFF)
        SSE_MULT_PS_M(XMM3, SSE_ARG_2, FLOAT_12)
        SSE_SUB_PS(XMM7, XMM3)
        SSE_INLINE_END_2

        v += 16;
      }
      v = aa + 16 * ai[++i];
      PREFETCH_NTA(v);
      STORE_PS(&t[idx], XMM7);
    }

    /* Backward solve the upper triangular factor.*/

    idt  = 4 * (n - 1);
    ai16 = 16 * diag[n - 1];
    v    = aa + ai16 + 16;
    for (i = n - 1; i >= 0;) {
      PREFETCH_NTA(&v[8]);
      vi = aj + diag[i] + 1;
      nz = ai[i + 1] - diag[i] - 1;

      LOAD_PS(&t[idt], XMM7);

      while (nz--) {
        PREFETCH_NTA(&v[16]);
        idx = 4 * (*vi++);
        /*          idx = *vi++; */

        /* 4x4 Matrix-Vector Product with negative accumulation: */
        SSE_INLINE_BEGIN_2(&t[idx], v)
        SSE_LOAD_PS(SSE_ARG_1, FLOAT_0, XMM6)

        /* First Column */
        SSE_COPY_PS(XMM0, XMM6)
        SSE_SHUFFLE(XMM0, XMM0, 0x00)
        SSE_MULT_PS_M(XMM0, SSE_ARG_2, FLOAT_0)
        SSE_SUB_PS(XMM7, XMM0)

        /* Second Column */
        SSE_COPY_PS(XMM1, XMM6)
        SSE_SHUFFLE(XMM1, XMM1, 0x55)
        SSE_MULT_PS_M(XMM1, SSE_ARG_2, FLOAT_4)
        SSE_SUB_PS(XMM7, XMM1)

        SSE_PREFETCH_NTA(SSE_ARG_2, FLOAT_24)

        /* Third Column */
        SSE_COPY_PS(XMM2, XMM6)
        SSE_SHUFFLE(XMM2, XMM2, 0xAA)
        SSE_MULT_PS_M(XMM2, SSE_ARG_2, FLOAT_8)
        SSE_SUB_PS(XMM7, XMM2)

        /* Fourth Column */
        SSE_COPY_PS(XMM3, XMM6)
        SSE_SHUFFLE(XMM3, XMM3, 0xFF)
        SSE_MULT_PS_M(XMM3, SSE_ARG_2, FLOAT_12)
        SSE_SUB_PS(XMM7, XMM3)
        SSE_INLINE_END_2
        v += 16;
      }
      v    = aa + ai16;
      ai16 = 16 * diag[--i];
      PREFETCH_NTA(aa + ai16 + 16);
      /*
         Scale the result by the diagonal 4x4 block,
         which was inverted as part of the factorization
      */
      SSE_INLINE_BEGIN_3(v, &t[idt], aa + ai16)
      /* First Column */
      SSE_COPY_PS(XMM0, XMM7)
      SSE_SHUFFLE(XMM0, XMM0, 0x00)
      SSE_MULT_PS_M(XMM0, SSE_ARG_1, FLOAT_0)

      /* Second Column */
      SSE_COPY_PS(XMM1, XMM7)
      SSE_SHUFFLE(XMM1, XMM1, 0x55)
      SSE_MULT_PS_M(XMM1, SSE_ARG_1, FLOAT_4)
      SSE_ADD_PS(XMM0, XMM1)

      SSE_PREFETCH_NTA(SSE_ARG_3, FLOAT_24)

      /* Third Column */
      SSE_COPY_PS(XMM2, XMM7)
      SSE_SHUFFLE(XMM2, XMM2, 0xAA)
      SSE_MULT_PS_M(XMM2, SSE_ARG_1, FLOAT_8)
      SSE_ADD_PS(XMM0, XMM2)

      /* Fourth Column */
      SSE_COPY_PS(XMM3, XMM7)
      SSE_SHUFFLE(XMM3, XMM3, 0xFF)
      SSE_MULT_PS_M(XMM3, SSE_ARG_1, FLOAT_12)
      SSE_ADD_PS(XMM0, XMM3)

      SSE_STORE_PS(SSE_ARG_2, FLOAT_0, XMM0)
      SSE_INLINE_END_3

      v = aa + ai16 + 16;
      idt -= 4;
    }

    /* Convert t from single precision back to double precision (inplace)*/
    idt = 4 * (n - 1);
    for (i = n - 1; i >= 0; i--) {
      /*     CONVERT_FLOAT4_DOUBLE4(&x[idt],&t[idt]); */
      /* Unfortunately, CONVERT_ will count from 0 to 3 which doesn't work here. */
      PetscScalar *xtemp = &x[idt];
      MatScalar   *ttemp = &t[idt];
      xtemp[3]           = (PetscScalar)ttemp[3];
      xtemp[2]           = (PetscScalar)ttemp[2];
      xtemp[1]           = (PetscScalar)ttemp[1];
      xtemp[0]           = (PetscScalar)ttemp[0];
      idt -= 4;
    }

  } /* End of artificial scope. */
  PetscCall(VecRestoreArray(bb, &b));
  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(PetscLogFlops(2.0 * 16 * (a->nz) - 4.0 * A->cmap->n));
  SSE_SCOPE_END;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
