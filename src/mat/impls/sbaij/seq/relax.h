/*
    This is included by sbaij.c to generate unsigned short and regular versions of these two functions
*/

/* We cut-and-past below from aij.h to make a "no_function" version of PetscSparseDensePlusDot().
 * This is necessary because the USESHORT case cannot use the inlined functions that may be employed. */

#if defined(PETSC_KERNEL_USE_UNROLL_4)
  #define PetscSparseDensePlusDot_no_function(sum, r, xv, xi, nnz) \
    do { \
      if (nnz > 0) { \
        PetscInt nnz2 = nnz, rem = nnz & 0x3; \
        switch (rem) { \
        case 3: \
          sum += *xv++ * r[*xi++]; \
        case 2: \
          sum += *xv++ * r[*xi++]; \
        case 1: \
          sum += *xv++ * r[*xi++]; \
          nnz2 -= rem; \
        } \
        while (nnz2 > 0) { \
          sum += xv[0] * r[xi[0]] + xv[1] * r[xi[1]] + xv[2] * r[xi[2]] + xv[3] * r[xi[3]]; \
          xv += 4; \
          xi += 4; \
          nnz2 -= 4; \
        } \
        xv -= nnz; \
        xi -= nnz; \
      } \
    } while (0)

#elif defined(PETSC_KERNEL_USE_UNROLL_2)
  #define PetscSparseDensePlusDot_no_function(sum, r, xv, xi, nnz) \
    do { \
      PetscInt __i, __i1, __i2; \
      for (__i = 0; __i < nnz - 1; __i += 2) { \
        __i1 = xi[__i]; \
        __i2 = xi[__i + 1]; \
        sum += (xv[__i] * r[__i1] + xv[__i + 1] * r[__i2]); \
      } \
      if (nnz & 0x1) sum += xv[__i] * r[xi[__i]]; \
    } while (0)

#else
  #define PetscSparseDensePlusDot_no_function(sum, r, xv, xi, nnz) \
    do { \
      PetscInt __i; \
      for (__i = 0; __i < nnz; __i++) sum += xv[__i] * r[xi[__i]]; \
    } while (0)
#endif

#if defined(USESHORT)
PetscErrorCode MatMult_SeqSBAIJ_1_ushort(Mat A, Vec xx, Vec zz)
#else
PetscErrorCode MatMult_SeqSBAIJ_1(Mat A, Vec xx, Vec zz)
#endif
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ *)A->data;
  const PetscScalar *x;
  PetscScalar       *z, x1, sum;
  const MatScalar   *v;
  MatScalar          vj;
  PetscInt           mbs = a->mbs, i, j, nz;
  const PetscInt    *ai  = a->i;
#if defined(USESHORT)
  const unsigned short *ib = a->jshort;
  unsigned short        ibt;
#else
  const PetscInt *ib = a->j;
  PetscInt        ibt;
#endif
  PetscInt nonzerorow = 0, jmin;
#if defined(PETSC_USE_COMPLEX)
  const int aconj = A->hermitian == PETSC_BOOL3_TRUE;
#else
  const int aconj = 0;
#endif

  PetscFunctionBegin;
  PetscCall(VecSet(zz, 0.0));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  v = a->a;
  for (i = 0; i < mbs; i++) {
    nz = ai[i + 1] - ai[i]; /* length of i_th row of A */
    if (!nz) continue;      /* Move to the next row if the current row is empty */
    nonzerorow++;
    sum  = 0.0;
    jmin = 0;
    x1   = x[i];
    if (ib[0] == i) {
      sum = v[0] * x1; /* diagonal term */
      jmin++;
    }
    PetscPrefetchBlock(ib + nz, nz, 0, PETSC_PREFETCH_HINT_NTA); /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v + nz, nz, 0, PETSC_PREFETCH_HINT_NTA);  /* Entries for the next row */
    if (aconj) {
      for (j = jmin; j < nz; j++) {
        ibt = ib[j];
        vj  = v[j];
        z[ibt] += PetscConj(vj) * x1; /* (strict lower triangular part of A)*x  */
        sum += vj * x[ibt];           /* (strict upper triangular part of A)*x  */
      }
    } else {
      for (j = jmin; j < nz; j++) {
        ibt = ib[j];
        vj  = v[j];
        z[ibt] += vj * x1;  /* (strict lower triangular part of A)*x  */
        sum += vj * x[ibt]; /* (strict upper triangular part of A)*x  */
      }
    }
    z[i] += sum;
    v += nz;
    ib += nz;
  }

  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscCall(PetscLogFlops(2.0 * (2.0 * a->nz - nonzerorow) - nonzerorow));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(USESHORT)
PetscErrorCode MatSOR_SeqSBAIJ_ushort(Mat A, Vec bb, PetscReal omega, MatSORType flag, PetscReal fshift, PetscInt its, PetscInt lits, Vec xx)
#else
PetscErrorCode MatSOR_SeqSBAIJ(Mat A, Vec bb, PetscReal omega, MatSORType flag, PetscReal fshift, PetscInt its, PetscInt lits, Vec xx)
#endif
{
  Mat_SeqSBAIJ      *a  = (Mat_SeqSBAIJ *)A->data;
  const MatScalar   *aa = a->a, *v, *v1, *aidiag;
  PetscScalar       *x, *t, sum;
  const PetscScalar *b;
  MatScalar          tmp;
  PetscInt           m = a->mbs, bs = A->rmap->bs, j;
  const PetscInt    *ai = a->i;
#if defined(USESHORT)
  const unsigned short *aj = a->jshort, *vj, *vj1;
#else
  const PetscInt *aj = a->j, *vj, *vj1;
#endif
  PetscInt nz, nz1, i;

  PetscFunctionBegin;
  if (fshift == -1.0) fshift = 0.0; /* negative fshift indicates do not error on zero diagonal; this code never errors on zero diagonal */
  PetscCheck(!(flag & SOR_EISENSTAT), PETSC_COMM_SELF, PETSC_ERR_SUP, "No support yet for Eisenstat");

  its = its * lits;
  PetscCheck(its > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive", its, lits);

  PetscCheck(bs <= 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "SSOR for block size > 1 is not yet implemented");

  PetscCall(VecGetArray(xx, &x));
  PetscCall(VecGetArrayRead(bb, &b));

  if (!a->idiagvalid) {
    if (!a->idiag) PetscCall(PetscMalloc1(m, &a->idiag));
    for (i = 0; i < a->mbs; i++) a->idiag[i] = 1.0 / a->a[a->i[i]];
    a->idiagvalid = PETSC_TRUE;
  }

  if (!a->sor_work) PetscCall(PetscMalloc1(m, &a->sor_work));
  t = a->sor_work;

  aidiag = a->idiag;

  if (flag == SOR_APPLY_UPPER) {
    /* apply (U + D/omega) to the vector */
    PetscScalar d;
    for (i = 0; i < m; i++) {
      d   = fshift + aa[ai[i]];
      nz  = ai[i + 1] - ai[i] - 1;
      vj  = aj + ai[i] + 1;
      v   = aa + ai[i] + 1;
      sum = b[i] * d / omega;
#ifdef USESHORT
      PetscSparseDensePlusDot_no_function(sum, b, v, vj, nz);
#else
      PetscSparseDensePlusDot(sum, b, v, vj, nz);
#endif
      x[i] = sum;
    }
    PetscCall(PetscLogFlops(a->nz));
  }

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      PetscCall(PetscArraycpy(t, b, m));

      v  = aa + 1;
      vj = aj + 1;
      for (i = 0; i < m; i++) {
        nz  = ai[i + 1] - ai[i] - 1;
        tmp = -(x[i] = omega * t[i] * aidiag[i]);
        for (j = 0; j < nz; j++) t[vj[j]] += tmp * v[j];
        v += nz + 1;
        vj += nz + 1;
      }
      PetscCall(PetscLogFlops(2.0 * a->nz));
    }

    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      PetscInt nz2;
      if (!(flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP)) {
        v  = aa + ai[m] - 1;
        vj = aj + ai[m] - 1;
        for (i = m - 1; i >= 0; i--) {
          sum = b[i];
          nz  = ai[i + 1] - ai[i] - 1;
          {
            PetscInt __i;
            for (__i = 0; __i < nz; __i++) sum -= v[-__i] * x[vj[-__i]];
          }
          x[i] = omega * sum * aidiag[i];
          v -= nz + 1;
          vj -= nz + 1;
        }
        PetscCall(PetscLogFlops(2.0 * a->nz));
      } else {
        v  = aa + ai[m - 1] + 1;
        vj = aj + ai[m - 1] + 1;
        nz = 0;
        for (i = m - 1; i >= 0; i--) {
          sum = t[i];
          nz2 = ai[i] - ai[PetscMax(i - 1, 0)] - 1; /* avoid referencing ai[-1], nonsense nz2 is okay on last iteration */
          PETSC_Prefetch(v - nz2 - 1, 0, PETSC_PREFETCH_HINT_NTA);
          PETSC_Prefetch(vj - nz2 - 1, 0, PETSC_PREFETCH_HINT_NTA);
          PetscSparseDenseMinusDot(sum, x, v, vj, nz);
          x[i] = (1 - omega) * x[i] + omega * sum * aidiag[i];
          nz   = nz2;
          v -= nz + 1;
          vj -= nz + 1;
        }
        PetscCall(PetscLogFlops(2.0 * a->nz));
      }
    }
    its--;
  }

  while (its--) {
    /*
       forward sweep:
       for i=0,...,m-1:
         sum[i] = (b[i] - U(i,:)x)/d[i];
         x[i]   = (1-omega)x[i] + omega*sum[i];
         b      = b - x[i]*U^T(i,:);

    */
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      PetscCall(PetscArraycpy(t, b, m));

      for (i = 0; i < m; i++) {
        v   = aa + ai[i] + 1;
        v1  = v;
        vj  = aj + ai[i] + 1;
        vj1 = vj;
        nz  = ai[i + 1] - ai[i] - 1;
        nz1 = nz;
        sum = t[i];
        while (nz1--) sum -= (*v1++) * x[*vj1++];
        x[i] = (1 - omega) * x[i] + omega * sum * aidiag[i];
        while (nz--) t[*vj++] -= x[i] * (*v++);
      }
      PetscCall(PetscLogFlops(4.0 * a->nz));
    }

    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      /*
       backward sweep:
       b = b - x[i]*U^T(i,:), i=0,...,n-2
       for i=m-1,...,0:
         sum[i] = (b[i] - U(i,:)x)/d[i];
         x[i]   = (1-omega)x[i] + omega*sum[i];
      */
      /* if there was a forward sweep done above then I thing the next two for loops are not needed */
      PetscCall(PetscArraycpy(t, b, m));

      for (i = 0; i < m - 1; i++) { /* update rhs */
        v  = aa + ai[i] + 1;
        vj = aj + ai[i] + 1;
        nz = ai[i + 1] - ai[i] - 1;
        while (nz--) t[*vj++] -= x[i] * (*v++);
      }
      PetscCall(PetscLogFlops(2.0 * (a->nz - m)));
      for (i = m - 1; i >= 0; i--) {
        v   = aa + ai[i] + 1;
        vj  = aj + ai[i] + 1;
        nz  = ai[i + 1] - ai[i] - 1;
        sum = t[i];
        while (nz--) sum -= x[*vj++] * (*v++);
        x[i] = (1 - omega) * x[i] + omega * sum * aidiag[i];
      }
      PetscCall(PetscLogFlops(2.0 * (a->nz + m)));
    }
  }

  PetscCall(VecRestoreArray(xx, &x));
  PetscCall(VecRestoreArrayRead(bb, &b));
  PetscFunctionReturn(PETSC_SUCCESS);
}
