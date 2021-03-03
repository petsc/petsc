
/*
   Factorization code for BAIJ format.
 */
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
#include <immintrin.h>
#endif
/*
   Version for when blocks are 9 by 9
 */
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
PetscErrorCode MatLUFactorNumeric_SeqBAIJ_9_NaturalOrdering(Mat B,Mat A,const MatFactorInfo *info)
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
        if (pc[j]!=0.0) {
          flg = 1;
          break;
        }
      }
      if (flg) {
        pv = b->a + bs2*bdiag[row];
        /* PetscKernel_A_gets_A_times_B(bs,pc,pv,mwork); *pc = *pc * (*pv); */
        ierr = PetscKernel_A_gets_A_times_B_9(pc,pv,mwork);CHKERRQ(ierr);

        pj = b->j + bdiag[row+1]+1; /* begining of U(row,:) */
        pv = b->a + bs2*(bdiag[row+1]+1);
        nz = bdiag[row] - bdiag[row+1] - 1; /* num of entries inU(row,:), excluding diag */
        for (j=0; j<nz; j++) {
          /* PetscKernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j); */
          /* rtmp+bs2*pj[j] = rtmp+bs2*pj[j] - (*pc)*(pv+bs2*j) */
          v    = rtmp + bs2*pj[j];
          ierr = PetscKernel_A_gets_A_minus_B_times_C_9(v,pc,pv+81*j);CHKERRQ(ierr);
          /* pv incremented in PetscKernel_A_gets_A_minus_B_times_C_9 */
        }
        ierr = PetscLogFlops(1458*nz+1377);CHKERRQ(ierr); /* flops = 2*bs^3*nz + 2*bs^3 - bs2) */
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
    ierr = PetscKernel_A_gets_inverse_A_9(pv,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
    if (zeropivotdetected) C->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;

    /* U part */
    pv = b->a + bs2*(bdiag[i+1]+1);
    pj = b->j + bdiag[i+1]+1;
    nz = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nz; j++) {
      ierr = PetscArraycpy(pv+bs2*j,rtmp+bs2*pj[j],bs2);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(rtmp,mwork);CHKERRQ(ierr);

  C->ops->solve          = MatSolve_SeqBAIJ_9_NaturalOrdering;
  C->ops->solvetranspose = MatSolveTranspose_SeqBAIJ_N;
  C->assembled           = PETSC_TRUE;

  ierr = PetscLogFlops(1.333333333333*9*9*9*n);CHKERRQ(ierr); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_9_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ    *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode ierr;
  const PetscInt *ai=a->i,*aj=a->j,*adiag=a->diag,*vi;
  PetscInt       i,k,n=a->mbs;
  PetscInt       nz,bs=A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,*s,*t,*ls;
  const PetscScalar *b;
  __m256d a0,a1,a2,a3,a4,a5,w0,w1,w2,w3,s0,s1,s2,v0,v1,v2,v3;

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

    __m256d s0,s1,s2;
    s0 = _mm256_loadu_pd(s+0);
    s1 = _mm256_loadu_pd(s+4);
    s2 = _mm256_maskload_pd(s+8, _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63));

    for (k=0;k<nz;k++) {

      w0 = _mm256_set1_pd((t+bs*vi[k])[0]);
      a0 = _mm256_loadu_pd(&v[ 0]); s0 = _mm256_fnmadd_pd(a0,w0,s0);
      a1 = _mm256_loadu_pd(&v[ 4]); s1 = _mm256_fnmadd_pd(a1,w0,s1);
      a2 = _mm256_loadu_pd(&v[ 8]); s2 = _mm256_fnmadd_pd(a2,w0,s2);

      w1 = _mm256_set1_pd((t+bs*vi[k])[1]);
      a3 = _mm256_loadu_pd(&v[ 9]); s0 = _mm256_fnmadd_pd(a3,w1,s0);
      a4 = _mm256_loadu_pd(&v[13]); s1 = _mm256_fnmadd_pd(a4,w1,s1);
      a5 = _mm256_loadu_pd(&v[17]); s2 = _mm256_fnmadd_pd(a5,w1,s2);

      w2 = _mm256_set1_pd((t+bs*vi[k])[2]);
      a0 = _mm256_loadu_pd(&v[18]); s0 = _mm256_fnmadd_pd(a0,w2,s0);
      a1 = _mm256_loadu_pd(&v[22]); s1 = _mm256_fnmadd_pd(a1,w2,s1);
      a2 = _mm256_loadu_pd(&v[26]); s2 = _mm256_fnmadd_pd(a2,w2,s2);

      w3 = _mm256_set1_pd((t+bs*vi[k])[3]);
      a3 = _mm256_loadu_pd(&v[27]); s0 = _mm256_fnmadd_pd(a3,w3,s0);
      a4 = _mm256_loadu_pd(&v[31]); s1 = _mm256_fnmadd_pd(a4,w3,s1);
      a5 = _mm256_loadu_pd(&v[35]); s2 = _mm256_fnmadd_pd(a5,w3,s2);

      w0 = _mm256_set1_pd((t+bs*vi[k])[4]);
      a0 = _mm256_loadu_pd(&v[36]); s0 = _mm256_fnmadd_pd(a0,w0,s0);
      a1 = _mm256_loadu_pd(&v[40]); s1 = _mm256_fnmadd_pd(a1,w0,s1);
      a2 = _mm256_loadu_pd(&v[44]); s2 = _mm256_fnmadd_pd(a2,w0,s2);

      w1 = _mm256_set1_pd((t+bs*vi[k])[5]);
      a3 = _mm256_loadu_pd(&v[45]); s0 = _mm256_fnmadd_pd(a3,w1,s0);
      a4 = _mm256_loadu_pd(&v[49]); s1 = _mm256_fnmadd_pd(a4,w1,s1);
      a5 = _mm256_loadu_pd(&v[53]); s2 = _mm256_fnmadd_pd(a5,w1,s2);

      w2 = _mm256_set1_pd((t+bs*vi[k])[6]);
      a0 = _mm256_loadu_pd(&v[54]); s0 = _mm256_fnmadd_pd(a0,w2,s0);
      a1 = _mm256_loadu_pd(&v[58]); s1 = _mm256_fnmadd_pd(a1,w2,s1);
      a2 = _mm256_loadu_pd(&v[62]); s2 = _mm256_fnmadd_pd(a2,w2,s2);

      w3 = _mm256_set1_pd((t+bs*vi[k])[7]);
      a3 = _mm256_loadu_pd(&v[63]); s0 = _mm256_fnmadd_pd(a3,w3,s0);
      a4 = _mm256_loadu_pd(&v[67]); s1 = _mm256_fnmadd_pd(a4,w3,s1);
      a5 = _mm256_loadu_pd(&v[71]); s2 = _mm256_fnmadd_pd(a5,w3,s2);

      w0 = _mm256_set1_pd((t+bs*vi[k])[8]);
      a0 = _mm256_loadu_pd(&v[72]); s0 = _mm256_fnmadd_pd(a0,w0,s0);
      a1 = _mm256_loadu_pd(&v[76]); s1 = _mm256_fnmadd_pd(a1,w0,s1);
      a2 = _mm256_maskload_pd(v+80, _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63));
      s2 = _mm256_fnmadd_pd(a2,w0,s2);
      v += bs2;
    }
         _mm256_storeu_pd(&s[0], s0);
         _mm256_storeu_pd(&s[4], s1);
         _mm256_maskstore_pd(&s[8], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), s2);
  }

  /* backward solve the upper triangular */
  ls = a->solve_work + A->cmap->n;
  for (i=n-1; i>=0; i--) {
    v    = aa + bs2*(adiag[i+1]+1);
    vi   = aj + adiag[i+1]+1;
    nz   = adiag[i] - adiag[i+1]-1;
    ierr = PetscArraycpy(ls,t+i*bs,bs);CHKERRQ(ierr);

    s0 = _mm256_loadu_pd(ls+0);
    s1 = _mm256_loadu_pd(ls+4);
    s2 = _mm256_maskload_pd(ls+8, _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63));

    for (k=0; k<nz; k++) {

      w0 = _mm256_set1_pd((t+bs*vi[k])[0]);
      a0 = _mm256_loadu_pd(&v[ 0]); s0 = _mm256_fnmadd_pd(a0,w0,s0);
      a1 = _mm256_loadu_pd(&v[ 4]); s1 = _mm256_fnmadd_pd(a1,w0,s1);
      a2 = _mm256_loadu_pd(&v[ 8]); s2 = _mm256_fnmadd_pd(a2,w0,s2);

      /* v += 9; */
      w1 = _mm256_set1_pd((t+bs*vi[k])[1]);
      a3 = _mm256_loadu_pd(&v[ 9]); s0 = _mm256_fnmadd_pd(a3,w1,s0);
      a4 = _mm256_loadu_pd(&v[13]); s1 = _mm256_fnmadd_pd(a4,w1,s1);
      a5 = _mm256_loadu_pd(&v[17]); s2 = _mm256_fnmadd_pd(a5,w1,s2);

      /* v += 9; */
      w2 = _mm256_set1_pd((t+bs*vi[k])[2]);
      a0 = _mm256_loadu_pd(&v[18]); s0 = _mm256_fnmadd_pd(a0,w2,s0);
      a1 = _mm256_loadu_pd(&v[22]); s1 = _mm256_fnmadd_pd(a1,w2,s1);
      a2 = _mm256_loadu_pd(&v[26]); s2 = _mm256_fnmadd_pd(a2,w2,s2);

      /* v += 9; */
      w3 = _mm256_set1_pd((t+bs*vi[k])[3]);
      a3 = _mm256_loadu_pd(&v[27]); s0 = _mm256_fnmadd_pd(a3,w3,s0);
      a4 = _mm256_loadu_pd(&v[31]); s1 = _mm256_fnmadd_pd(a4,w3,s1);
      a5 = _mm256_loadu_pd(&v[35]); s2 = _mm256_fnmadd_pd(a5,w3,s2);

      /* v += 9; */
      w0 = _mm256_set1_pd((t+bs*vi[k])[4]);
      a0 = _mm256_loadu_pd(&v[36]); s0 = _mm256_fnmadd_pd(a0,w0,s0);
      a1 = _mm256_loadu_pd(&v[40]); s1 = _mm256_fnmadd_pd(a1,w0,s1);
      a2 = _mm256_loadu_pd(&v[44]); s2 = _mm256_fnmadd_pd(a2,w0,s2);

      /* v += 9; */
      w1 = _mm256_set1_pd((t+bs*vi[k])[5]);
      a3 = _mm256_loadu_pd(&v[45]); s0 = _mm256_fnmadd_pd(a3,w1,s0);
      a4 = _mm256_loadu_pd(&v[49]); s1 = _mm256_fnmadd_pd(a4,w1,s1);
      a5 = _mm256_loadu_pd(&v[53]); s2 = _mm256_fnmadd_pd(a5,w1,s2);

      /* v += 9; */
      w2 = _mm256_set1_pd((t+bs*vi[k])[6]);
      a0 = _mm256_loadu_pd(&v[54]); s0 = _mm256_fnmadd_pd(a0,w2,s0);
      a1 = _mm256_loadu_pd(&v[58]); s1 = _mm256_fnmadd_pd(a1,w2,s1);
      a2 = _mm256_loadu_pd(&v[62]); s2 = _mm256_fnmadd_pd(a2,w2,s2);

      /* v += 9; */
      w3 = _mm256_set1_pd((t+bs*vi[k])[7]);
      a3 = _mm256_loadu_pd(&v[63]); s0 = _mm256_fnmadd_pd(a3,w3,s0);
      a4 = _mm256_loadu_pd(&v[67]); s1 = _mm256_fnmadd_pd(a4,w3,s1);
      a5 = _mm256_loadu_pd(&v[71]); s2 = _mm256_fnmadd_pd(a5,w3,s2);

      /* v += 9; */
      w0 = _mm256_set1_pd((t+bs*vi[k])[8]);
      a0 = _mm256_loadu_pd(&v[72]); s0 = _mm256_fnmadd_pd(a0,w0,s0);
      a1 = _mm256_loadu_pd(&v[76]); s1 = _mm256_fnmadd_pd(a1,w0,s1);
      a2 = _mm256_maskload_pd(v+80, _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63));
      s2 = _mm256_fnmadd_pd(a2,w0,s2);
      v += bs2;
    }

         _mm256_storeu_pd(&ls[0], s0); _mm256_storeu_pd(&ls[4], s1); _mm256_maskstore_pd(&ls[8], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), s2);

    w0 = _mm256_setzero_pd(); w1 = _mm256_setzero_pd(); w2 = _mm256_setzero_pd();

    /* first row */
    v0 = _mm256_set1_pd(ls[0]);
    a0 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[0]); w0 = _mm256_fmadd_pd(a0,v0,w0);
    a1 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[4]); w1 = _mm256_fmadd_pd(a1,v0,w1);
    a2 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[8]); w2 = _mm256_fmadd_pd(a2,v0,w2);

    /* second row */
    v1 = _mm256_set1_pd(ls[1]);
    a3 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[9]); w0 = _mm256_fmadd_pd(a3,v1,w0);
    a4 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[13]); w1 = _mm256_fmadd_pd(a4,v1,w1);
    a5 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[17]); w2 = _mm256_fmadd_pd(a5,v1,w2);

    /* third row */
    v2 = _mm256_set1_pd(ls[2]);
    a0 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[18]); w0 = _mm256_fmadd_pd(a0,v2,w0);
    a1 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[22]); w1 = _mm256_fmadd_pd(a1,v2,w1);
    a2 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[26]); w2 = _mm256_fmadd_pd(a2,v2,w2);

    /* fourth row */
    v3 = _mm256_set1_pd(ls[3]);
    a3 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[27]); w0 = _mm256_fmadd_pd(a3,v3,w0);
    a4 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[31]); w1 = _mm256_fmadd_pd(a4,v3,w1);
    a5 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[35]); w2 = _mm256_fmadd_pd(a5,v3,w2);

    /* fifth row */
    v0 = _mm256_set1_pd(ls[4]);
    a0 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[36]); w0 = _mm256_fmadd_pd(a0,v0,w0);
    a1 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[40]); w1 = _mm256_fmadd_pd(a1,v0,w1);
    a2 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[44]); w2 = _mm256_fmadd_pd(a2,v0,w2);

    /* sixth row */
    v1 = _mm256_set1_pd(ls[5]);
    a3 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[45]); w0 = _mm256_fmadd_pd(a3,v1,w0);
    a4 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[49]); w1 = _mm256_fmadd_pd(a4,v1,w1);
    a5 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[53]); w2 = _mm256_fmadd_pd(a5,v1,w2);

    /* seventh row */
    v2 = _mm256_set1_pd(ls[6]);
    a0 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[54]); w0 = _mm256_fmadd_pd(a0,v2,w0);
    a1 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[58]); w1 = _mm256_fmadd_pd(a1,v2,w1);
    a2 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[62]); w2 = _mm256_fmadd_pd(a2,v2,w2);

    /* eighth row */
    v3 = _mm256_set1_pd(ls[7]);
    a3 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[63]); w0 = _mm256_fmadd_pd(a3,v3,w0);
    a4 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[67]); w1 = _mm256_fmadd_pd(a4,v3,w1);
    a5 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[71]); w2 = _mm256_fmadd_pd(a5,v3,w2);

    /* ninth row */
    v0 = _mm256_set1_pd(ls[8]);
    a3 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[72]); w0 = _mm256_fmadd_pd(a3,v0,w0);
    a4 = _mm256_loadu_pd(&(aa+bs2*adiag[i])[76]); w1 = _mm256_fmadd_pd(a4,v0,w1);
    a2 = _mm256_maskload_pd((&(aa+bs2*adiag[i])[80]), _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63));
    w2 = _mm256_fmadd_pd(a2,v0,w2);

    _mm256_storeu_pd(&(t+i*bs)[0], w0); _mm256_storeu_pd(&(t+i*bs)[4], w1); _mm256_maskstore_pd(&(t+i*bs)[8], _mm256_set_epi64x(0LL, 0LL, 0LL, 1LL<<63), w2);

    ierr = PetscArraycpy(x+i*bs,t+i*bs,bs);CHKERRQ(ierr);
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(a->bs2)*(a->nz) - A->rmap->bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
