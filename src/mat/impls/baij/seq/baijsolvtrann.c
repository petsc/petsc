#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/* ----------------------------------------------------------- */
PetscErrorCode MatSolveTranspose_SeqBAIJ_N_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout,*ai=a->i,*aj=a->j,*vi;
  PetscInt          i,nz,j;
  const PetscInt    n  =a->mbs,bs=A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,*t,*ls;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      t[i*bs+j] = b[c[i]*bs+j];
    }
  }


  /* forward solve the upper triangular transpose */
  ls = a->solve_work + A->cmap->n;
  for (i=0; i<n; i++) {
    ierr = PetscArraycpy(ls,t+i*bs,bs);CHKERRQ(ierr);
    PetscKernel_w_gets_transA_times_v(bs,ls,aa+bs2*a->diag[i],t+i*bs);
    v  = aa + bs2*(a->diag[i] + 1);
    vi = aj + a->diag[i] + 1;
    nz = ai[i+1] - a->diag[i] - 1;
    while (nz--) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs,t+bs*(*vi++),v,t+i*bs);
      v += bs2;
    }
  }

  /* backward solve the lower triangular transpose */
  for (i=n-1; i>=0; i--) {
    v  = aa + bs2*ai[i];
    vi = aj + ai[i];
    nz = a->diag[i] - ai[i];
    while (nz--) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs,t+bs*(*vi++),v,t+i*bs);
      v += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      x[bs*r[i]+j]   = t[bs*i+j];
    }
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(a->bs2)*(a->nz) - A->rmap->bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_N(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    n=a->mbs,*ai=a->i,*aj=a->j,*vi,*diag=a->diag;
  PetscInt          i,j,nz;
  const PetscInt    bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,*t,*ls;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      t[i*bs+j] = b[c[i]*bs+j];
    }
  }


  /* forward solve the upper triangular transpose */
  ls = a->solve_work + A->cmap->n;
  for (i=0; i<n; i++) {
    ierr = PetscArraycpy(ls,t+i*bs,bs);CHKERRQ(ierr);
    PetscKernel_w_gets_transA_times_v(bs,ls,aa+bs2*diag[i],t+i*bs);
    v  = aa + bs2*(diag[i] - 1);
    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs,t+bs*(vi[j]),v,t+i*bs);
      v -= bs2;
    }
  }

  /* backward solve the lower triangular transpose */
  for (i=n-1; i>=0; i--) {
    v  = aa + bs2*ai[i];
    vi = aj + ai[i];
    nz = ai[i+1] - ai[i];
    for (j=0; j<nz; j++) {
      PetscKernel_v_gets_v_minus_transA_times_w(bs,t+bs*(vi[j]),v,t+i*bs);
      v += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    for (j=0; j<bs; j++) {
      x[bs*r[i]+j]   = t[bs*i+j];
    }
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(a->bs2)*(a->nz) - A->rmap->bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

