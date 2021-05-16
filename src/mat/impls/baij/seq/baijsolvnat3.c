#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*ai=a->i,*aj=a->j;
  PetscErrorCode    ierr;
  const PetscInt    *diag = a->diag,*vi;
  const MatScalar   *aa   =a->a,*v;
  PetscScalar       *x,s1,s2,s3,x1,x2,x3;
  const PetscScalar *b;
  PetscInt          jdx,idt,idx,nz,i;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[0]; x[1] = b[1]; x[2] = b[2];
  for (i=1; i<n; i++) {
    v    =  aa      + 9*ai[i];
    vi   =  aj      + ai[i];
    nz   =  diag[i] - ai[i];
    idx +=  3;
    s1   =  b[idx];s2 = b[1+idx];s3 = b[2+idx];
    while (nz--) {
      jdx = 3*(*vi++);
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;
      v  += 9;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 9*diag[i] + 9;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 3*i;
    s1  = x[idt];  s2 = x[1+idt];
    s3  = x[2+idt];
    while (nz--) {
      idx = 3*(*vi++);
      x1  = x[idx];   x2 = x[1+idx];x3    = x[2+idx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;
      v  += 9;
    }
    v        = aa +  9*diag[i];
    x[idt]   = v[0]*s1 + v[3]*s2 + v[6]*s3;
    x[1+idt] = v[1]*s1 + v[4]*s2 + v[7]*s3;
    x[2+idt] = v[2]*s1 + v[5]*s2 + v[8]*s3;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*9*(a->nz) - 3.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,jdx,idt;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,x1,x2,x3;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];s3 = x[2+idt];

    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[3]*s2 + v[6]*s3;
    x[1+idt] = v[1]*s1 + v[4]*s2 + v[7]*s3;
    x[2+idt] = v[2]*s1 + v[5]*s2 + v[8]*s3;

  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqBAIJ_3_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,jdx;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,x1,x2,x3;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(1.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqBAIJ_3_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*aj=a->j,*adiag=a->diag;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,idt;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,x1,x2,x3;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = b[idt];  s2 = b[1+idt];s3 = b[2+idt];

    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];
      s1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      s2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      s3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;

      v +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[3]*s2 + v[6]*s3;
    x[1+idt] = v[1]*s1 + v[4]*s2 + v[7]*s3;
    x[2+idt] = v[2]*s1 + v[5]*s2 + v[8]*s3;

  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
