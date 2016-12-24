#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSolveTranspose_SeqBAIJ_1(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a    = (Mat_SeqBAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  const PetscInt    *rout,*cout,*r,*c,*adiag = a->diag,*ai = a->i,*aj = a->j,*vi;
  PetscInt          i,n = a->mbs,j;
  PetscInt          nz;
  PetscScalar       *x,*tmp,s1;
  const MatScalar   *aa = a->a,*v;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) tmp[i] = b[c[i]];

  /* forward solve the U^T */
  for (i=0; i<n; i++) {
    v   = aa + adiag[i+1] + 1;
    vi  = aj + adiag[i+1] + 1;
    nz  = adiag[i] - adiag[i+1] - 1;
    s1  = tmp[i];
    s1 *= v[nz];  /* multiply by inverse of diagonal entry */
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
    tmp[i] = s1;
  }

  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v  = aa + ai[i];
    vi = aj + ai[i];
    nz = ai[i+1] - ai[i];
    s1 = tmp[i];
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
  }

  /* copy tmp into x according to permutation */
  for (i=0; i<n; i++) x[r[i]] = tmp[i];

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);

  ierr = PetscLogFlops(2.0*a->nz-A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_1_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) t[i] = b[c[i]];

  /* forward solve the U^T */
  for (i=0; i<n; i++) {

    v = aa + diag[i];
    /* multiply by the inverse of the block diagonal */
    s1 = (*v++)*t[i];
    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      t[*vi++] -= (*v++)*s1;
    }
    t[i] = s1;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v  = aa + diag[i] - 1;
    vi = aj + diag[i] - 1;
    nz = diag[i] - ai[i];
    s1 = t[i];
    while (nz--) {
      t[*vi--] -=  (*v--)*s1;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) x[r[i]] = t[i];

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*(a->nz) - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_2_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,ii,ic,ir,oidx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,x1,x2,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ic      = 2*c[i];
    t[ii]   = b[ic];
    t[ii+1] = b[ic+1];
    ii     += 2;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 4*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx];
    s1 = v[0]*x1  +  v[1]*x2;
    s2 = v[2]*x1  +  v[3]*x2;
    v += 4;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 2*(*vi++);
      t[oidx]   -= v[0]*s1  +  v[1]*s2;
      t[oidx+1] -= v[2]*s1  +  v[3]*s2;
      v         += 4;
    }
    t[idx] = s1;t[1+idx] = s2;
    idx   += 2;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 4*diag[i] - 4;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 2*i;
    s1  = t[idt];  s2 = t[1+idt];
    while (nz--) {
      idx       = 2*(*vi--);
      t[idx]   -=  v[0]*s1 +  v[1]*s2;
      t[idx+1] -=  v[2]*s1 +  v[3]*s2;
      v        -= 4;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ir      = 2*r[i];
    x[ir]   = t[ii];
    x[ir+1] = t[ii+1];
    ii     += 2;
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*4*(a->nz) - 2.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_2(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  IS                iscol=a->col,isrow=a->row;
  const PetscInt    n    =a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          nz,idx,idt,j,i,oidx,ii,ic,ir;
  const PetscInt    bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,x1,x2,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    ii    = bs*i; ic = bs*c[i];
    t[ii] = b[ic]; t[ii+1] = b[ic+1];
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx];
    s1 = v[0]*x1  +  v[1]*x2;
    s2 = v[2]*x1  +  v[3]*x2;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      t[oidx]   -= v[0]*s1  +  v[1]*s2;
      t[oidx+1] -= v[2]*s1  +  v[3]*s2;
      v         -= bs2;
    }
    t[idx] = s1;t[1+idx] = s2;
    idx   += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = t[idt];  s2 = t[1+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      t[idx]   -=  v[0]*s1 +  v[1]*s2;
      t[idx+1] -=  v[2]*s1 +  v[3]*s2;
      v        += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    ii    = bs*i;  ir = bs*r[i];
    x[ir] = t[ii];  x[ir+1] = t[ii+1];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_3_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,ii,ic,ir,oidx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,x1,x2,x3,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ic      = 3*c[i];
    t[ii]   = b[ic];
    t[ii+1] = b[ic+1];
    t[ii+2] = b[ic+2];
    ii     += 3;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 9*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3;
    s2 = v[3]*x1  +  v[4]*x2 +  v[5]*x3;
    s3 = v[6]*x1  +  v[7]*x2 + v[8]*x3;
    v += 9;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 3*(*vi++);
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3;
      t[oidx+1] -= v[3]*s1  +  v[4]*s2 +  v[5]*s3;
      t[oidx+2] -= v[6]*s1 + v[7]*s2 + v[8]*s3;
      v         += 9;
    }
    t[idx] = s1;t[1+idx] = s2; t[2+idx] = s3;
    idx   += 3;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 9*diag[i] - 9;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 3*i;
    s1  = t[idt];  s2 = t[1+idt]; s3 = t[2+idt];
    while (nz--) {
      idx       = 3*(*vi--);
      t[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3;
      t[idx+1] -=  v[3]*s1 +  v[4]*s2 +  v[5]*s3;
      t[idx+2] -= v[6]*s1 + v[7]*s2 + v[8]*s3;
      v        -= 9;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ir      = 3*r[i];
    x[ir]   = t[ii];
    x[ir+1] = t[ii+1];
    x[ir+2] = t[ii+2];
    ii     += 3;
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*9*(a->nz) - 3.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_3(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  IS                iscol=a->col,isrow=a->row;
  const PetscInt    n    =a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          nz,idx,idt,j,i,oidx,ii,ic,ir;
  const PetscInt    bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,x1,x2,x3,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    ii    = bs*i; ic = bs*c[i];
    t[ii] = b[ic]; t[ii+1] = b[ic+1]; t[ii+2] = b[ic+2];
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3;
    s2 = v[3]*x1  +  v[4]*x2 +  v[5]*x3;
    s3 = v[6]*x1  +  v[7]*x2 + v[8]*x3;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3;
      t[oidx+1] -= v[3]*s1  +  v[4]*s2 +  v[5]*s3;
      t[oidx+2] -= v[6]*s1 + v[7]*s2 + v[8]*s3;
      v         -= bs2;
    }
    t[idx] = s1;t[1+idx] = s2;  t[2+idx] = s3;
    idx   += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = t[idt];  s2 = t[1+idt];  s3 = t[2+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      t[idx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3;
      t[idx+1] -= v[3]*s1  +  v[4]*s2 +  v[5]*s3;
      t[idx+2] -= v[6]*s1 + v[7]*s2 + v[8]*s3;
      v        += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    ii    = bs*i;  ir = bs*r[i];
    x[ir] = t[ii];  x[ir+1] = t[ii+1]; x[ir+2] = t[ii+2];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_4_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,ii,ic,ir,oidx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,x1,x2,x3,x4,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ic      = 4*c[i];
    t[ii]   = b[ic];
    t[ii+1] = b[ic+1];
    t[ii+2] = b[ic+2];
    t[ii+3] = b[ic+3];
    ii     += 4;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 16*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx]; x4 = t[3+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4;
    s2 = v[4]*x1  +  v[5]*x2 +  v[6]*x3 +  v[7]*x4;
    s3 = v[8]*x1  +  v[9]*x2 + v[10]*x3 + v[11]*x4;
    s4 = v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
    v += 16;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 4*(*vi++);
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4;
      t[oidx+1] -= v[4]*s1  +  v[5]*s2 +  v[6]*s3 +  v[7]*s4;
      t[oidx+2] -= v[8]*s1 + v[9]*s2 + v[10]*s3 + v[11]*s4;
      t[oidx+3] -= v[12]*s1 + v[13]*s2 + v[14]*s3 + v[15]*s4;
      v         += 16;
    }
    t[idx] = s1;t[1+idx] = s2; t[2+idx] = s3;t[3+idx] = s4;
    idx   += 4;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 16*diag[i] - 16;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 4*i;
    s1  = t[idt];  s2 = t[1+idt]; s3 = t[2+idt];s4 = t[3+idt];
    while (nz--) {
      idx       = 4*(*vi--);
      t[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3 +  v[3]*s4;
      t[idx+1] -=  v[4]*s1 +  v[5]*s2 +  v[6]*s3 +  v[7]*s4;
      t[idx+2] -= v[8]*s1 + v[9]*s2 + v[10]*s3 + v[11]*s4;
      t[idx+3] -= v[12]*s1 + v[13]*s2 + v[14]*s3 + v[15]*s4;
      v        -= 16;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ir      = 4*r[i];
    x[ir]   = t[ii];
    x[ir+1] = t[ii+1];
    x[ir+2] = t[ii+2];
    x[ir+3] = t[ii+3];
    ii     += 4;
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*16*(a->nz) - 4.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_4(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  IS                iscol=a->col,isrow=a->row;
  const PetscInt    n    =a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          nz,idx,idt,j,i,oidx,ii,ic,ir;
  const PetscInt    bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,x1,x2,x3,x4,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    ii    = bs*i; ic = bs*c[i];
    t[ii] = b[ic]; t[ii+1] = b[ic+1]; t[ii+2] = b[ic+2]; t[ii+3] = b[ic+3];
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx];  x4 = t[3+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4;
    s2 = v[4]*x1  +  v[5]*x2 +  v[6]*x3 +  v[7]*x4;
    s3 = v[8]*x1  +  v[9]*x2 + v[10]*x3 + v[11]*x4;
    s4 = v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4;
      t[oidx+1] -= v[4]*s1  +  v[5]*s2 +  v[6]*s3 +  v[7]*s4;
      t[oidx+2] -= v[8]*s1 + v[9]*s2 + v[10]*s3 + v[11]*s4;
      t[oidx+3] -= v[12]*s1 + v[13]*s2 + v[14]*s3 + v[15]*s4;
      v         -= bs2;
    }
    t[idx] = s1;t[1+idx] = s2;  t[2+idx] = s3;  t[3+idx] = s4;
    idx   += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = t[idt];  s2 = t[1+idt];  s3 = t[2+idt];  s4 = t[3+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      t[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3  +  v[3]*s4;
      t[idx+1] -=  v[4]*s1 +  v[5]*s2 +  v[6]*s3  +  v[7]*s4;
      t[idx+2] -=  v[8]*s1 +  v[9]*s2 +  v[10]*s3 + v[11]*s4;
      t[idx+3] -= v[12]*s1 +  v[13]*s2 + v[14]*s3 + v[15]*s4;
      v        += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    ii    = bs*i;  ir = bs*r[i];
    x[ir] = t[ii];  x[ir+1] = t[ii+1]; x[ir+2] = t[ii+2];  x[ir+3] = t[ii+3];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_5_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,ii,ic,ir,oidx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,s5,x1,x2,x3,x4,x5,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ic      = 5*c[i];
    t[ii]   = b[ic];
    t[ii+1] = b[ic+1];
    t[ii+2] = b[ic+2];
    t[ii+3] = b[ic+3];
    t[ii+4] = b[ic+4];
    ii     += 5;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 25*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx]; x4 = t[3+idx]; x5 = t[4+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5;
    s2 = v[5]*x1  +  v[6]*x2 +  v[7]*x3 +  v[8]*x4 +  v[9]*x5;
    s3 = v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4 + v[14]*x5;
    s4 = v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4 + v[19]*x5;
    s5 = v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4 + v[24]*x5;
    v += 25;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 5*(*vi++);
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5;
      t[oidx+1] -= v[5]*s1  +  v[6]*s2 +  v[7]*s3 +  v[8]*s4 +  v[9]*s5;
      t[oidx+2] -= v[10]*s1 + v[11]*s2 + v[12]*s3 + v[13]*s4 + v[14]*s5;
      t[oidx+3] -= v[15]*s1 + v[16]*s2 + v[17]*s3 + v[18]*s4 + v[19]*s5;
      t[oidx+4] -= v[20]*s1 + v[21]*s2 + v[22]*s3 + v[23]*s4 + v[24]*s5;
      v         += 25;
    }
    t[idx] = s1;t[1+idx] = s2; t[2+idx] = s3;t[3+idx] = s4; t[4+idx] = s5;
    idx   += 5;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 25*diag[i] - 25;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 5*i;
    s1  = t[idt];  s2 = t[1+idt]; s3 = t[2+idt];s4 = t[3+idt]; s5 = t[4+idt];
    while (nz--) {
      idx       = 5*(*vi--);
      t[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5;
      t[idx+1] -=  v[5]*s1 +  v[6]*s2 +  v[7]*s3 +  v[8]*s4 +  v[9]*s5;
      t[idx+2] -= v[10]*s1 + v[11]*s2 + v[12]*s3 + v[13]*s4 + v[14]*s5;
      t[idx+3] -= v[15]*s1 + v[16]*s2 + v[17]*s3 + v[18]*s4 + v[19]*s5;
      t[idx+4] -= v[20]*s1 + v[21]*s2 + v[22]*s3 + v[23]*s4 + v[24]*s5;
      v        -= 25;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ir      = 5*r[i];
    x[ir]   = t[ii];
    x[ir+1] = t[ii+1];
    x[ir+2] = t[ii+2];
    x[ir+3] = t[ii+3];
    x[ir+4] = t[ii+4];
    ii     += 5;
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*25*(a->nz) - 5.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_5(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  IS                iscol=a->col,isrow=a->row;
  const PetscInt    n    =a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          nz,idx,idt,j,i,oidx,ii,ic,ir;
  const PetscInt    bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,s5,x1,x2,x3,x4,x5,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    ii      = bs*i; ic = bs*c[i];
    t[ii]   = b[ic]; t[ii+1] = b[ic+1]; t[ii+2] = b[ic+2]; t[ii+3] = b[ic+3];
    t[ii+4] = b[ic+4];
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx]; x4 = t[3+idx]; x5 = t[4+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5;
    s2 = v[5]*x1  +  v[6]*x2 +  v[7]*x3 +  v[8]*x4 +  v[9]*x5;
    s3 = v[10]*x1 + v[11]*x2 + v[12]*x3 + v[13]*x4 + v[14]*x5;
    s4 = v[15]*x1 + v[16]*x2 + v[17]*x3 + v[18]*x4 + v[19]*x5;
    s5 = v[20]*x1 + v[21]*x2 + v[22]*x3 + v[23]*x4 + v[24]*x5;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5;
      t[oidx+1] -= v[5]*s1  +  v[6]*s2 +  v[7]*s3 +  v[8]*s4 +  v[9]*s5;
      t[oidx+2] -= v[10]*s1 + v[11]*s2 + v[12]*s3 + v[13]*s4 + v[14]*s5;
      t[oidx+3] -= v[15]*s1 + v[16]*s2 + v[17]*s3 + v[18]*s4 + v[19]*s5;
      t[oidx+4] -= v[20]*s1 + v[21]*s2 + v[22]*s3 + v[23]*s4 + v[24]*s5;
      v         -= bs2;
    }
    t[idx] = s1;t[1+idx] = s2;  t[2+idx] = s3;  t[3+idx] = s4; t[4+idx] =s5;
    idx   += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = t[idt];  s2 = t[1+idt];  s3 = t[2+idt];  s4 = t[3+idt]; s5 = t[4+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      t[idx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5;
      t[idx+1] -= v[5]*s1  +  v[6]*s2 +  v[7]*s3 +  v[8]*s4 +  v[9]*s5;
      t[idx+2] -= v[10]*s1 + v[11]*s2 + v[12]*s3 + v[13]*s4 + v[14]*s5;
      t[idx+3] -= v[15]*s1 + v[16]*s2 + v[17]*s3 + v[18]*s4 + v[19]*s5;
      t[idx+4] -= v[20]*s1 + v[21]*s2 + v[22]*s3 + v[23]*s4 + v[24]*s5;
      v        += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    ii      = bs*i;  ir = bs*r[i];
    x[ir]   = t[ii];  x[ir+1] = t[ii+1]; x[ir+2] = t[ii+2];  x[ir+3] = t[ii+3];
    x[ir+4] = t[ii+4];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_6_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,ii,ic,ir,oidx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,s5,s6,x1,x2,x3,x4,x5,x6,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ic      = 6*c[i];
    t[ii]   = b[ic];
    t[ii+1] = b[ic+1];
    t[ii+2] = b[ic+2];
    t[ii+3] = b[ic+3];
    t[ii+4] = b[ic+4];
    t[ii+5] = b[ic+5];
    ii     += 6;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 36*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx]; x4 = t[3+idx]; x5 = t[4+idx];
    x6 = t[5+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5 +  v[5]*x6;
    s2 = v[6]*x1  +  v[7]*x2 +  v[8]*x3 +  v[9]*x4 + v[10]*x5 + v[11]*x6;
    s3 = v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4 + v[16]*x5 + v[17]*x6;
    s4 = v[18]*x1 + v[19]*x2 + v[20]*x3 + v[21]*x4 + v[22]*x5 + v[23]*x6;
    s5 = v[24]*x1 + v[25]*x2 + v[26]*x3 + v[27]*x4 + v[28]*x5 + v[29]*x6;
    s6 = v[30]*x1 + v[31]*x2 + v[32]*x3 + v[33]*x4 + v[34]*x5 + v[35]*x6;
    v += 36;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 6*(*vi++);
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6;
      t[oidx+1] -= v[6]*s1  +  v[7]*s2 +  v[8]*s3 +  v[9]*s4 + v[10]*s5 + v[11]*s6;
      t[oidx+2] -= v[12]*s1 + v[13]*s2 + v[14]*s3 + v[15]*s4 + v[16]*s5 + v[17]*s6;
      t[oidx+3] -= v[18]*s1 + v[19]*s2 + v[20]*s3 + v[21]*s4 + v[22]*s5 + v[23]*s6;
      t[oidx+4] -= v[24]*s1 + v[25]*s2 + v[26]*s3 + v[27]*s4 + v[28]*s5 + v[29]*s6;
      t[oidx+5] -= v[30]*s1 + v[31]*s2 + v[32]*s3 + v[33]*s4 + v[34]*s5 + v[35]*s6;
      v         += 36;
    }
    t[idx]   = s1;t[1+idx] = s2; t[2+idx] = s3;t[3+idx] = s4; t[4+idx] = s5;
    t[5+idx] = s6;
    idx     += 6;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 36*diag[i] - 36;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 6*i;
    s1  = t[idt];  s2 = t[1+idt]; s3 = t[2+idt];s4 = t[3+idt]; s5 = t[4+idt];
    s6  = t[5+idt];
    while (nz--) {
      idx       = 6*(*vi--);
      t[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6;
      t[idx+1] -=  v[6]*s1 +  v[7]*s2 +  v[8]*s3 +  v[9]*s4 + v[10]*s5 + v[11]*s6;
      t[idx+2] -= v[12]*s1 + v[13]*s2 + v[14]*s3 + v[15]*s4 + v[16]*s5 + v[17]*s6;
      t[idx+3] -= v[18]*s1 + v[19]*s2 + v[20]*s3 + v[21]*s4 + v[22]*s5 + v[23]*s6;
      t[idx+4] -= v[24]*s1 + v[25]*s2 + v[26]*s3 + v[27]*s4 + v[28]*s5 + v[29]*s6;
      t[idx+5] -= v[30]*s1 + v[31]*s2 + v[32]*s3 + v[33]*s4 + v[34]*s5 + v[35]*s6;
      v        -= 36;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ir      = 6*r[i];
    x[ir]   = t[ii];
    x[ir+1] = t[ii+1];
    x[ir+2] = t[ii+2];
    x[ir+3] = t[ii+3];
    x[ir+4] = t[ii+4];
    x[ir+5] = t[ii+5];
    ii     += 6;
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*36*(a->nz) - 6.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_6(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  IS                iscol=a->col,isrow=a->row;
  const PetscInt    n    =a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          nz,idx,idt,j,i,oidx,ii,ic,ir;
  const PetscInt    bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,s5,s6,x1,x2,x3,x4,x5,x6,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    ii      = bs*i; ic = bs*c[i];
    t[ii]   = b[ic]; t[ii+1] = b[ic+1]; t[ii+2] = b[ic+2]; t[ii+3] = b[ic+3];
    t[ii+4] = b[ic+4];  t[ii+5] = b[ic+5];
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx]; x4 = t[3+idx]; x5 = t[4+idx];
    x6 = t[5+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5 +  v[5]*x6;
    s2 = v[6]*x1  +  v[7]*x2 +  v[8]*x3 +  v[9]*x4 + v[10]*x5 + v[11]*x6;
    s3 = v[12]*x1 + v[13]*x2 + v[14]*x3 + v[15]*x4 + v[16]*x5 + v[17]*x6;
    s4 = v[18]*x1 + v[19]*x2 + v[20]*x3 + v[21]*x4 + v[22]*x5 + v[23]*x6;
    s5 = v[24]*x1 + v[25]*x2 + v[26]*x3 + v[27]*x4 + v[28]*x5 + v[29]*x6;
    s6 = v[30]*x1 + v[31]*x2 + v[32]*x3 + v[33]*x4 + v[34]*x5 + v[35]*x6;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6;
      t[oidx+1] -= v[6]*s1  +  v[7]*s2 +  v[8]*s3 +  v[9]*s4 + v[10]*s5 + v[11]*s6;
      t[oidx+2] -= v[12]*s1 + v[13]*s2 + v[14]*s3 + v[15]*s4 + v[16]*s5 + v[17]*s6;
      t[oidx+3] -= v[18]*s1 + v[19]*s2 + v[20]*s3 + v[21]*s4 + v[22]*s5 + v[23]*s6;
      t[oidx+4] -= v[24]*s1 + v[25]*s2 + v[26]*s3 + v[27]*s4 + v[28]*s5 + v[29]*s6;
      t[oidx+5] -= v[30]*s1 + v[31]*s2 + v[32]*s3 + v[33]*s4 + v[34]*s5 + v[35]*s6;
      v         -= bs2;
    }
    t[idx]   = s1;t[1+idx] = s2;  t[2+idx] = s3;  t[3+idx] = s4; t[4+idx] =s5;
    t[5+idx] = s6;
    idx     += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = t[idt];  s2 = t[1+idt];  s3 = t[2+idt];  s4 = t[3+idt]; s5 = t[4+idt];
    s6  = t[5+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      t[idx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6;
      t[idx+1] -= v[6]*s1  +  v[7]*s2 +  v[8]*s3 +  v[9]*s4 + v[10]*s5 + v[11]*s6;
      t[idx+2] -= v[12]*s1 + v[13]*s2 + v[14]*s3 + v[15]*s4 + v[16]*s5 + v[17]*s6;
      t[idx+3] -= v[18]*s1 + v[19]*s2 + v[20]*s3 + v[21]*s4 + v[22]*s5 + v[23]*s6;
      t[idx+4] -= v[24]*s1 + v[25]*s2 + v[26]*s3 + v[27]*s4 + v[28]*s5 + v[29]*s6;
      t[idx+5] -= v[30]*s1 + v[31]*s2 + v[32]*s3 + v[33]*s4 + v[34]*s5 + v[35]*s6;
      v        += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    ii      = bs*i;  ir = bs*r[i];
    x[ir]   = t[ii];  x[ir+1] = t[ii+1]; x[ir+2] = t[ii+2];  x[ir+3] = t[ii+3];
    x[ir+4] = t[ii+4];  x[ir+5] = t[ii+5];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_7_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   =(Mat_SeqBAIJ*)A->data;
  IS                iscol=a->col,isrow=a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,ii,ic,ir,oidx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,s5,s6,s7,x1,x2,x3,x4,x5,x6,x7,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ic      = 7*c[i];
    t[ii]   = b[ic];
    t[ii+1] = b[ic+1];
    t[ii+2] = b[ic+2];
    t[ii+3] = b[ic+3];
    t[ii+4] = b[ic+4];
    t[ii+5] = b[ic+5];
    t[ii+6] = b[ic+6];
    ii     += 7;
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 49*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx]; x4 = t[3+idx]; x5 = t[4+idx];
    x6 = t[5+idx]; x7 = t[6+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5 +  v[5]*x6 +  v[6]*x7;
    s2 = v[7]*x1  +  v[8]*x2 +  v[9]*x3 + v[10]*x4 + v[11]*x5 + v[12]*x6 + v[13]*x7;
    s3 = v[14]*x1 + v[15]*x2 + v[16]*x3 + v[17]*x4 + v[18]*x5 + v[19]*x6 + v[20]*x7;
    s4 = v[21]*x1 + v[22]*x2 + v[23]*x3 + v[24]*x4 + v[25]*x5 + v[26]*x6 + v[27]*x7;
    s5 = v[28]*x1 + v[29]*x2 + v[30]*x3 + v[31]*x4 + v[32]*x5 + v[33]*x6 + v[34]*x7;
    s6 = v[35]*x1 + v[36]*x2 + v[37]*x3 + v[38]*x4 + v[39]*x5 + v[40]*x6 + v[41]*x7;
    s7 = v[42]*x1 + v[43]*x2 + v[44]*x3 + v[45]*x4 + v[46]*x5 + v[47]*x6 + v[48]*x7;
    v += 49;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 7*(*vi++);
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6 +  v[6]*s7;
      t[oidx+1] -= v[7]*s1  +  v[8]*s2 +  v[9]*s3 + v[10]*s4 + v[11]*s5 + v[12]*s6 + v[13]*s7;
      t[oidx+2] -= v[14]*s1 + v[15]*s2 + v[16]*s3 + v[17]*s4 + v[18]*s5 + v[19]*s6 + v[20]*s7;
      t[oidx+3] -= v[21]*s1 + v[22]*s2 + v[23]*s3 + v[24]*s4 + v[25]*s5 + v[26]*s6 + v[27]*s7;
      t[oidx+4] -= v[28]*s1 + v[29]*s2 + v[30]*s3 + v[31]*s4 + v[32]*s5 + v[33]*s6 + v[34]*s7;
      t[oidx+5] -= v[35]*s1 + v[36]*s2 + v[37]*s3 + v[38]*s4 + v[39]*s5 + v[40]*s6 + v[41]*s7;
      t[oidx+6] -= v[42]*s1 + v[43]*s2 + v[44]*s3 + v[45]*s4 + v[46]*s5 + v[47]*s6 + v[48]*s7;
      v         += 49;
    }
    t[idx]   = s1;t[1+idx] = s2; t[2+idx] = s3;t[3+idx] = s4; t[4+idx] = s5;
    t[5+idx] = s6;t[6+idx] = s7;
    idx     += 7;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 49*diag[i] - 49;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 7*i;
    s1  = t[idt];  s2 = t[1+idt]; s3 = t[2+idt];s4 = t[3+idt]; s5 = t[4+idt];
    s6  = t[5+idt];s7 = t[6+idt];
    while (nz--) {
      idx       = 7*(*vi--);
      t[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6 +  v[6]*s7;
      t[idx+1] -=  v[7]*s1 +  v[8]*s2 +  v[9]*s3 + v[10]*s4 + v[11]*s5 + v[12]*s6 + v[13]*s7;
      t[idx+2] -= v[14]*s1 + v[15]*s2 + v[16]*s3 + v[17]*s4 + v[18]*s5 + v[19]*s6 + v[20]*s7;
      t[idx+3] -= v[21]*s1 + v[22]*s2 + v[23]*s3 + v[24]*s4 + v[25]*s5 + v[26]*s6 + v[27]*s7;
      t[idx+4] -= v[28]*s1 + v[29]*s2 + v[30]*s3 + v[31]*s4 + v[32]*s5 + v[33]*s6 + v[34]*s7;
      t[idx+5] -= v[35]*s1 + v[36]*s2 + v[37]*s3 + v[38]*s4 + v[39]*s5 + v[40]*s6 + v[41]*s7;
      t[idx+6] -= v[42]*s1 + v[43]*s2 + v[44]*s3 + v[45]*s4 + v[46]*s5 + v[47]*s6 + v[48]*s7;
      v        -= 49;
    }
  }

  /* copy t into x according to permutation */
  ii = 0;
  for (i=0; i<n; i++) {
    ir      = 7*r[i];
    x[ir]   = t[ii];
    x[ir+1] = t[ii+1];
    x[ir+2] = t[ii+2];
    x[ir+3] = t[ii+3];
    x[ir+4] = t[ii+4];
    x[ir+5] = t[ii+5];
    x[ir+6] = t[ii+6];
    ii     += 7;
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*49*(a->nz) - 7.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode MatSolveTranspose_SeqBAIJ_7(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a=(Mat_SeqBAIJ*)A->data;
  PetscErrorCode    ierr;
  IS                iscol=a->col,isrow=a->row;
  const PetscInt    n    =a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          nz,idx,idt,j,i,oidx,ii,ic,ir;
  const PetscInt    bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       s1,s2,s3,s4,s5,s6,s7,x1,x2,x3,x4,x5,x6,x7,*x,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  t    = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy b into temp work space according to permutation */
  for (i=0; i<n; i++) {
    ii      = bs*i; ic = bs*c[i];
    t[ii]   = b[ic]; t[ii+1] = b[ic+1]; t[ii+2] = b[ic+2]; t[ii+3] = b[ic+3];
    t[ii+4] = b[ic+4];  t[ii+5] = b[ic+5];  t[ii+6] = b[ic+6];
  }

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = t[idx];   x2 = t[1+idx]; x3    = t[2+idx]; x4 = t[3+idx]; x5 = t[4+idx];
    x6 = t[5+idx]; x7 = t[6+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3 +  v[3]*x4 +  v[4]*x5 +  v[5]*x6 +  v[6]*x7;
    s2 = v[7]*x1  +  v[8]*x2 +  v[9]*x3 + v[10]*x4 + v[11]*x5 + v[12]*x6 + v[13]*x7;
    s3 = v[14]*x1 + v[15]*x2 + v[16]*x3 + v[17]*x4 + v[18]*x5 + v[19]*x6 + v[20]*x7;
    s4 = v[21]*x1 + v[22]*x2 + v[23]*x3 + v[24]*x4 + v[25]*x5 + v[26]*x6 + v[27]*x7;
    s5 = v[28]*x1 + v[29]*x2 + v[30]*x3 + v[31]*x4 + v[32]*x5 + v[33]*x6 + v[34]*x7;
    s6 = v[35]*x1 + v[36]*x2 + v[37]*x3 + v[38]*x4 + v[39]*x5 + v[40]*x6 + v[41]*x7;
    s7 = v[42]*x1 + v[43]*x2 + v[44]*x3 + v[45]*x4 + v[46]*x5 + v[47]*x6 + v[48]*x7;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      t[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6 +  v[6]*s7;
      t[oidx+1] -= v[7]*s1  +  v[8]*s2 +  v[9]*s3 + v[10]*s4 + v[11]*s5 + v[12]*s6 + v[13]*s7;
      t[oidx+2] -= v[14]*s1 + v[15]*s2 + v[16]*s3 + v[17]*s4 + v[18]*s5 + v[19]*s6 + v[20]*s7;
      t[oidx+3] -= v[21]*s1 + v[22]*s2 + v[23]*s3 + v[24]*s4 + v[25]*s5 + v[26]*s6 + v[27]*s7;
      t[oidx+4] -= v[28]*s1 + v[29]*s2 + v[30]*s3 + v[31]*s4 + v[32]*s5 + v[33]*s6 + v[34]*s7;
      t[oidx+5] -= v[35]*s1 + v[36]*s2 + v[37]*s3 + v[38]*s4 + v[39]*s5 + v[40]*s6 + v[41]*s7;
      t[oidx+6] -= v[42]*s1 + v[43]*s2 + v[44]*s3 + v[45]*s4 + v[46]*s5 + v[47]*s6 + v[48]*s7;
      v         -= bs2;
    }
    t[idx]   = s1;t[1+idx] = s2;  t[2+idx] = s3;  t[3+idx] = s4; t[4+idx] =s5;
    t[5+idx] = s6;  t[6+idx] = s7;
    idx     += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = t[idt];  s2 = t[1+idt];  s3 = t[2+idt];  s4 = t[3+idt]; s5 = t[4+idt];
    s6  = t[5+idt];  s7 = t[6+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      t[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5 +  v[5]*s6 +  v[6]*s7;
      t[idx+1] -=  v[7]*s1 +  v[8]*s2 +  v[9]*s3 + v[10]*s4 + v[11]*s5 + v[12]*s6 + v[13]*s7;
      t[idx+2] -= v[14]*s1 + v[15]*s2 + v[16]*s3 + v[17]*s4 + v[18]*s5 + v[19]*s6 + v[20]*s7;
      t[idx+3] -= v[21]*s1 + v[22]*s2 + v[23]*s3 + v[24]*s4 + v[25]*s5 + v[26]*s6 + v[27]*s7;
      t[idx+4] -= v[28]*s1 + v[29]*s2 + v[30]*s3 + v[31]*s4 + v[32]*s5 + v[33]*s6 + v[34]*s7;
      t[idx+5] -= v[35]*s1 + v[36]*s2 + v[37]*s3 + v[38]*s4 + v[39]*s5 + v[40]*s6 + v[41]*s7;
      t[idx+6] -= v[42]*s1 + v[43]*s2 + v[44]*s3 + v[45]*s4 + v[46]*s5 + v[47]*s6 + v[48]*s7;
      v        += bs2;
    }
  }

  /* copy t into x according to permutation */
  for (i=0; i<n; i++) {
    ii      = bs*i;  ir = bs*r[i];
    x[ir]   = t[ii];  x[ir+1] = t[ii+1]; x[ir+2] = t[ii+2];  x[ir+3] = t[ii+3];
    x[ir+4] = t[ii+4];  x[ir+5] = t[ii+5];  x[ir+6] = t[ii+6];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
    ierr = PetscMemcpy(ls,t+i*bs,bs*sizeof(PetscScalar));CHKERRQ(ierr);
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
    ierr = PetscMemcpy(ls,t+i*bs,bs*sizeof(PetscScalar));CHKERRQ(ierr);
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

