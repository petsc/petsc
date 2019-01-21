#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

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
