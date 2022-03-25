#include <../src/mat/impls/baij/seq/baij.h>

PetscErrorCode MatSolveTranspose_SeqBAIJ_3_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ*)A->data;
  const PetscInt  n=a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  PetscInt        i,nz,idx,idt,oidx;
  const MatScalar *aa=a->a,*v;
  PetscScalar     s1,s2,s3,x1,x2,x3,*x;

  PetscFunctionBegin;
  PetscCall(VecCopy(bb,xx));
  PetscCall(VecGetArray(xx,&x));

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 9*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = x[idx];   x2 = x[1+idx]; x3    = x[2+idx];
    s1 = v[0]*x1  +  v[1]*x2 +  v[2]*x3;
    s2 = v[3]*x1  +  v[4]*x2 +  v[5]*x3;
    s3 = v[6]*x1  +  v[7]*x2 + v[8]*x3;
    v += 9;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 3*(*vi++);
      x[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3;
      x[oidx+1] -= v[3]*s1  +  v[4]*s2 +  v[5]*s3;
      x[oidx+2] -= v[6]*s1 + v[7]*s2 + v[8]*s3;
      v         += 9;
    }
    x[idx] = s1;x[1+idx] = s2; x[2+idx] = s3;
    idx   += 3;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 9*diag[i] - 9;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 3*i;
    s1  = x[idt];  s2 = x[1+idt]; s3 = x[2+idt];
    while (nz--) {
      idx       = 3*(*vi--);
      x[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3;
      x[idx+1] -=  v[3]*s1 +  v[4]*s2 +  v[5]*s3;
      x[idx+2] -= v[6]*s1 + v[7]*s2 + v[8]*s3;
      v        -= 9;
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*9.0*(a->nz) - 3.0*A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_3_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ*)A->data;
  const PetscInt  n=a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  PetscInt        nz,idx,idt,j,i,oidx;
  const PetscInt  bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar *aa=a->a,*v;
  PetscScalar     s1,s2,s3,x1,x2,x3,*x;

  PetscFunctionBegin;
  PetscCall(VecCopy(bb,xx));
  PetscCall(VecGetArray(xx,&x));

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = x[idx];   x2 = x[1+idx];  x3 = x[2+idx];
    s1 = v[0]*x1  +  v[1]*x2  + v[2]*x3;
    s2 = v[3]*x1  +  v[4]*x2  + v[5]*x3;
    s3 = v[6]*x1  +  v[7]*x2  + v[8]*x3;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      x[oidx]   -= v[0]*s1  +  v[1]*s2  + v[2]*s3;
      x[oidx+1] -= v[3]*s1  +  v[4]*s2  + v[5]*s3;
      x[oidx+2] -= v[6]*s1  +  v[7]*s2  + v[8]*s3;
      v         -= bs2;
    }
    x[idx] = s1;x[1+idx] = s2;  x[2+idx] = s3;
    idx   += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];  s3 = x[2+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      x[idx]   -= v[0]*s1  +  v[1]*s2  + v[2]*s3;
      x[idx+1] -= v[3]*s1  +  v[4]*s2  + v[5]*s3;
      x[idx+2] -= v[6]*s1  +  v[7]*s2  + v[8]*s3;
      v        += bs2;
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n));
  PetscFunctionReturn(0);
}
