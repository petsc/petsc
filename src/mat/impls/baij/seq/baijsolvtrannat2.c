#include <../src/mat/impls/baij/seq/baij.h>

PetscErrorCode MatSolveTranspose_SeqBAIJ_2_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ*)A->data;
  PetscInt        i,nz,idx,idt,oidx;
  const PetscInt  *diag = a->diag,*vi,n=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar *aa   =a->a,*v;
  PetscScalar     s1,s2,x1,x2,*x;

  PetscFunctionBegin;
  PetscCall(VecCopy(bb,xx));
  PetscCall(VecGetArray(xx,&x));

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 4*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = x[idx];   x2 = x[1+idx];
    s1 = v[0]*x1  +  v[1]*x2;
    s2 = v[2]*x1  +  v[3]*x2;
    v += 4;

    vi = aj + diag[i] + 1;
    nz = ai[i+1] - diag[i] - 1;
    while (nz--) {
      oidx       = 2*(*vi++);
      x[oidx]   -= v[0]*s1  +  v[1]*s2;
      x[oidx+1] -= v[2]*s1  +  v[3]*s2;
      v         += 4;
    }
    x[idx] = s1;x[1+idx] = s2;
    idx   += 2;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 4*diag[i] - 4;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 2*i;
    s1  = x[idt];  s2 = x[1+idt];
    while (nz--) {
      idx       = 2*(*vi--);
      x[idx]   -=  v[0]*s1 +  v[1]*s2;
      x[idx+1] -=  v[2]*s1 +  v[3]*s2;
      v        -= 4;
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*4.0*(a->nz) - 2.0*A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_2_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ*)A->data;
  const PetscInt  n=a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  PetscInt        nz,idx,idt,j,i,oidx;
  const PetscInt  bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar *aa=a->a,*v;
  PetscScalar     s1,s2,x1,x2,*x;

  PetscFunctionBegin;
  PetscCall(VecCopy(bb,xx));
  PetscCall(VecGetArray(xx,&x));

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = x[idx];   x2 = x[1+idx];
    s1 = v[0]*x1  +  v[1]*x2;
    s2 = v[2]*x1  +  v[3]*x2;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      x[oidx]   -= v[0]*s1  +  v[1]*s2;
      x[oidx+1] -= v[2]*s1  +  v[3]*s2;
      v         -= bs2;
    }
    x[idx] = s1;x[1+idx] = s2;
    idx   += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      x[idx]   -=  v[0]*s1 +  v[1]*s2;
      x[idx+1] -=  v[2]*s1 +  v[3]*s2;
      v        += bs2;
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n));
  PetscFunctionReturn(0);
}
