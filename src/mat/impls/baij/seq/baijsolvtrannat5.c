#include <../src/mat/impls/baij/seq/baij.h>

PetscErrorCode MatSolveTranspose_SeqBAIJ_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ*)A->data;
  const PetscInt  *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt        i,nz,idx,idt,oidx;
  const MatScalar *aa=a->a,*v;
  PetscScalar     s1,s2,s3,s4,s5,x1,x2,x3,x4,x5,*x;

  PetscFunctionBegin;
  PetscCall(VecCopy(bb,xx));
  PetscCall(VecGetArray(xx,&x));

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {

    v = aa + 25*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = x[idx];   x2 = x[1+idx]; x3    = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
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
      x[oidx]   -= v[0]*s1  +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5;
      x[oidx+1] -= v[5]*s1  +  v[6]*s2 +  v[7]*s3 +  v[8]*s4 +  v[9]*s5;
      x[oidx+2] -= v[10]*s1 + v[11]*s2 + v[12]*s3 + v[13]*s4 + v[14]*s5;
      x[oidx+3] -= v[15]*s1 + v[16]*s2 + v[17]*s3 + v[18]*s4 + v[19]*s5;
      x[oidx+4] -= v[20]*s1 + v[21]*s2 + v[22]*s3 + v[23]*s4 + v[24]*s5;
      v         += 25;
    }
    x[idx] = s1;x[1+idx] = s2; x[2+idx] = s3;x[3+idx] = s4; x[4+idx] = s5;
    idx   += 5;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + 25*diag[i] - 25;
    vi  = aj + diag[i] - 1;
    nz  = diag[i] - ai[i];
    idt = 5*i;
    s1  = x[idt];  s2 = x[1+idt]; s3 = x[2+idt];s4 = x[3+idt]; s5 = x[4+idt];
    while (nz--) {
      idx       = 5*(*vi--);
      x[idx]   -=  v[0]*s1 +  v[1]*s2 +  v[2]*s3 +  v[3]*s4 +  v[4]*s5;
      x[idx+1] -=  v[5]*s1 +  v[6]*s2 +  v[7]*s3 +  v[8]*s4 +  v[9]*s5;
      x[idx+2] -= v[10]*s1 + v[11]*s2 + v[12]*s3 + v[13]*s4 + v[14]*s5;
      x[idx+3] -= v[15]*s1 + v[16]*s2 + v[17]*s3 + v[18]*s4 + v[19]*s5;
      x[idx+4] -= v[20]*s1 + v[21]*s2 + v[22]*s3 + v[23]*s4 + v[24]*s5;
      v        -= 25;
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*25*(a->nz) - 5.0*A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SeqBAIJ_5_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ*)A->data;
  const PetscInt  n=a->mbs,*vi,*ai=a->i,*aj=a->j,*diag=a->diag;
  PetscInt        nz,idx,idt,j,i,oidx;
  const PetscInt  bs =A->rmap->bs,bs2=a->bs2;
  const MatScalar *aa=a->a,*v;
  PetscScalar     s1,s2,s3,s4,s5,x1,x2,x3,x4,x5,*x;

  PetscFunctionBegin;
  PetscCall(VecCopy(bb,xx));
  PetscCall(VecGetArray(xx,&x));

  /* forward solve the U^T */
  idx = 0;
  for (i=0; i<n; i++) {
    v = aa + bs2*diag[i];
    /* multiply by the inverse of the block diagonal */
    x1 = x[idx];   x2 = x[1+idx];  x3 = x[2+idx];  x4 = x[3+idx];
    x5 = x[4+idx];
    s1 =  v[0]*x1   +  v[1]*x2   + v[2]*x3   + v[3]*x4   + v[4]*x5;
    s2 =  v[5]*x1   +  v[6]*x2   + v[7]*x3   + v[8]*x4   + v[9]*x5;
    s3 =  v[10]*x1  +  v[11]*x2  + v[12]*x3  + v[13]*x4  + v[14]*x5;
    s4 =  v[15]*x1  +  v[16]*x2  + v[17]*x3  + v[18]*x4  + v[19]*x5;
    s5 =  v[20]*x1  +  v[21]*x2  + v[22]*x3  + v[23]*x4   + v[24]*x5;
    v -= bs2;

    vi = aj + diag[i] - 1;
    nz = diag[i] - diag[i+1] - 1;
    for (j=0; j>-nz; j--) {
      oidx       = bs*vi[j];
      x[oidx]   -=  v[0]*s1   +  v[1]*s2   + v[2]*s3   + v[3]*s4   + v[4]*s5;
      x[oidx+1] -=  v[5]*s1   +  v[6]*s2   + v[7]*s3   + v[8]*s4   + v[9]*s5;
      x[oidx+2] -=  v[10]*s1  +  v[11]*s2  + v[12]*s3  + v[13]*s4  + v[14]*s5;
      x[oidx+3] -=  v[15]*s1  +  v[16]*s2  + v[17]*s3  + v[18]*s4  + v[19]*s5;
      x[oidx+4] -=  v[20]*s1  +  v[21]*s2  + v[22]*s3  + v[23]*s4   + v[24]*s5;
      v         -= bs2;
    }
    x[idx] = s1;x[1+idx] = s2;  x[2+idx] = s3;  x[3+idx] = s4; x[4+idx] = s5;
    idx   += bs;
  }
  /* backward solve the L^T */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];  s3 = x[2+idt];  s4 = x[3+idt];  s5 = x[4+idt];
    for (j=0; j<nz; j++) {
      idx       = bs*vi[j];
      x[idx]   -=  v[0]*s1   +  v[1]*s2   + v[2]*s3   + v[3]*s4   + v[4]*s5;
      x[idx+1] -=  v[5]*s1   +  v[6]*s2   + v[7]*s3   + v[8]*s4   + v[9]*s5;
      x[idx+2] -=  v[10]*s1  +  v[11]*s2  + v[12]*s3  + v[13]*s4  + v[14]*s5;
      x[idx+3] -=  v[15]*s1  +  v[16]*s2  + v[17]*s3  + v[18]*s4  + v[19]*s5;
      x[idx+4] -=  v[20]*s1  +  v[21]*s2  + v[22]*s3  + v[23]*s4   + v[24]*s5;
      v        += bs2;
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n));
  PetscFunctionReturn(0);
}
