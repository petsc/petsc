#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   = (Mat_SeqBAIJ*)A->data;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscInt          i,nz,idx,idt,jdx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,x1,x2,x3,x4,x5;
  const PetscScalar *b;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(bb,&b));
  CHKERRQ(VecGetArray(xx,&x));
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx]; x[2] = b[2+idx]; x[3] = b[3+idx];x[4] = b[4+idx];
  for (i=1; i<n; i++) {
    v   =  aa + 25*ai[i];
    vi  =  aj + ai[i];
    nz  =  diag[i] - ai[i];
    idx =  5*i;
    s1  =  b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];s5 = b[4+idx];
    while (nz--) {
      jdx = 5*(*vi++);
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];x5 = x[4+jdx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 25*diag[i] + 25;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 5*i;
    s1  = x[idt];  s2 = x[1+idt];
    s3  = x[2+idt];s4 = x[3+idt]; s5 = x[4+idt];
    while (nz--) {
      idx = 5*(*vi++);
      x1  = x[idx];   x2 = x[1+idx];x3    = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    v        = aa + 25*diag[i];
    x[idt]   = v[0]*s1 + v[5]*s2 + v[10]*s3  + v[15]*s4 + v[20]*s5;
    x[1+idt] = v[1]*s1 + v[6]*s2 + v[11]*s3  + v[16]*s4 + v[21]*s5;
    x[2+idt] = v[2]*s1 + v[7]*s2 + v[12]*s3  + v[17]*s4 + v[22]*s5;
    x[3+idt] = v[3]*s1 + v[8]*s2 + v[13]*s3  + v[18]*s4 + v[23]*s5;
    x[4+idt] = v[4]*s1 + v[9]*s2 + v[14]*s3  + v[19]*s4 + v[24]*s5;
  }

  CHKERRQ(VecRestoreArrayRead(bb,&b));
  CHKERRQ(VecRestoreArray(xx,&x));
  CHKERRQ(PetscLogFlops(2.0*25*(a->nz) - 5.0*A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  = a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscInt          i,k,nz,idx,idt,jdx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,x1,x2,x3,x4,x5;
  const PetscScalar *b;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(bb,&b));
  CHKERRQ(VecGetArray(xx,&x));
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx]; x[2] = b[2+idx]; x[3] = b[3+idx];x[4] = b[4+idx];
  for (i=1; i<n; i++) {
    v   = aa + 25*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = 5*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];s5 = b[4+idx];
    for (k=0; k<nz; k++) {
      jdx = 5*vi[k];
      x1  = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];x5 = x[4+jdx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 25*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = 5*i;
    s1  = x[idt];  s2 = x[1+idt];
    s3  = x[2+idt];s4 = x[3+idt]; s5 = x[4+idt];
    for (k=0; k<nz; k++) {
      idx = 5*vi[k];
      x1  = x[idx];   x2 = x[1+idx];x3    = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
      s1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      s2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      s3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      s4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      s5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v  += 25;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[5]*s2 + v[10]*s3  + v[15]*s4 + v[20]*s5;
    x[1+idt] = v[1]*s1 + v[6]*s2 + v[11]*s3  + v[16]*s4 + v[21]*s5;
    x[2+idt] = v[2]*s1 + v[7]*s2 + v[12]*s3  + v[17]*s4 + v[22]*s5;
    x[3+idt] = v[3]*s1 + v[8]*s2 + v[13]*s3  + v[18]*s4 + v[23]*s5;
    x[4+idt] = v[4]*s1 + v[9]*s2 + v[14]*s3  + v[19]*s4 + v[24]*s5;
  }

  CHKERRQ(VecRestoreArrayRead(bb,&b));
  CHKERRQ(VecRestoreArray(xx,&x));
  CHKERRQ(PetscLogFlops(2.0*25*(a->nz) - 5.0*A->cmap->n));
  PetscFunctionReturn(0);
}
