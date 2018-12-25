#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a   = (Mat_SeqBAIJ*)A->data;
  const PetscInt    *diag=a->diag,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  PetscErrorCode    ierr;
  PetscInt          i,nz,idx,idt,jdx;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,s1,s2,s3,s4,s5,s6,s7,x1,x2,x3,x4,x5,x6,x7;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx];   x[1] = b[1+idx]; x[2] = b[2+idx];
  x[3] = b[3+idx]; x[4] = b[4+idx]; x[5] = b[5+idx];
  x[6] = b[6+idx];
  for (i=1; i<n; i++) {
    v   =  aa + 49*ai[i];
    vi  =  aj + ai[i];
    nz  =  diag[i] - ai[i];
    idx =  7*i;
    s1  =  b[idx];   s2 = b[1+idx]; s3 = b[2+idx];
    s4  =  b[3+idx]; s5 = b[4+idx]; s6 = b[5+idx];
    s7  =  b[6+idx];
    while (nz--) {
      jdx = 7*(*vi++);
      x1  = x[jdx];   x2 = x[1+jdx]; x3 = x[2+jdx];
      x4  = x[3+jdx]; x5 = x[4+jdx]; x6 = x[5+jdx];
      x7  = x[6+jdx];
      s1 -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  += 49;
    }
    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
    x[5+idx] = s6;
    x[6+idx] = s7;
  }
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + 49*diag[i] + 49;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    idt = 7*i;
    s1  = x[idt];   s2 = x[1+idt];
    s3  = x[2+idt]; s4 = x[3+idt];
    s5  = x[4+idt]; s6 = x[5+idt];
    s7  = x[6+idt];
    while (nz--) {
      idx = 7*(*vi++);
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];
      x4  = x[3+idx]; x5 = x[4+idx]; x6 = x[5+idx];
      x7  = x[6+idx];
      s1 -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  += 49;
    }
    v      = aa + 49*diag[i];
    x[idt] = v[0]*s1 + v[7]*s2  + v[14]*s3 + v[21]*s4
             + v[28]*s5 + v[35]*s6 + v[42]*s7;
    x[1+idt] = v[1]*s1 + v[8]*s2  + v[15]*s3 + v[22]*s4
               + v[29]*s5 + v[36]*s6 + v[43]*s7;
    x[2+idt] = v[2]*s1 + v[9]*s2  + v[16]*s3 + v[23]*s4
               + v[30]*s5 + v[37]*s6 + v[44]*s7;
    x[3+idt] = v[3]*s1 + v[10]*s2  + v[17]*s3 + v[24]*s4
               + v[31]*s5 + v[38]*s6 + v[45]*s7;
    x[4+idt] = v[4]*s1 + v[11]*s2  + v[18]*s3 + v[25]*s4
               + v[32]*s5 + v[39]*s6 + v[46]*s7;
    x[5+idt] = v[5]*s1 + v[12]*s2  + v[19]*s3 + v[26]*s4
               + v[33]*s5 + v[40]*s6 + v[47]*s7;
    x[6+idt] = v[6]*s1 + v[13]*s2  + v[20]*s3 + v[27]*s4
               + v[34]*s5 + v[41]*s6 + v[48]*s7;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*36*(a->nz) - 6.0*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  const PetscInt    n  =a->mbs,*vi,*ai=a->i,*aj=a->j,*adiag=a->diag;
  PetscErrorCode    ierr;
  PetscInt          i,k,nz,idx,jdx,idt;
  const PetscInt    bs = A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscScalar       s1,s2,s3,s4,s5,s6,s7,x1,x2,x3,x4,x5,x6,x7;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* forward solve the lower triangular */
  idx  = 0;
  x[0] = b[idx]; x[1] = b[1+idx];x[2] = b[2+idx];x[3] = b[3+idx];
  x[4] = b[4+idx];x[5] = b[5+idx];x[6] = b[6+idx];
  for (i=1; i<n; i++) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    idx = bs*i;
    s1  = b[idx];s2 = b[1+idx];s3 = b[2+idx];s4 = b[3+idx];
    s5  = b[4+idx];s6 = b[5+idx];s7 = b[6+idx];
    for (k=0; k<nz; k++) {
      jdx = bs*vi[k];
      x1  = x[jdx];x2 = x[1+jdx]; x3 =x[2+jdx];x4 =x[3+jdx];
      x5  = x[4+jdx]; x6 = x[5+jdx];x7 = x[6+jdx];
      s1 -= v[0]*x1 + v[7]*x2 + v[14]*x3 + v[21]*x4  + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2 + v[15]*x3 + v[22]*x4  + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2 + v[16]*x3 + v[23]*x4  + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4  + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4  + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4  + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4  + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  +=  bs2;
    }

    x[idx]   = s1;
    x[1+idx] = s2;
    x[2+idx] = s3;
    x[3+idx] = s4;
    x[4+idx] = s5;
    x[5+idx] = s6;
    x[6+idx] = s7;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--) {
    v   = aa + bs2*(adiag[i+1]+1);
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i] - adiag[i+1]-1;
    idt = bs*i;
    s1  = x[idt];  s2 = x[1+idt];s3 = x[2+idt];s4 = x[3+idt];
    s5  = x[4+idt];s6 = x[5+idt];s7 = x[6+idt];
    for (k=0; k<nz; k++) {
      idx = bs*vi[k];
      x1  = x[idx];   x2 = x[1+idx]; x3 = x[2+idx];x4 = x[3+idx];
      x5  = x[4+idx];x6 = x[5+idx];x7 = x[6+idx];
      s1 -= v[0]*x1 + v[7]*x2 + v[14]*x3 + v[21]*x4  + v[28]*x5 + v[35]*x6 + v[42]*x7;
      s2 -= v[1]*x1 + v[8]*x2 + v[15]*x3 + v[22]*x4  + v[29]*x5 + v[36]*x6 + v[43]*x7;
      s3 -= v[2]*x1 + v[9]*x2 + v[16]*x3 + v[23]*x4  + v[30]*x5 + v[37]*x6 + v[44]*x7;
      s4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4  + v[31]*x5 + v[38]*x6 + v[45]*x7;
      s5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4  + v[32]*x5 + v[39]*x6 + v[46]*x7;
      s6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4  + v[33]*x5 + v[40]*x6 + v[47]*x7;
      s7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4  + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v  +=  bs2;
    }
    /* x = inv_diagonal*x */
    x[idt]   = v[0]*s1 + v[7]*s2 + v[14]*s3 + v[21]*s4  + v[28]*s5 + v[35]*s6 + v[42]*s7;
    x[1+idt] = v[1]*s1 + v[8]*s2 + v[15]*s3 + v[22]*s4  + v[29]*s5 + v[36]*s6 + v[43]*s7;
    x[2+idt] = v[2]*s1 + v[9]*s2 + v[16]*s3 + v[23]*s4  + v[30]*s5 + v[37]*s6 + v[44]*s7;
    x[3+idt] = v[3]*s1 + v[10]*s2 + v[17]*s3 + v[24]*s4  + v[31]*s5 + v[38]*s6 + v[45]*s7;
    x[4+idt] = v[4]*s1 + v[11]*s2 + v[18]*s3 + v[25]*s4  + v[32]*s5 + v[39]*s6 + v[46]*s7;
    x[5+idt] = v[5]*s1 + v[12]*s2 + v[19]*s3 + v[26]*s4  + v[33]*s5 + v[40]*s6 + v[47]*s7;
    x[6+idt] = v[6]*s1 + v[13]*s2 + v[20]*s3 + v[27]*s4  + v[34]*s5 + v[41]*s6 + v[48]*s7;
  }

  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*bs2*(a->nz) - bs*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

