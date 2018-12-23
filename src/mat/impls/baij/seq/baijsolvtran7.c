#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

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
