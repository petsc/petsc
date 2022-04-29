
/*
    Factorization code for SBAIJ format.
*/

#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSolve_SeqSBAIJ_N_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   =(Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  PetscInt          mbs  =a->mbs,*ai=a->i,*aj=a->j;
  const PetscInt    *r;
  PetscInt          nz,*vj,k,idx,k1;
  PetscInt          bs =A->rmap->bs,bs2 = a->bs2;
  const MatScalar   *aa=a->a,*v,*diag;
  PetscScalar       *x,*xk,*xj,*xk_tmp,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&r));
  PetscCall(PetscMalloc1(bs,&xk_tmp));

  /* solve U^T * D * y = b by forward substitution */
  xk = t;
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx = bs*r[k];
    for (k1=0; k1<bs; k1++) *xk++ = b[idx+k1];
  }
  for (k=0; k<mbs; k++) {
    v    = aa + bs2*ai[k];
    xk   = t + k*bs;    /* Dk*xk = k-th block of x */
    PetscCall(PetscArraycpy(xk_tmp,xk,bs)); /* xk_tmp <- xk */
    nz   = ai[k+1] - ai[k];
    vj   = aj + ai[k];
    xj   = t + (*vj)*bs; /* *vj-th block of x, *vj>k */
    while (nz--) {
      /* x(:) += U(k,:)^T*(Dk*xk) */
      PetscKernel_v_gets_v_plus_Atranspose_times_w(bs,xj,v,xk_tmp); /* xj <- xj + v^t * xk */
      vj++; xj = t + (*vj)*bs;
      v       += bs2;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag = aa+k*bs2;                            /* ptr to inv(Dk) */
    PetscKernel_w_gets_A_times_v(bs,xk_tmp,diag,xk); /* xk <- diag * xk */
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + bs2*ai[k];
    xk = t + k*bs;        /* xk */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    xj = t + (*vj)*bs;
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      PetscKernel_v_gets_v_plus_A_times_w(bs,xk,v,xj); /* xk <- xk + v*xj */
      vj++;
      v += bs2; xj = t + (*vj)*bs;
    }
    idx = bs*r[k];
    for (k1=0; k1<bs; k1++) x[idx+k1] = *xk++;
  }

  PetscCall(PetscFree(xk_tmp));
  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*bs2*a->nz -(bs+2.0*bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_N_inplace(Mat A,Vec bb,Vec xx)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"not implemented yet");
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_N_inplace(Mat A,Vec bb,Vec xx)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_N_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscInt bs,PetscScalar *x)
{
  PetscInt        nz,k;
  const PetscInt  *vj,bs2 = bs*bs;
  const MatScalar *v,*diag;
  PetscScalar     *xk,*xj,*xk_tmp;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(bs,&xk_tmp));
  for (k=0; k<mbs; k++) {
    v    = aa + bs2*ai[k];
    xk   = x + k*bs;    /* Dk*xk = k-th block of x */
    PetscCall(PetscArraycpy(xk_tmp,xk,bs)); /* xk_tmp <- xk */
    nz   = ai[k+1] - ai[k];
    vj   = aj + ai[k];
    xj   = x + (*vj)*bs; /* *vj-th block of x, *vj>k */
    while (nz--) {
      /* x(:) += U(k,:)^T*(Dk*xk) */
      PetscKernel_v_gets_v_plus_Atranspose_times_w(bs,xj,v,xk_tmp); /* xj <- xj + v^t * xk */
      vj++; xj = x + (*vj)*bs;
      v       += bs2;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag = aa+k*bs2;                            /* ptr to inv(Dk) */
    PetscKernel_w_gets_A_times_v(bs,xk_tmp,diag,xk); /* xk <- diag * xk */
  }
  PetscCall(PetscFree(xk_tmp));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_N_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscInt bs,PetscScalar *x)
{
  PetscInt        nz,k;
  const PetscInt  *vj,bs2 = bs*bs;
  const MatScalar *v;
  PetscScalar     *xk,*xj;

  PetscFunctionBegin;
  for (k=mbs-1; k>=0; k--) {
    v  = aa + bs2*ai[k];
    xk = x + k*bs;        /* xk */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    xj = x + (*vj)*bs;
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      PetscKernel_v_gets_v_plus_A_times_w(bs,xk,v,xj); /* xk <- xk + v*xj */
      vj++;
      v += bs2; xj = x + (*vj)*bs;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a=(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  PetscInt          bs =A->rmap->bs;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T * D * y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,bs*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_N_NaturalOrdering(ai,aj,aa,mbs,bs,x));

  /* solve U*x = y by back substitution */
  PetscCall(MatBackwardSolve_SeqSBAIJ_N_NaturalOrdering(ai,aj,aa,mbs,bs,x));

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a=(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  PetscInt          bs =A->rmap->bs;
  const MatScalar   *aa=a->a;
  const PetscScalar *b;
  PetscScalar       *x;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,bs*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_N_NaturalOrdering(ai,aj,aa,mbs,bs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - bs*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a=(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  PetscInt          bs =A->rmap->bs;
  const MatScalar   *aa=a->a;
  const PetscScalar *b;
  PetscScalar       *x;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,bs*mbs));
  PetscCall(MatBackwardSolve_SeqSBAIJ_N_NaturalOrdering(ai,aj,aa,mbs,bs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*(a->nz-mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_7_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs  =a->mbs,*ai=a->i,*aj=a->j,*r,*vj;
  PetscInt          nz,k,idx;
  const MatScalar   *aa=a->a,*v,*d;
  PetscScalar       *x,x0,x1,x2,x3,x4,x5,x6,*t,*tp;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&r));

  /* solve U^T * D * y = b by forward substitution */
  tp = t;
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx   = 7*r[k];
    tp[0] = b[idx];
    tp[1] = b[idx+1];
    tp[2] = b[idx+2];
    tp[3] = b[idx+3];
    tp[4] = b[idx+4];
    tp[5] = b[idx+5];
    tp[6] = b[idx+6];
    tp   += 7;
  }

  for (k=0; k<mbs; k++) {
    v  = aa + 49*ai[k];
    vj = aj + ai[k];
    tp = t + k*7;
    x0 =tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4]; x5=tp[5]; x6=tp[6];
    nz = ai[k+1] - ai[k];
    tp = t + (*vj)*7;
    while (nz--) {
      tp[0]+=  v[0]*x0 +  v[1]*x1 +  v[2]*x2 + v[3]*x3 + v[4]*x4 + v[5]*x5 + v[6]*x6;
      tp[1]+=  v[7]*x0 +  v[8]*x1 +  v[9]*x2+ v[10]*x3+ v[11]*x4+ v[12]*x5+ v[13]*x6;
      tp[2]+= v[14]*x0 + v[15]*x1 + v[16]*x2+ v[17]*x3+ v[18]*x4+ v[19]*x5+ v[20]*x6;
      tp[3]+= v[21]*x0 + v[22]*x1 + v[23]*x2+ v[24]*x3+ v[25]*x4+ v[26]*x5+ v[27]*x6;
      tp[4]+= v[28]*x0 + v[29]*x1 + v[30]*x2+ v[31]*x3+ v[32]*x4+ v[33]*x5+ v[34]*x6;
      tp[5]+= v[35]*x0 + v[36]*x1 + v[37]*x2+ v[38]*x3+ v[39]*x4+ v[40]*x5+ v[41]*x6;
      tp[6]+= v[42]*x0 + v[43]*x1 + v[44]*x2+ v[45]*x3+ v[46]*x4+ v[47]*x5+ v[48]*x6;
      vj++;
      tp = t + (*vj)*7;
      v += 49;
    }

    /* xk = inv(Dk)*(Dk*xk) */
    d     = aa+k*49;       /* ptr to inv(Dk) */
    tp    = t + k*7;
    tp[0] = d[0]*x0 + d[7]*x1 + d[14]*x2 + d[21]*x3 + d[28]*x4 + d[35]*x5 + d[42]*x6;
    tp[1] = d[1]*x0 + d[8]*x1 + d[15]*x2 + d[22]*x3 + d[29]*x4 + d[36]*x5 + d[43]*x6;
    tp[2] = d[2]*x0 + d[9]*x1 + d[16]*x2 + d[23]*x3 + d[30]*x4 + d[37]*x5 + d[44]*x6;
    tp[3] = d[3]*x0+ d[10]*x1 + d[17]*x2 + d[24]*x3 + d[31]*x4 + d[38]*x5 + d[45]*x6;
    tp[4] = d[4]*x0+ d[11]*x1 + d[18]*x2 + d[25]*x3 + d[32]*x4 + d[39]*x5 + d[46]*x6;
    tp[5] = d[5]*x0+ d[12]*x1 + d[19]*x2 + d[26]*x3 + d[33]*x4 + d[40]*x5 + d[47]*x6;
    tp[6] = d[6]*x0+ d[13]*x1 + d[20]*x2 + d[27]*x3 + d[34]*x4 + d[41]*x5 + d[48]*x6;
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 49*ai[k];
    vj = aj + ai[k];
    tp = t + k*7;
    x0 = tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4]; x5=tp[5];  x6=tp[6]; /* xk */
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*7;
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*tp[0] + v[7]*tp[1] + v[14]*tp[2] + v[21]*tp[3] + v[28]*tp[4] + v[35]*tp[5] + v[42]*tp[6];
      x1 += v[1]*tp[0] + v[8]*tp[1] + v[15]*tp[2] + v[22]*tp[3] + v[29]*tp[4] + v[36]*tp[5] + v[43]*tp[6];
      x2 += v[2]*tp[0] + v[9]*tp[1] + v[16]*tp[2] + v[23]*tp[3] + v[30]*tp[4] + v[37]*tp[5] + v[44]*tp[6];
      x3 += v[3]*tp[0]+ v[10]*tp[1] + v[17]*tp[2] + v[24]*tp[3] + v[31]*tp[4] + v[38]*tp[5] + v[45]*tp[6];
      x4 += v[4]*tp[0]+ v[11]*tp[1] + v[18]*tp[2] + v[25]*tp[3] + v[32]*tp[4] + v[39]*tp[5] + v[46]*tp[6];
      x5 += v[5]*tp[0]+ v[12]*tp[1] + v[19]*tp[2] + v[26]*tp[3] + v[33]*tp[4] + v[40]*tp[5] + v[47]*tp[6];
      x6 += v[6]*tp[0]+ v[13]*tp[1] + v[20]*tp[2] + v[27]*tp[3] + v[34]*tp[4] + v[41]*tp[5] + v[48]*tp[6];
      vj++;
      tp = t + (*vj)*7;
      v += 49;
    }
    tp       = t + k*7;
    tp[0]    = x0; tp[1]=x1; tp[2]=x2; tp[3]=x3; tp[4]=x4; tp[5]=x5; tp[6]=x6;
    idx      = 7*r[k];
    x[idx]   = x0;
    x[idx+1] = x1;
    x[idx+2] = x2;
    x[idx+3] = x3;
    x[idx+4] = x4;
    x[idx+5] = x5;
    x[idx+6] = x6;
  }

  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_7_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v,*d;
  PetscScalar     *xp,x0,x1,x2,x3,x4,x5,x6;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=0; k<mbs; k++) {
    v  = aa + 49*ai[k];
    xp = x + k*7;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4]; x5=xp[5]; x6=xp[6]; /* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj+nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v+49*nz,49*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*7;
      /* x(:) += U(k,:)^T*(Dk*xk) */
      xp[0]+=  v[0]*x0 +  v[1]*x1 +  v[2]*x2 + v[3]*x3 + v[4]*x4 + v[5]*x5 + v[6]*x6;
      xp[1]+=  v[7]*x0 +  v[8]*x1 +  v[9]*x2+ v[10]*x3+ v[11]*x4+ v[12]*x5+ v[13]*x6;
      xp[2]+= v[14]*x0 + v[15]*x1 + v[16]*x2+ v[17]*x3+ v[18]*x4+ v[19]*x5+ v[20]*x6;
      xp[3]+= v[21]*x0 + v[22]*x1 + v[23]*x2+ v[24]*x3+ v[25]*x4+ v[26]*x5+ v[27]*x6;
      xp[4]+= v[28]*x0 + v[29]*x1 + v[30]*x2+ v[31]*x3+ v[32]*x4+ v[33]*x5+ v[34]*x6;
      xp[5]+= v[35]*x0 + v[36]*x1 + v[37]*x2+ v[38]*x3+ v[39]*x4+ v[40]*x5+ v[41]*x6;
      xp[6]+= v[42]*x0 + v[43]*x1 + v[44]*x2+ v[45]*x3+ v[46]*x4+ v[47]*x5+ v[48]*x6;
      vj++;
      v += 49;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    d     = aa+k*49;       /* ptr to inv(Dk) */
    xp    = x + k*7;
    xp[0] = d[0]*x0 + d[7]*x1 + d[14]*x2 + d[21]*x3 + d[28]*x4 + d[35]*x5 + d[42]*x6;
    xp[1] = d[1]*x0 + d[8]*x1 + d[15]*x2 + d[22]*x3 + d[29]*x4 + d[36]*x5 + d[43]*x6;
    xp[2] = d[2]*x0 + d[9]*x1 + d[16]*x2 + d[23]*x3 + d[30]*x4 + d[37]*x5 + d[44]*x6;
    xp[3] = d[3]*x0+ d[10]*x1 + d[17]*x2 + d[24]*x3 + d[31]*x4 + d[38]*x5 + d[45]*x6;
    xp[4] = d[4]*x0+ d[11]*x1 + d[18]*x2 + d[25]*x3 + d[32]*x4 + d[39]*x5 + d[46]*x6;
    xp[5] = d[5]*x0+ d[12]*x1 + d[19]*x2 + d[26]*x3 + d[33]*x4 + d[40]*x5 + d[47]*x6;
    xp[6] = d[6]*x0+ d[13]*x1 + d[20]*x2 + d[27]*x3 + d[34]*x4 + d[41]*x5 + d[48]*x6;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_7_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v;
  PetscScalar     *xp,x0,x1,x2,x3,x4,x5,x6;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 49*ai[k];
    xp = x + k*7;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4]; x5=xp[5]; x6=xp[6]; /* xk */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj-nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v-49*nz,49*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*7;
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*xp[0] + v[7]*xp[1] + v[14]*xp[2] + v[21]*xp[3] + v[28]*xp[4] + v[35]*xp[5] + v[42]*xp[6];
      x1 += v[1]*xp[0] + v[8]*xp[1] + v[15]*xp[2] + v[22]*xp[3] + v[29]*xp[4] + v[36]*xp[5] + v[43]*xp[6];
      x2 += v[2]*xp[0] + v[9]*xp[1] + v[16]*xp[2] + v[23]*xp[3] + v[30]*xp[4] + v[37]*xp[5] + v[44]*xp[6];
      x3 += v[3]*xp[0]+ v[10]*xp[1] + v[17]*xp[2] + v[24]*xp[3] + v[31]*xp[4] + v[38]*xp[5] + v[45]*xp[6];
      x4 += v[4]*xp[0]+ v[11]*xp[1] + v[18]*xp[2] + v[25]*xp[3] + v[32]*xp[4] + v[39]*xp[5] + v[46]*xp[6];
      x5 += v[5]*xp[0]+ v[12]*xp[1] + v[19]*xp[2] + v[26]*xp[3] + v[33]*xp[4] + v[40]*xp[5] + v[47]*xp[6];
      x6 += v[6]*xp[0]+ v[13]*xp[1] + v[20]*xp[2] + v[27]*xp[3] + v[34]*xp[4] + v[41]*xp[5] + v[48]*xp[6];
      vj++;
      v += 49;
    }
    xp = x + k*7;
    xp[0]=x0; xp[1]=x1; xp[2]=x2; xp[3]=x3; xp[4]=x4; xp[5]=x5; xp[6]=x6;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a=(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T * D * y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,7*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_7_NaturalOrdering(ai,aj,aa,mbs,x));

  /* solve U*x = y by back substitution */
  PetscCall(MatBackwardSolve_SeqSBAIJ_7_NaturalOrdering(ai,aj,aa,mbs,x));

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a=(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,7*mbs));
  PetscCall(MatForwardSolve_SeqSBAIJ_7_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - A->rmap->bs*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a=(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,7*mbs));
  PetscCall(MatBackwardSolve_SeqSBAIJ_7_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*(a->nz-mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_6_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   =(Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs  =a->mbs,*ai=a->i,*aj=a->j,*r,*vj;
  PetscInt          nz,k,idx;
  const MatScalar   *aa=a->a,*v,*d;
  PetscScalar       *x,x0,x1,x2,x3,x4,x5,*t,*tp;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&r));

  /* solve U^T * D * y = b by forward substitution */
  tp = t;
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx   = 6*r[k];
    tp[0] = b[idx];
    tp[1] = b[idx+1];
    tp[2] = b[idx+2];
    tp[3] = b[idx+3];
    tp[4] = b[idx+4];
    tp[5] = b[idx+5];
    tp   += 6;
  }

  for (k=0; k<mbs; k++) {
    v  = aa + 36*ai[k];
    vj = aj + ai[k];
    tp = t + k*6;
    x0 =tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4]; x5=tp[5];
    nz = ai[k+1] - ai[k];
    tp = t + (*vj)*6;
    while (nz--) {
      tp[0] +=  v[0]*x0 +  v[1]*x1 +  v[2]*x2 + v[3]*x3 + v[4]*x4 + v[5]*x5;
      tp[1] +=  v[6]*x0 +  v[7]*x1 +  v[8]*x2 + v[9]*x3+ v[10]*x4+ v[11]*x5;
      tp[2] += v[12]*x0 + v[13]*x1 + v[14]*x2+ v[15]*x3+ v[16]*x4+ v[17]*x5;
      tp[3] += v[18]*x0 + v[19]*x1 + v[20]*x2+ v[21]*x3+ v[22]*x4+ v[23]*x5;
      tp[4] += v[24]*x0 + v[25]*x1 + v[26]*x2+ v[27]*x3+ v[28]*x4+ v[29]*x5;
      tp[5] += v[30]*x0 + v[31]*x1 + v[32]*x2+ v[33]*x3+ v[34]*x4+ v[35]*x5;
      vj++;
      tp = t + (*vj)*6;
      v += 36;
    }

    /* xk = inv(Dk)*(Dk*xk) */
    d     = aa+k*36;       /* ptr to inv(Dk) */
    tp    = t + k*6;
    tp[0] = d[0]*x0 + d[6]*x1 + d[12]*x2 + d[18]*x3 + d[24]*x4 + d[30]*x5;
    tp[1] = d[1]*x0 + d[7]*x1 + d[13]*x2 + d[19]*x3 + d[25]*x4 + d[31]*x5;
    tp[2] = d[2]*x0 + d[8]*x1 + d[14]*x2 + d[20]*x3 + d[26]*x4 + d[32]*x5;
    tp[3] = d[3]*x0 + d[9]*x1 + d[15]*x2 + d[21]*x3 + d[27]*x4 + d[33]*x5;
    tp[4] = d[4]*x0+ d[10]*x1 + d[16]*x2 + d[22]*x3 + d[28]*x4 + d[34]*x5;
    tp[5] = d[5]*x0+ d[11]*x1 + d[17]*x2 + d[23]*x3 + d[29]*x4 + d[35]*x5;
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 36*ai[k];
    vj = aj + ai[k];
    tp = t + k*6;
    x0 = tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4]; x5=tp[5]; /* xk */
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*6;
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*tp[0] + v[6]*tp[1] + v[12]*tp[2] + v[18]*tp[3] + v[24]*tp[4] + v[30]*tp[5];
      x1 += v[1]*tp[0] + v[7]*tp[1] + v[13]*tp[2] + v[19]*tp[3] + v[25]*tp[4] + v[31]*tp[5];
      x2 += v[2]*tp[0] + v[8]*tp[1] + v[14]*tp[2] + v[20]*tp[3] + v[26]*tp[4] + v[32]*tp[5];
      x3 += v[3]*tp[0] + v[9]*tp[1] + v[15]*tp[2] + v[21]*tp[3] + v[27]*tp[4] + v[33]*tp[5];
      x4 += v[4]*tp[0]+ v[10]*tp[1] + v[16]*tp[2] + v[22]*tp[3] + v[28]*tp[4] + v[34]*tp[5];
      x5 += v[5]*tp[0]+ v[11]*tp[1] + v[17]*tp[2] + v[23]*tp[3] + v[29]*tp[4] + v[35]*tp[5];
      vj++;
      tp = t + (*vj)*6;
      v += 36;
    }
    tp       = t + k*6;
    tp[0]    = x0; tp[1]=x1; tp[2]=x2; tp[3]=x3; tp[4]=x4; tp[5]=x5;
    idx      = 6*r[k];
    x[idx]   = x0;
    x[idx+1] = x1;
    x[idx+2] = x2;
    x[idx+3] = x3;
    x[idx+4] = x4;
    x[idx+5] = x5;
  }

  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_6_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v,*d;
  PetscScalar     *xp,x0,x1,x2,x3,x4,x5;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=0; k<mbs; k++) {
    v  = aa + 36*ai[k];
    xp = x + k*6;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4]; x5=xp[5]; /* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj+nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v+36*nz,36*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*6;
      /* x(:) += U(k,:)^T*(Dk*xk) */
      xp[0] +=  v[0]*x0 +  v[1]*x1 +  v[2]*x2 + v[3]*x3 + v[4]*x4 + v[5]*x5;
      xp[1] +=  v[6]*x0 +  v[7]*x1 +  v[8]*x2 + v[9]*x3+ v[10]*x4+ v[11]*x5;
      xp[2] += v[12]*x0 + v[13]*x1 + v[14]*x2+ v[15]*x3+ v[16]*x4+ v[17]*x5;
      xp[3] += v[18]*x0 + v[19]*x1 + v[20]*x2+ v[21]*x3+ v[22]*x4+ v[23]*x5;
      xp[4] += v[24]*x0 + v[25]*x1 + v[26]*x2+ v[27]*x3+ v[28]*x4+ v[29]*x5;
      xp[5] += v[30]*x0 + v[31]*x1 + v[32]*x2+ v[33]*x3+ v[34]*x4+ v[35]*x5;
      vj++;
      v += 36;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    d     = aa+k*36;       /* ptr to inv(Dk) */
    xp    = x + k*6;
    xp[0] = d[0]*x0 + d[6]*x1 + d[12]*x2 + d[18]*x3 + d[24]*x4 + d[30]*x5;
    xp[1] = d[1]*x0 + d[7]*x1 + d[13]*x2 + d[19]*x3 + d[25]*x4 + d[31]*x5;
    xp[2] = d[2]*x0 + d[8]*x1 + d[14]*x2 + d[20]*x3 + d[26]*x4 + d[32]*x5;
    xp[3] = d[3]*x0 + d[9]*x1 + d[15]*x2 + d[21]*x3 + d[27]*x4 + d[33]*x5;
    xp[4] = d[4]*x0+ d[10]*x1 + d[16]*x2 + d[22]*x3 + d[28]*x4 + d[34]*x5;
    xp[5] = d[5]*x0+ d[11]*x1 + d[17]*x2 + d[23]*x3 + d[29]*x4 + d[35]*x5;
  }
  PetscFunctionReturn(0);
}
PetscErrorCode MatBackwardSolve_SeqSBAIJ_6_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar   *v;
  PetscScalar       *xp,x0,x1,x2,x3,x4,x5;
  PetscInt          nz,k;
  const PetscInt    *vj;

  PetscFunctionBegin;
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 36*ai[k];
    xp = x + k*6;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4]; x5=xp[5]; /* xk */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj-nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v-36*nz,36*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*6;
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*xp[0] + v[6]*xp[1] + v[12]*xp[2] + v[18]*xp[3] + v[24]*xp[4] + v[30]*xp[5];
      x1 += v[1]*xp[0] + v[7]*xp[1] + v[13]*xp[2] + v[19]*xp[3] + v[25]*xp[4] + v[31]*xp[5];
      x2 += v[2]*xp[0] + v[8]*xp[1] + v[14]*xp[2] + v[20]*xp[3] + v[26]*xp[4] + v[32]*xp[5];
      x3 += v[3]*xp[0] + v[9]*xp[1] + v[15]*xp[2] + v[21]*xp[3] + v[27]*xp[4] + v[33]*xp[5];
      x4 += v[4]*xp[0]+ v[10]*xp[1] + v[16]*xp[2] + v[22]*xp[3] + v[28]*xp[4] + v[34]*xp[5];
      x5 += v[5]*xp[0]+ v[11]*xp[1] + v[17]*xp[2] + v[23]*xp[3] + v[29]*xp[4] + v[35]*xp[5];
      vj++;
      v += 36;
    }
    xp   = x + k*6;
    xp[0]=x0; xp[1]=x1; xp[2]=x2; xp[3]=x3; xp[4]=x4; xp[5]=x5;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T * D * y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,6*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_6_NaturalOrdering(ai,aj,aa,mbs,x));

  /* solve U*x = y by back substitution */
  PetscCall(MatBackwardSolve_SeqSBAIJ_6_NaturalOrdering(ai,aj,aa,mbs,x));

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,6*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_6_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - A->rmap->bs*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,6*mbs)); /* x <- b */
  PetscCall(MatBackwardSolve_SeqSBAIJ_6_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*(a->nz - mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_5_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a=(Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs  =a->mbs,*ai=a->i,*aj=a->j;
  const PetscInt    *r,*vj;
  PetscInt          nz,k,idx;
  const MatScalar   *aa=a->a,*v,*diag;
  PetscScalar       *x,x0,x1,x2,x3,x4,*t,*tp;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&r));

  /* solve U^T * D * y = b by forward substitution */
  tp = t;
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx   = 5*r[k];
    tp[0] = b[idx];
    tp[1] = b[idx+1];
    tp[2] = b[idx+2];
    tp[3] = b[idx+3];
    tp[4] = b[idx+4];
    tp   += 5;
  }

  for (k=0; k<mbs; k++) {
    v  = aa + 25*ai[k];
    vj = aj + ai[k];
    tp = t + k*5;
    x0 = tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4];
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*5;
    while (nz--) {
      tp[0] +=  v[0]*x0 + v[1]*x1 + v[2]*x2 + v[3]*x3 + v[4]*x4;
      tp[1] +=  v[5]*x0 + v[6]*x1 + v[7]*x2 + v[8]*x3 + v[9]*x4;
      tp[2] += v[10]*x0+ v[11]*x1+ v[12]*x2+ v[13]*x3+ v[14]*x4;
      tp[3] += v[15]*x0+ v[16]*x1+ v[17]*x2+ v[18]*x3+ v[19]*x4;
      tp[4] += v[20]*x0+ v[21]*x1+ v[22]*x2+ v[23]*x3+ v[24]*x4;
      vj++;
      tp = t + (*vj)*5;
      v += 25;
    }

    /* xk = inv(Dk)*(Dk*xk) */
    diag  = aa+k*25;          /* ptr to inv(Dk) */
    tp    = t + k*5;
    tp[0] = diag[0]*x0 + diag[5]*x1 + diag[10]*x2 + diag[15]*x3 + diag[20]*x4;
    tp[1] = diag[1]*x0 + diag[6]*x1 + diag[11]*x2 + diag[16]*x3 + diag[21]*x4;
    tp[2] = diag[2]*x0 + diag[7]*x1 + diag[12]*x2 + diag[17]*x3 + diag[22]*x4;
    tp[3] = diag[3]*x0 + diag[8]*x1 + diag[13]*x2 + diag[18]*x3 + diag[23]*x4;
    tp[4] = diag[4]*x0 + diag[9]*x1 + diag[14]*x2 + diag[19]*x3 + diag[24]*x4;
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 25*ai[k];
    vj = aj + ai[k];
    tp = t + k*5;
    x0 = tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; x4=tp[4]; /* xk */
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*5;
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*tp[0] + v[5]*tp[1] + v[10]*tp[2] + v[15]*tp[3] + v[20]*tp[4];
      x1 += v[1]*tp[0] + v[6]*tp[1] + v[11]*tp[2] + v[16]*tp[3] + v[21]*tp[4];
      x2 += v[2]*tp[0] + v[7]*tp[1] + v[12]*tp[2] + v[17]*tp[3] + v[22]*tp[4];
      x3 += v[3]*tp[0] + v[8]*tp[1] + v[13]*tp[2] + v[18]*tp[3] + v[23]*tp[4];
      x4 += v[4]*tp[0] + v[9]*tp[1] + v[14]*tp[2] + v[19]*tp[3] + v[24]*tp[4];
      vj++;
      tp = t + (*vj)*5;
      v += 25;
    }
    tp       = t + k*5;
    tp[0]    = x0; tp[1]=x1; tp[2]=x2; tp[3]=x3; tp[4]=x4;
    idx      = 5*r[k];
    x[idx]   = x0;
    x[idx+1] = x1;
    x[idx+2] = x2;
    x[idx+3] = x3;
    x[idx+4] = x4;
  }

  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_5_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v,*diag;
  PetscScalar     *xp,x0,x1,x2,x3,x4;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=0; k<mbs; k++) {
    v  = aa + 25*ai[k];
    xp = x + k*5;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4];/* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj+nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v+25*nz,25*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*5;
      /* x(:) += U(k,:)^T*(Dk*xk) */
      xp[0] +=  v[0]*x0 +  v[1]*x1 +  v[2]*x2 + v[3]*x3 + v[4]*x4;
      xp[1] +=  v[5]*x0 +  v[6]*x1 +  v[7]*x2 + v[8]*x3 + v[9]*x4;
      xp[2] += v[10]*x0 + v[11]*x1 + v[12]*x2+ v[13]*x3+ v[14]*x4;
      xp[3] += v[15]*x0 + v[16]*x1 + v[17]*x2+ v[18]*x3+ v[19]*x4;
      xp[4] += v[20]*x0 + v[21]*x1 + v[22]*x2+ v[23]*x3+ v[24]*x4;
      vj++;
      v += 25;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag  = aa+k*25;         /* ptr to inv(Dk) */
    xp    = x + k*5;
    xp[0] = diag[0]*x0 + diag[5]*x1 + diag[10]*x2 + diag[15]*x3 + diag[20]*x4;
    xp[1] = diag[1]*x0 + diag[6]*x1 + diag[11]*x2 + diag[16]*x3 + diag[21]*x4;
    xp[2] = diag[2]*x0 + diag[7]*x1 + diag[12]*x2 + diag[17]*x3 + diag[22]*x4;
    xp[3] = diag[3]*x0 + diag[8]*x1 + diag[13]*x2 + diag[18]*x3 + diag[23]*x4;
    xp[4] = diag[4]*x0 + diag[9]*x1 + diag[14]*x2 + diag[19]*x3 + diag[24]*x4;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_5_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v;
  PetscScalar     *xp,x0,x1,x2,x3,x4;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 25*ai[k];
    xp = x + k*5;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; x4=xp[4]; /* xk */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj-nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v-25*nz,25*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*5;
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*xp[0] + v[5]*xp[1] + v[10]*xp[2] + v[15]*xp[3] + v[20]*xp[4];
      x1 += v[1]*xp[0] + v[6]*xp[1] + v[11]*xp[2] + v[16]*xp[3] + v[21]*xp[4];
      x2 += v[2]*xp[0] + v[7]*xp[1] + v[12]*xp[2] + v[17]*xp[3] + v[22]*xp[4];
      x3 += v[3]*xp[0] + v[8]*xp[1] + v[13]*xp[2] + v[18]*xp[3] + v[23]*xp[4];
      x4 += v[4]*xp[0] + v[9]*xp[1] + v[14]*xp[2] + v[19]*xp[3] + v[24]*xp[4];
      vj++;
      v += 25;
    }
    xp   = x + k*5;
    xp[0]=x0; xp[1]=x1; xp[2]=x2; xp[3]=x3; xp[4]=x4;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T * D * y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,5*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_5_NaturalOrdering(ai,aj,aa,mbs,x));

  /* solve U*x = y by back substitution */
  PetscCall(MatBackwardSolve_SeqSBAIJ_5_NaturalOrdering(ai,aj,aa,mbs,x));

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,5*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_5_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - A->rmap->bs*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,5*mbs));
  PetscCall(MatBackwardSolve_SeqSBAIJ_5_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*(a->nz-mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_4_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   =(Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs  =a->mbs,*ai=a->i,*aj=a->j;
  const PetscInt    *r,*vj;
  PetscInt          nz,k,idx;
  const MatScalar   *aa=a->a,*v,*diag;
  PetscScalar       *x,x0,x1,x2,x3,*t,*tp;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&r));

  /* solve U^T * D * y = b by forward substitution */
  tp = t;
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx   = 4*r[k];
    tp[0] = b[idx];
    tp[1] = b[idx+1];
    tp[2] = b[idx+2];
    tp[3] = b[idx+3];
    tp   += 4;
  }

  for (k=0; k<mbs; k++) {
    v  = aa + 16*ai[k];
    vj = aj + ai[k];
    tp = t + k*4;
    x0 = tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3];
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*4;
    while (nz--) {
      tp[0] += v[0]*x0 + v[1]*x1 + v[2]*x2 + v[3]*x3;
      tp[1] += v[4]*x0 + v[5]*x1 + v[6]*x2 + v[7]*x3;
      tp[2] += v[8]*x0 + v[9]*x1 + v[10]*x2+ v[11]*x3;
      tp[3] += v[12]*x0+ v[13]*x1+ v[14]*x2+ v[15]*x3;
      vj++;
      tp = t + (*vj)*4;
      v += 16;
    }

    /* xk = inv(Dk)*(Dk*xk) */
    diag  = aa+k*16;          /* ptr to inv(Dk) */
    tp    = t + k*4;
    tp[0] = diag[0]*x0 + diag[4]*x1 + diag[8]*x2 + diag[12]*x3;
    tp[1] = diag[1]*x0 + diag[5]*x1 + diag[9]*x2 + diag[13]*x3;
    tp[2] = diag[2]*x0 + diag[6]*x1 + diag[10]*x2+ diag[14]*x3;
    tp[3] = diag[3]*x0 + diag[7]*x1 + diag[11]*x2+ diag[15]*x3;
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 16*ai[k];
    vj = aj + ai[k];
    tp = t + k*4;
    x0 = tp[0]; x1=tp[1]; x2=tp[2]; x3=tp[3]; /* xk */
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*4;
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*tp[0] + v[4]*tp[1] + v[8]*tp[2] + v[12]*tp[3];
      x1 += v[1]*tp[0] + v[5]*tp[1] + v[9]*tp[2] + v[13]*tp[3];
      x2 += v[2]*tp[0] + v[6]*tp[1]+ v[10]*tp[2] + v[14]*tp[3];
      x3 += v[3]*tp[0] + v[7]*tp[1]+ v[11]*tp[2] + v[15]*tp[3];
      vj++;
      tp = t + (*vj)*4;
      v += 16;
    }
    tp       = t + k*4;
    tp[0]    =x0; tp[1]=x1; tp[2]=x2; tp[3]=x3;
    idx      = 4*r[k];
    x[idx]   = x0;
    x[idx+1] = x1;
    x[idx+2] = x2;
    x[idx+3] = x3;
  }

  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_4_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v,*diag;
  PetscScalar     *xp,x0,x1,x2,x3;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=0; k<mbs; k++) {
    v  = aa + 16*ai[k];
    xp = x + k*4;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; /* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj+nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v+16*nz,16*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*4;
      /* x(:) += U(k,:)^T*(Dk*xk) */
      xp[0] += v[0]*x0 + v[1]*x1 + v[2]*x2 + v[3]*x3;
      xp[1] += v[4]*x0 + v[5]*x1 + v[6]*x2 + v[7]*x3;
      xp[2] += v[8]*x0 + v[9]*x1 + v[10]*x2+ v[11]*x3;
      xp[3] += v[12]*x0+ v[13]*x1+ v[14]*x2+ v[15]*x3;
      vj++;
      v += 16;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag  = aa+k*16;         /* ptr to inv(Dk) */
    xp    = x + k*4;
    xp[0] = diag[0]*x0 + diag[4]*x1 + diag[8]*x2 + diag[12]*x3;
    xp[1] = diag[1]*x0 + diag[5]*x1 + diag[9]*x2 + diag[13]*x3;
    xp[2] = diag[2]*x0 + diag[6]*x1 + diag[10]*x2+ diag[14]*x3;
    xp[3] = diag[3]*x0 + diag[7]*x1 + diag[11]*x2+ diag[15]*x3;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_4_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v;
  PetscScalar     *xp,x0,x1,x2,x3;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 16*ai[k];
    xp = x + k*4;
    x0 = xp[0]; x1=xp[1]; x2=xp[2]; x3=xp[3]; /* xk */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj-nz,nz,0,PETSC_PREFETCH_HINT_NTA);      /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v-16*nz,16*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*4;
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*xp[0] + v[4]*xp[1] + v[8]*xp[2] + v[12]*xp[3];
      x1 += v[1]*xp[0] + v[5]*xp[1] + v[9]*xp[2] + v[13]*xp[3];
      x2 += v[2]*xp[0] + v[6]*xp[1]+ v[10]*xp[2] + v[14]*xp[3];
      x3 += v[3]*xp[0] + v[7]*xp[1]+ v[11]*xp[2] + v[15]*xp[3];
      vj++;
      v += 16;
    }
    xp    = x + k*4;
    xp[0] = x0; xp[1] = x1; xp[2] = x2; xp[3] = x3;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T * D * y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,4*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_4_NaturalOrdering(ai,aj,aa,mbs,x));

  /* solve U*x = y by back substitution */
  PetscCall(MatBackwardSolve_SeqSBAIJ_4_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,4*mbs)); /* x <- b */
  PetscCall(MatForwardSolve_SeqSBAIJ_4_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - A->rmap->bs*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,4*mbs));
  PetscCall(MatBackwardSolve_SeqSBAIJ_4_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*(a->nz-mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_3_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs  =a->mbs,*ai=a->i,*aj=a->j;
  const PetscInt    *r;
  PetscInt          nz,k,idx;
  const PetscInt    *vj;
  const MatScalar   *aa=a->a,*v,*diag;
  PetscScalar       *x,x0,x1,x2,*t,*tp;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&r));

  /* solve U^T * D * y = b by forward substitution */
  tp = t;
  for (k=0; k<mbs; k++) { /* t <- perm(b) */
    idx   = 3*r[k];
    tp[0] = b[idx];
    tp[1] = b[idx+1];
    tp[2] = b[idx+2];
    tp   += 3;
  }

  for (k=0; k<mbs; k++) {
    v  = aa + 9*ai[k];
    vj = aj + ai[k];
    tp = t + k*3;
    x0 = tp[0]; x1 = tp[1]; x2 = tp[2];
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*3;
    while (nz--) {
      tp[0] += v[0]*x0 + v[1]*x1 + v[2]*x2;
      tp[1] += v[3]*x0 + v[4]*x1 + v[5]*x2;
      tp[2] += v[6]*x0 + v[7]*x1 + v[8]*x2;
      vj++;
      tp = t + (*vj)*3;
      v += 9;
    }

    /* xk = inv(Dk)*(Dk*xk) */
    diag  = aa+k*9;          /* ptr to inv(Dk) */
    tp    = t + k*3;
    tp[0] = diag[0]*x0 + diag[3]*x1 + diag[6]*x2;
    tp[1] = diag[1]*x0 + diag[4]*x1 + diag[7]*x2;
    tp[2] = diag[2]*x0 + diag[5]*x1 + diag[8]*x2;
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 9*ai[k];
    vj = aj + ai[k];
    tp = t + k*3;
    x0 = tp[0]; x1 = tp[1]; x2 = tp[2];  /* xk */
    nz = ai[k+1] - ai[k];

    tp = t + (*vj)*3;
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*tp[0] + v[3]*tp[1] + v[6]*tp[2];
      x1 += v[1]*tp[0] + v[4]*tp[1] + v[7]*tp[2];
      x2 += v[2]*tp[0] + v[5]*tp[1] + v[8]*tp[2];
      vj++;
      tp = t + (*vj)*3;
      v += 9;
    }
    tp       = t + k*3;
    tp[0]    = x0; tp[1] = x1; tp[2] = x2;
    idx      = 3*r[k];
    x[idx]   = x0;
    x[idx+1] = x1;
    x[idx+2] = x2;
  }

  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_3_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v,*diag;
  PetscScalar     *xp,x0,x1,x2;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=0; k<mbs; k++) {
    v  = aa + 9*ai[k];
    xp = x + k*3;
    x0 = xp[0]; x1 = xp[1]; x2 = xp[2]; /* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj+nz,nz,0,PETSC_PREFETCH_HINT_NTA);    /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v+9*nz,9*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*3;
      /* x(:) += U(k,:)^T*(Dk*xk) */
      xp[0] += v[0]*x0 + v[1]*x1 + v[2]*x2;
      xp[1] += v[3]*x0 + v[4]*x1 + v[5]*x2;
      xp[2] += v[6]*x0 + v[7]*x1 + v[8]*x2;
      vj++;
      v += 9;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag  = aa+k*9;         /* ptr to inv(Dk) */
    xp    = x + k*3;
    xp[0] = diag[0]*x0 + diag[3]*x1 + diag[6]*x2;
    xp[1] = diag[1]*x0 + diag[4]*x1 + diag[7]*x2;
    xp[2] = diag[2]*x0 + diag[5]*x1 + diag[8]*x2;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_3_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v;
  PetscScalar     *xp,x0,x1,x2;
  PetscInt        nz,k;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 9*ai[k];
    xp = x + k*3;
    x0 = xp[0]; x1 = xp[1]; x2 = xp[2];  /* xk */
    nz = ai[k+1] - ai[k];
    vj = aj + ai[k];
    PetscPrefetchBlock(vj-nz,nz,0,PETSC_PREFETCH_HINT_NTA);    /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v-9*nz,9*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      xp = x + (*vj)*3;
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*xp[0] + v[3]*xp[1] + v[6]*xp[2];
      x1 += v[1]*xp[0] + v[4]*xp[1] + v[7]*xp[2];
      x2 += v[2]*xp[0] + v[5]*xp[1] + v[8]*xp[2];
      vj++;
      v += 9;
    }
    xp    = x + k*3;
    xp[0] = x0; xp[1] = x1; xp[2] = x2;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T * D * y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,3*mbs));
  PetscCall(MatForwardSolve_SeqSBAIJ_3_NaturalOrdering(ai,aj,aa,mbs,x));

  /* solve U*x = y by back substitution */
  PetscCall(MatBackwardSolve_SeqSBAIJ_3_NaturalOrdering(ai,aj,aa,mbs,x));

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,3*mbs));
  PetscCall(MatForwardSolve_SeqSBAIJ_3_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - A->rmap->bs*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,3*mbs));
  PetscCall(MatBackwardSolve_SeqSBAIJ_3_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*(a->nz-mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_2_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   =(Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs  =a->mbs,*ai=a->i,*aj=a->j;
  const PetscInt    *r,*vj;
  PetscInt          nz,k,k2,idx;
  const MatScalar   *aa=a->a,*v,*diag;
  PetscScalar       *x,x0,x1,*t;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&r));

  /* solve U^T * D * y = perm(b) by forward substitution */
  for (k=0; k<mbs; k++) {  /* t <- perm(b) */
    idx      = 2*r[k];
    t[k*2]   = b[idx];
    t[k*2+1] = b[idx+1];
  }
  for (k=0; k<mbs; k++) {
    v  = aa + 4*ai[k];
    vj = aj + ai[k];
    k2 = k*2;
    x0 = t[k2]; x1 = t[k2+1];
    nz = ai[k+1] - ai[k];
    while (nz--) {
      t[(*vj)*2]   += v[0]*x0 + v[1]*x1;
      t[(*vj)*2+1] += v[2]*x0 + v[3]*x1;
      vj++; v      += 4;
    }
    diag    = aa+k*4; /* ptr to inv(Dk) */
    t[k2]   = diag[0]*x0 + diag[2]*x1;
    t[k2+1] = diag[1]*x0 + diag[3]*x1;
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 4*ai[k];
    vj = aj + ai[k];
    k2 = k*2;
    x0 = t[k2]; x1 = t[k2+1];
    nz = ai[k+1] - ai[k];
    while (nz--) {
      x0 += v[0]*t[(*vj)*2] + v[2]*t[(*vj)*2+1];
      x1 += v[1]*t[(*vj)*2] + v[3]*t[(*vj)*2+1];
      vj++;
      v += 4;
    }
    t[k2]    = x0;
    t[k2+1]  = x1;
    idx      = 2*r[k];
    x[idx]   = x0;
    x[idx+1] = x1;
  }

  PetscCall(ISRestoreIndices(isrow,&r));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_2_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v,*diag;
  PetscScalar     x0,x1;
  PetscInt        nz,k,k2;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=0; k<mbs; k++) {
    v  = aa + 4*ai[k];
    vj = aj + ai[k];
    k2 = k*2;
    x0 = x[k2]; x1 = x[k2+1];  /* Dk*xk = k-th block of x */
    nz = ai[k+1] - ai[k];
    PetscPrefetchBlock(vj+nz,nz,0,PETSC_PREFETCH_HINT_NTA);    /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v+4*nz,4*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      /* x(:) += U(k,:)^T*(Dk*xk) */
      x[(*vj)*2]   += v[0]*x0 + v[1]*x1;
      x[(*vj)*2+1] += v[2]*x0 + v[3]*x1;
      vj++; v      += 4;
    }
    /* xk = inv(Dk)*(Dk*xk) */
    diag    = aa+k*4;       /* ptr to inv(Dk) */
    x[k2]   = diag[0]*x0 + diag[2]*x1;
    x[k2+1] = diag[1]*x0 + diag[3]*x1;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_2_NaturalOrdering(const PetscInt *ai,const PetscInt *aj,const MatScalar *aa,PetscInt mbs,PetscScalar *x)
{
  const MatScalar *v;
  PetscScalar     x0,x1;
  PetscInt        nz,k,k2;
  const PetscInt  *vj;

  PetscFunctionBegin;
  for (k=mbs-1; k>=0; k--) {
    v  = aa + 4*ai[k];
    vj = aj + ai[k];
    k2 = k*2;
    x0 = x[k2]; x1 = x[k2+1];  /* xk */
    nz = ai[k+1] - ai[k];
    PetscPrefetchBlock(vj-nz,nz,0,PETSC_PREFETCH_HINT_NTA);    /* Indices for the next row (assumes same size as this one) */
    PetscPrefetchBlock(v-4*nz,4*nz,0,PETSC_PREFETCH_HINT_NTA); /* Entries for the next row */
    while (nz--) {
      /* xk += U(k,:)*x(:) */
      x0 += v[0]*x[(*vj)*2] + v[2]*x[(*vj)*2+1];
      x1 += v[1]*x[(*vj)*2] + v[3]*x[(*vj)*2+1];
      vj++;
      v += 4;
    }
    x[k2]   = x0;
    x[k2+1] = x1;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T * D * y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,2*mbs));
  PetscCall(MatForwardSolve_SeqSBAIJ_2_NaturalOrdering(ai,aj,aa,mbs,x));

  /* solve U*x = y by back substitution */
  PetscCall(MatBackwardSolve_SeqSBAIJ_2_NaturalOrdering(ai,aj,aa,mbs,x));

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->bs2*a->nz - (A->rmap->bs+2.0*a->bs2)*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,2*mbs));
  PetscCall(MatForwardSolve_SeqSBAIJ_2_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*a->nz - A->rmap->bs*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a =(Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j;
  const MatScalar   *aa=a->a;
  PetscScalar       *x;
  const PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,2*mbs));
  PetscCall(MatBackwardSolve_SeqSBAIJ_2_NaturalOrdering(ai,aj,aa,mbs,x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->bs2*(a->nz - mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_1(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   = (Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*rp,*vj,*adiag=a->diag;
  const MatScalar   *aa=a->a,*v;
  const PetscScalar *b;
  PetscScalar       *x,xk,*t;
  PetscInt          nz,k,j;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&rp));

  /* solve U^T*D*y = perm(b) by forward substitution */
  for (k=0; k<mbs; k++) t[k] = b[rp[k]];
  for (k=0; k<mbs; k++) {
    v  = aa + ai[k];
    vj = aj + ai[k];
    xk = t[k];
    nz = ai[k+1] - ai[k] - 1;
    for (j=0; j<nz; j++) t[vj[j]] += v[j]*xk;
    t[k] = xk*v[nz];   /* v[nz] = 1/D(k) */
  }

  /* solve U*perm(x) = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + adiag[k] - 1;
    vj = aj + adiag[k] - 1;
    nz = ai[k+1] - ai[k] - 1;
    for (j=0; j<nz; j++) t[k] += v[-j]*t[vj[-j]];
    x[rp[k]] = t[k];
  }

  PetscCall(ISRestoreIndices(isrow,&rp));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->nz - 3.0*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_1_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   = (Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*rp,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,xk,*t;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&rp));

  /* solve U^T*D*y = perm(b) by forward substitution */
  for (k=0; k<mbs; k++) t[k] = b[rp[k]];
  for (k=0; k<mbs; k++) {
    v  = aa + ai[k] + 1;
    vj = aj + ai[k] + 1;
    xk = t[k];
    nz = ai[k+1] - ai[k] - 1;
    while (nz--) t[*vj++] += (*v++) * xk;
    t[k] = xk*aa[ai[k]];  /* aa[k] = 1/D(k) */
  }

  /* solve U*perm(x) = y by back substitution */
  for (k=mbs-1; k>=0; k--) {
    v  = aa + ai[k] + 1;
    vj = aj + ai[k] + 1;
    nz = ai[k+1] - ai[k] - 1;
    while (nz--) t[k] += (*v++) * t[*vj++];
    x[rp[k]] = t[k];
  }

  PetscCall(ISRestoreIndices(isrow,&rp));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->nz - 3*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_1(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   = (Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*rp,*vj,*adiag=a->diag;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x,xk;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve U^T*D^(1/2)*x = perm(b) by forward substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(ISGetIndices(isrow,&rp));

  for (k=0; k<mbs; k++) x[k] = b[rp[k]];
  for (k=0; k<mbs; k++) {
    v  = aa + ai[k];
    vj = aj + ai[k];
    xk = x[k];
    nz = ai[k+1] - ai[k] - 1;
    while (nz--) x[*vj++] += (*v++) * xk;

    diagk = PetscRealPart(aa[adiag[k]]); /* note: aa[diag[k]] = 1/D(k) */
    PetscCheck(!PetscImaginaryPart(aa[adiag[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal must be real and nonnegative");
    x[k] = xk*PetscSqrtReal(diagk);
  }
  PetscCall(ISRestoreIndices(isrow,&rp));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_1_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   = (Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*rp,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x,xk;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve U^T*D^(1/2)*x = perm(b) by forward substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(ISGetIndices(isrow,&rp));

  for (k=0; k<mbs; k++) x[k] = b[rp[k]];
  for (k=0; k<mbs; k++) {
    v  = aa + ai[k] + 1;
    vj = aj + ai[k] + 1;
    xk = x[k];
    nz = ai[k+1] - ai[k] - 1;
    while (nz--) x[*vj++] += (*v++) * xk;

    diagk = PetscRealPart(aa[ai[k]]); /* note: aa[diag[k]] = 1/D(k) */
    PetscCheck(!PetscImaginaryPart(aa[ai[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal must be real and nonnegative");
    x[k] = xk*PetscSqrtReal(diagk);
  }
  PetscCall(ISRestoreIndices(isrow,&rp));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_1(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   = (Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*rp,*vj,*adiag=a->diag;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x,*t;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve D^(1/2)*U*perm(x) = b by back substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&rp));

  for (k=mbs-1; k>=0; k--) {
    v     = aa + ai[k];
    vj    = aj + ai[k];
    diagk = PetscRealPart(aa[adiag[k]]);
    PetscCheck(!PetscImaginaryPart(aa[adiag[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal must be real and nonnegative");
    t[k] = b[k] * PetscSqrtReal(diagk);
    nz   = ai[k+1] - ai[k] - 1;
    while (nz--) t[k] += (*v++) * t[*vj++];
    x[rp[k]] = t[k];
  }
  PetscCall(ISRestoreIndices(isrow,&rp));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a   = (Mat_SeqSBAIJ*)A->data;
  IS                isrow=a->row;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*rp,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x,*t;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve D^(1/2)*U*perm(x) = b by back substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  t    = a->solve_work;
  PetscCall(ISGetIndices(isrow,&rp));

  for (k=mbs-1; k>=0; k--) {
    v     = aa + ai[k] + 1;
    vj    = aj + ai[k] + 1;
    diagk = PetscRealPart(aa[ai[k]]);
    PetscCheck(!PetscImaginaryPart(aa[ai[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal must be real and nonnegative");
    t[k] = b[k] * PetscSqrtReal(diagk);
    nz   = ai[k+1] - ai[k] - 1;
    while (nz--) t[k] += (*v++) * t[*vj++];
    x[rp[k]] = t[k];
  }
  PetscCall(ISRestoreIndices(isrow,&rp));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolves_SeqSBAIJ_1(Mat A,Vecs bb,Vecs xx)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  if (A->rmap->bs == 1) {
    PetscCall(MatSolve_SeqSBAIJ_1(A,bb->v,xx->v));
  } else {
    IS                isrow=a->row;
    const PetscInt    *vj,mbs=a->mbs,*ai=a->i,*aj=a->j,*rp;
    const MatScalar   *aa=a->a,*v;
    PetscScalar       *x,*t;
    const PetscScalar *b;
    PetscInt          nz,k,n,i,j;

    if (bb->n > a->solves_work_n) {
      PetscCall(PetscFree(a->solves_work));
      PetscCall(PetscMalloc1(bb->n*A->rmap->N,&a->solves_work));
      a->solves_work_n = bb->n;
    }
    n    = bb->n;
    PetscCall(VecGetArrayRead(bb->v,&b));
    PetscCall(VecGetArray(xx->v,&x));
    t    = a->solves_work;

    PetscCall(ISGetIndices(isrow,&rp));

    /* solve U^T*D*y = perm(b) by forward substitution */
    for (k=0; k<mbs; k++) {
      for (i=0; i<n; i++) t[n*k+i] = b[rp[k]+i*mbs]; /* values are stored interlaced in t */
    }
    for (k=0; k<mbs; k++) {
      v  = aa + ai[k];
      vj = aj + ai[k];
      nz = ai[k+1] - ai[k] - 1;
      for (j=0; j<nz; j++) {
        for (i=0; i<n; i++) t[n*(*vj)+i] += (*v) * t[n*k+i];
        v++;vj++;
      }
      for (i=0; i<n; i++) t[n*k+i] *= aa[nz];  /* note: aa[nz] = 1/D(k) */
    }

    /* solve U*perm(x) = y by back substitution */
    for (k=mbs-1; k>=0; k--) {
      v  = aa + ai[k] - 1;
      vj = aj + ai[k] - 1;
      nz = ai[k+1] - ai[k] - 1;
      for (j=0; j<nz; j++) {
        for (i=0; i<n; i++) t[n*k+i] += (*v) * t[n*(*vj)+i];
        v++;vj++;
      }
      for (i=0; i<n; i++) x[rp[k]+i*mbs] = t[n*k+i];
    }

    PetscCall(ISRestoreIndices(isrow,&rp));
    PetscCall(VecRestoreArrayRead(bb->v,&b));
    PetscCall(VecRestoreArray(xx->v,&x));
    PetscCall(PetscLogFlops(bb->n*(4.0*a->nz - 3.0*mbs)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolves_SeqSBAIJ_1_inplace(Mat A,Vecs bb,Vecs xx)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  if (A->rmap->bs == 1) {
    PetscCall(MatSolve_SeqSBAIJ_1_inplace(A,bb->v,xx->v));
  } else {
    IS                isrow=a->row;
    const PetscInt    *vj,mbs=a->mbs,*ai=a->i,*aj=a->j,*rp;
    const MatScalar   *aa=a->a,*v;
    PetscScalar       *x,*t;
    const PetscScalar *b;
    PetscInt          nz,k,n,i;

    if (bb->n > a->solves_work_n) {
      PetscCall(PetscFree(a->solves_work));
      PetscCall(PetscMalloc1(bb->n*A->rmap->N,&a->solves_work));
      a->solves_work_n = bb->n;
    }
    n    = bb->n;
    PetscCall(VecGetArrayRead(bb->v,&b));
    PetscCall(VecGetArray(xx->v,&x));
    t    = a->solves_work;

    PetscCall(ISGetIndices(isrow,&rp));

    /* solve U^T*D*y = perm(b) by forward substitution */
    for (k=0; k<mbs; k++) {
      for (i=0; i<n; i++) t[n*k+i] = b[rp[k]+i*mbs];  /* values are stored interlaced in t */
    }
    for (k=0; k<mbs; k++) {
      v  = aa + ai[k];
      vj = aj + ai[k];
      nz = ai[k+1] - ai[k];
      while (nz--) {
        for (i=0; i<n; i++) t[n*(*vj)+i] += (*v) * t[n*k+i];
        v++;vj++;
      }
      for (i=0; i<n; i++) t[n*k+i] *= aa[k];  /* note: aa[k] = 1/D(k) */
    }

    /* solve U*perm(x) = y by back substitution */
    for (k=mbs-1; k>=0; k--) {
      v  = aa + ai[k];
      vj = aj + ai[k];
      nz = ai[k+1] - ai[k];
      while (nz--) {
        for (i=0; i<n; i++) t[n*k+i] += (*v) * t[n*(*vj)+i];
        v++;vj++;
      }
      for (i=0; i<n; i++) x[rp[k]+i*mbs] = t[n*k+i];
    }

    PetscCall(ISRestoreIndices(isrow,&rp));
    PetscCall(VecRestoreArrayRead(bb->v,&b));
    PetscCall(VecRestoreArray(xx->v,&x));
    PetscCall(PetscLogFlops(bb->n*(4.0*a->nz - 3.0*mbs)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*vj,*adiag=a->diag;
  const MatScalar   *aa=a->a,*v;
  const PetscScalar *b;
  PetscScalar       *x,xi;
  PetscInt          nz,i,j;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  /* solve U^T*D*y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,mbs));
  for (i=0; i<mbs; i++) {
    v  = aa + ai[i];
    vj = aj + ai[i];
    xi = x[i];
    nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */
    for (j=0; j<nz; j++) x[vj[j]] += v[j]*xi;
    x[i] = xi*v[nz];  /* v[nz] = aa[diag[i]] = 1/D(i) */
  }
  /* solve U*x = y by backward substitution */
  for (i=mbs-2; i>=0; i--) {
    xi = x[i];
    v  = aa + adiag[i] - 1; /* end of row i, excluding diag */
    vj = aj + adiag[i] - 1;
    nz = ai[i+1] - ai[i] - 1;
    for (j=0; j<nz; j++) xi += v[-j]*x[vj[-j]];
    x[i] = xi;
  }
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->nz - 3*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatSolve_SeqSBAIJ_1_NaturalOrdering(Mat A,Mat B,Mat X)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*vj,*adiag=a->diag;
  const MatScalar   *aa=a->a,*v;
  const PetscScalar *b;
  PetscScalar       *x,xi;
  PetscInt          nz,i,j,neq,ldb,ldx;
  PetscBool         isdense;

  PetscFunctionBegin;
  if (!mbs) PetscFunctionReturn(0);
  PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&isdense));
  PetscCheck(isdense,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"B matrix must be a SeqDense matrix");
  if (X != B) {
    PetscCall(PetscObjectTypeCompare((PetscObject)X,MATSEQDENSE,&isdense));
    PetscCheck(isdense,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"X matrix must be a SeqDense matrix");
  }
  PetscCall(MatDenseGetArrayRead(B,&b));
  PetscCall(MatDenseGetLDA(B,&ldb));
  PetscCall(MatDenseGetArray(X,&x));
  PetscCall(MatDenseGetLDA(X,&ldx));
  for (neq=0; neq<B->cmap->n; neq++) {
    /* solve U^T*D*y = b by forward substitution */
    PetscCall(PetscArraycpy(x,b,mbs));
    for (i=0; i<mbs; i++) {
      v  = aa + ai[i];
      vj = aj + ai[i];
      xi = x[i];
      nz = ai[i+1] - ai[i] - 1; /* exclude diag[i] */
      for (j=0; j<nz; j++) x[vj[j]] += v[j]*xi;
      x[i] = xi*v[nz];  /* v[nz] = aa[diag[i]] = 1/D(i) */
    }
    /* solve U*x = y by backward substitution */
    for (i=mbs-2; i>=0; i--) {
      xi = x[i];
      v  = aa + adiag[i] - 1; /* end of row i, excluding diag */
      vj = aj + adiag[i] - 1;
      nz = ai[i+1] - ai[i] - 1;
      for (j=0; j<nz; j++) xi += v[-j]*x[vj[-j]];
      x[i] = xi;
    }
    b += ldb;
    x += ldx;
  }
  PetscCall(MatDenseRestoreArrayRead(B,&b));
  PetscCall(MatDenseRestoreArray(X,&x));
  PetscCall(PetscLogFlops(B->cmap->n*(4.0*a->nz - 3*mbs)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscScalar       *x,xk;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  /* solve U^T*D*y = b by forward substitution */
  PetscCall(PetscArraycpy(x,b,mbs));
  for (k=0; k<mbs; k++) {
    v  = aa + ai[k] + 1;
    vj = aj + ai[k] + 1;
    xk = x[k];
    nz = ai[k+1] - ai[k] - 1;     /* exclude diag[k] */
    while (nz--) x[*vj++] += (*v++) * xk;
    x[k] = xk*aa[ai[k]];  /* note: aa[diag[k]] = 1/D(k) */
  }

  /* solve U*x = y by back substitution */
  for (k=mbs-2; k>=0; k--) {
    v  = aa + ai[k] + 1;
    vj = aj + ai[k] + 1;
    xk = x[k];
    nz = ai[k+1] - ai[k] - 1;
    while (nz--) {
      xk += (*v++) * x[*vj++];
    }
    x[k] = xk;
  }

  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(4.0*a->nz - 3*mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve U^T*D^(1/2)*x = b by forward substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,mbs));
  for (k=0; k<mbs; k++) {
    v  = aa + ai[k];
    vj = aj + ai[k];
    nz = ai[k+1] - ai[k] - 1;     /* exclude diag[k] */
    while (nz--) x[*vj++] += (*v++) * x[k];
    diagk = PetscRealPart(aa[adiag[k]]); /* note: aa[adiag[k]] = 1/D(k) */
    PetscCheck(!PetscImaginaryPart(aa[adiag[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal (%g,%g) must be real and nonnegative",(double)PetscRealPart(aa[adiag[k]]),(double)PetscImaginaryPart(aa[adiag[k]]));
    x[k] *= PetscSqrtReal(diagk);
  }
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatForwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve U^T*D^(1/2)*x = b by forward substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(PetscArraycpy(x,b,mbs));
  for (k=0; k<mbs; k++) {
    v  = aa + ai[k] + 1;
    vj = aj + ai[k] + 1;
    nz = ai[k+1] - ai[k] - 1;     /* exclude diag[k] */
    while (nz--) x[*vj++] += (*v++) * x[k];
    diagk = PetscRealPart(aa[ai[k]]); /* note: aa[diag[k]] = 1/D(k) */
    PetscCheck(!PetscImaginaryPart(aa[ai[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal (%g,%g) must be real and nonnegative",(double)PetscRealPart(aa[ai[k]]),(double)PetscImaginaryPart(aa[ai[k]]));
    x[k] *= PetscSqrtReal(diagk);
  }
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*adiag=a->diag,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve D^(1/2)*U*x = b by back substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  for (k=mbs-1; k>=0; k--) {
    v     = aa + ai[k];
    vj    = aj + ai[k];
    diagk = PetscRealPart(aa[adiag[k]]); /* note: aa[diag[k]] = 1/D(k) */
    PetscCheck(!PetscImaginaryPart(aa[adiag[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal must be real and nonnegative");
    x[k] = PetscSqrtReal(diagk)*b[k];
    nz   = ai[k+1] - ai[k] - 1;
    while (nz--) x[k] += (*v++) * x[*vj++];
  }
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqSBAIJ      *a = (Mat_SeqSBAIJ*)A->data;
  const PetscInt    mbs=a->mbs,*ai=a->i,*aj=a->j,*vj;
  const MatScalar   *aa=a->a,*v;
  PetscReal         diagk;
  PetscScalar       *x;
  const PetscScalar *b;
  PetscInt          nz,k;

  PetscFunctionBegin;
  /* solve D^(1/2)*U*x = b by back substitution */
  PetscCall(VecGetArrayRead(bb,&b));
  PetscCall(VecGetArray(xx,&x));

  for (k=mbs-1; k>=0; k--) {
    v     = aa + ai[k] + 1;
    vj    = aj + ai[k] + 1;
    diagk = PetscRealPart(aa[ai[k]]); /* note: aa[diag[k]] = 1/D(k) */
    PetscCheck(!PetscImaginaryPart(aa[ai[k]]) && diagk >= 0,PETSC_COMM_SELF,PETSC_ERR_SUP,"Diagonal must be real and nonnegative");
    x[k] = PetscSqrtReal(diagk)*b[k];
    nz   = ai[k+1] - ai[k] - 1;
    while (nz--) x[k] += (*v++) * x[*vj++];
  }
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(PetscLogFlops(2.0*a->nz - mbs));
  PetscFunctionReturn(0);
}

/* Use Modified Sparse Row storage for u and ju, see Saad pp.85 */
PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ_MSR(Mat B,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ   *a = (Mat_SeqSBAIJ*)A->data,*b;
  const PetscInt *rip,mbs = a->mbs,*ai,*aj;
  PetscInt       *jutmp,bs = A->rmap->bs,i;
  PetscInt       m,reallocs = 0,*levtmp;
  PetscInt       *prowl,*q,jmin,jmax,juidx,nzk,qm,*iu,*ju,k,j,vj,umax,maxadd;
  PetscInt       incrlev,*lev,shift,prow,nz;
  PetscReal      f = info->fill,levels = info->levels;
  PetscBool      perm_identity;

  PetscFunctionBegin;
  /* check whether perm is the identity mapping */
  PetscCall(ISIdentity(perm,&perm_identity));

  if (perm_identity) {
    a->permute = PETSC_FALSE;
    ai         = a->i; aj = a->j;
  } else { /*  non-trivial permutation */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported for sbaij matrix. Use aij format");
  }

  /* initialization */
  PetscCall(ISGetIndices(perm,&rip));
  umax  = (PetscInt)(f*ai[mbs] + 1);
  PetscCall(PetscMalloc1(umax,&lev));
  umax += mbs + 1;
  shift = mbs + 1;
  PetscCall(PetscMalloc1(mbs+1,&iu));
  PetscCall(PetscMalloc1(umax,&ju));
  iu[0] = mbs + 1;
  juidx = mbs + 1;
  /* prowl: linked list for pivot row */
  PetscCall(PetscMalloc3(mbs,&prowl,mbs,&q,mbs,&levtmp));
  /* q: linked list for col index */

  for (i=0; i<mbs; i++) {
    prowl[i] = mbs;
    q[i]     = 0;
  }

  /* for each row k */
  for (k=0; k<mbs; k++) {
    nzk  = 0;
    q[k] = mbs;
    /* copy current row into linked list */
    nz = ai[rip[k]+1] - ai[rip[k]];
    j  = ai[rip[k]];
    while (nz--) {
      vj = rip[aj[j++]];
      if (vj > k) {
        qm = k;
        do {
          m = qm; qm = q[m];
        } while (qm < vj);
        PetscCheck(qm != vj,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Duplicate entry in A");
        nzk++;
        q[m]       = vj;
        q[vj]      = qm;
        levtmp[vj] = 0;   /* initialize lev for nonzero element */
      }
    }

    /* modify nonzero structure of k-th row by computing fill-in
       for each row prow to be merged in */
    prow = k;
    prow = prowl[prow]; /* next pivot row (== 0 for symbolic factorization) */

    while (prow < k) {
      /* merge row prow into k-th row */
      jmin = iu[prow] + 1;
      jmax = iu[prow+1];
      qm   = k;
      for (j=jmin; j<jmax; j++) {
        incrlev = lev[j-shift] + 1;
        if (incrlev > levels) continue;
        vj = ju[j];
        do {
          m = qm; qm = q[m];
        } while (qm < vj);
        if (qm != vj) {      /* a fill */
          nzk++; q[m] = vj; q[vj] = qm; qm = vj;
          levtmp[vj] = incrlev;
        } else if (levtmp[vj] > incrlev) levtmp[vj] = incrlev;
      }
      prow = prowl[prow]; /* next pivot row */
    }

    /* add k to row list for first nonzero element in k-th row */
    if (nzk > 1) {
      i        = q[k]; /* col value of first nonzero element in k_th row of U */
      prowl[k] = prowl[i]; prowl[i] = k;
    }
    iu[k+1] = iu[k] + nzk;

    /* allocate more space to ju and lev if needed */
    if (iu[k+1] > umax) {
      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      maxadd = umax;
      if (maxadd < nzk) maxadd = (mbs-k)*(nzk+1)/2;
      umax += maxadd;

      /* allocate a longer ju */
      PetscCall(PetscMalloc1(umax,&jutmp));
      PetscCall(PetscArraycpy(jutmp,ju,iu[k]));
      PetscCall(PetscFree(ju));
      ju   = jutmp;

      PetscCall(PetscMalloc1(umax,&jutmp));
      PetscCall(PetscArraycpy(jutmp,lev,iu[k]-shift));
      PetscCall(PetscFree(lev));
      lev       = jutmp;
      reallocs += 2; /* count how many times we realloc */
    }

    /* save nonzero structure of k-th row in ju */
    i=k;
    while (nzk--) {
      i                = q[i];
      ju[juidx]        = i;
      lev[juidx-shift] = levtmp[i];
      juidx++;
    }
  }

#if defined(PETSC_USE_INFO)
  if (ai[mbs] != 0) {
    PetscReal af = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
    PetscCall(PetscInfo(A,"Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n",reallocs,(double)f,(double)af));
    PetscCall(PetscInfo(A,"Run with -pc_factor_fill %g or use \n",(double)af));
    PetscCall(PetscInfo(A,"PCFactorSetFill(pc,%g);\n",(double)af));
    PetscCall(PetscInfo(A,"for best performance.\n"));
  } else {
    PetscCall(PetscInfo(A,"Empty matrix\n"));
  }
#endif

  PetscCall(ISRestoreIndices(perm,&rip));
  PetscCall(PetscFree3(prowl,q,levtmp));
  PetscCall(PetscFree(lev));

  /* put together the new matrix */
  PetscCall(MatSeqSBAIJSetPreallocation(B,bs,0,NULL));

  /* PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)iperm)); */
  b    = (Mat_SeqSBAIJ*)(B)->data;
  PetscCall(PetscFree2(b->imax,b->ilen));

  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  /* the next line frees the default space generated by the Create() */
  PetscCall(PetscFree3(b->a,b->j,b->i));
  PetscCall(PetscMalloc1((iu[mbs]+1)*a->bs2,&b->a));
  b->j    = ju;
  b->i    = iu;
  b->diag = NULL;
  b->ilen = NULL;
  b->imax = NULL;

  PetscCall(ISDestroy(&b->row));
  PetscCall(ISDestroy(&b->icol));
  b->row  = perm;
  b->icol = perm;
  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(PetscMalloc1(bs*mbs+bs,&b->solve_work));
  /* In b structure:  Free imax, ilen, old a, old j.
     Allocate idnew, solve_work, new a, new j */
  PetscCall(PetscLogObjectMemory((PetscObject)B,(iu[mbs]-mbs)*(sizeof(PetscInt)+sizeof(MatScalar))));
  b->maxnz = b->nz = iu[mbs];

  (B)->info.factor_mallocs   = reallocs;
  (B)->info.fill_ratio_given = f;
  if (ai[mbs] != 0) {
    (B)->info.fill_ratio_needed = ((PetscReal)iu[mbs])/((PetscReal)ai[mbs]);
  } else {
    (B)->info.fill_ratio_needed = 0.0;
  }
  PetscCall(MatSeqSBAIJSetNumericFactorization_inplace(B,perm_identity));
  PetscFunctionReturn(0);
}

/*
  See MatICCFactorSymbolic_SeqAIJ() for description of its data structure
*/
#include <petscbt.h>
#include <../src/mat/utils/freespace.h>
PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data,*b;
  PetscBool          perm_identity,free_ij = PETSC_TRUE,missing;
  PetscInt           bs=A->rmap->bs,am=a->mbs,d,*ai=a->i,*aj= a->j;
  const PetscInt     *rip;
  PetscInt           reallocs=0,i,*ui,*udiag,*cols;
  PetscInt           jmin,jmax,nzk,k,j,*jl,prow,*il,nextprow;
  PetscInt           nlnk,*lnk,*lnk_lvl=NULL,ncols,*uj,**uj_ptr,**uj_lvl_ptr;
  PetscReal          fill          =info->fill,levels=info->levels;
  PetscFreeSpaceList free_space    =NULL,current_space=NULL;
  PetscFreeSpaceList free_space_lvl=NULL,current_space_lvl=NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT,A->rmap->n,A->cmap->n);
  PetscCall(MatMissingDiagonal(A,&missing,&d));
  PetscCheck(!missing,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %" PetscInt_FMT,d);
  if (bs > 1) {
    PetscCall(MatICCFactorSymbolic_SeqSBAIJ_inplace(fact,A,perm,info));
    PetscFunctionReturn(0);
  }

  /* check whether perm is the identity mapping */
  PetscCall(ISIdentity(perm,&perm_identity));
  PetscCheck(perm_identity,PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported for sbaij matrix. Use aij format");
  a->permute = PETSC_FALSE;

  PetscCall(PetscMalloc1(am+1,&ui));
  PetscCall(PetscMalloc1(am+1,&udiag));
  ui[0] = 0;

  /* ICC(0) without matrix ordering: simply rearrange column indices */
  if (!levels) {
    /* reuse the column pointers and row offsets for memory savings */
    for (i=0; i<am; i++) {
      ncols    = ai[i+1] - ai[i];
      ui[i+1]  = ui[i] + ncols;
      udiag[i] = ui[i+1] - 1; /* points to the last entry of U(i,:) */
    }
    PetscCall(PetscMalloc1(ui[am]+1,&uj));
    cols = uj;
    for (i=0; i<am; i++) {
      aj    = a->j + ai[i] + 1; /* 1st entry of U(i,:) without diagonal */
      ncols = ai[i+1] - ai[i] -1;
      for (j=0; j<ncols; j++) *cols++ = aj[j];
      *cols++ = i; /* diagonal is located as the last entry of U(i,:) */
    }
  } else { /* case: levels>0 */
    PetscCall(ISGetIndices(perm,&rip));

    /* initialization */
    /* jl: linked list for storing indices of the pivot rows
       il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
    PetscCall(PetscMalloc4(am,&uj_ptr,am,&uj_lvl_ptr,am,&il,am,&jl));
    for (i=0; i<am; i++) {
      jl[i] = am; il[i] = 0;
    }

    /* create and initialize a linked list for storing column indices of the active row k */
    nlnk = am + 1;
    PetscCall(PetscIncompleteLLCreate(am,am,nlnk,lnk,lnk_lvl,lnkbt));

    /* initial FreeSpace size is fill*(ai[am]+1) */
    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,ai[am]+1),&free_space));

    current_space = free_space;

    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,ai[am]+1),&free_space_lvl));

    current_space_lvl = free_space_lvl;

    for (k=0; k<am; k++) {  /* for each active row k */
      /* initialize lnk by the column indices of row k */
      nzk   = 0;
      ncols = ai[k+1] - ai[k];
      PetscCheck(ncols,PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Empty row %" PetscInt_FMT " in matrix ",k);
      cols = aj+ai[k];
      PetscCall(PetscIncompleteLLInit(ncols,cols,am,rip,&nlnk,lnk,lnk_lvl,lnkbt));
      nzk += nlnk;

      /* update lnk by computing fill-in for each pivot row to be merged in */
      prow = jl[k]; /* 1st pivot row */

      while (prow < k) {
        nextprow = jl[prow];

        /* merge prow into k-th row */
        jmin  = il[prow] + 1; /* index of the 2nd nzero entry in U(prow,k:am-1) */
        jmax  = ui[prow+1];
        ncols = jmax-jmin;
        i     = jmin - ui[prow];

        cols = uj_ptr[prow] + i;  /* points to the 2nd nzero entry in U(prow,k:am-1) */
        uj   = uj_lvl_ptr[prow] + i;  /* levels of cols */
        j    = *(uj - 1);
        PetscCall(PetscICCLLAddSorted(ncols,cols,levels,uj,am,&nlnk,lnk,lnk_lvl,lnkbt,j));
        nzk += nlnk;

        /* update il and jl for prow */
        if (jmin < jmax) {
          il[prow] = jmin;

          j = *cols; jl[prow] = jl[j]; jl[j] = prow;
        }
        prow = nextprow;
      }

      /* if free space is not available, make more free space */
      if (current_space->local_remaining<nzk) {
        i    = am - k + 1; /* num of unfactored rows */
        i    = PetscIntMultTruncate(i,PetscMin(nzk, i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
        PetscCall(PetscFreeSpaceGet(i,&current_space));
        PetscCall(PetscFreeSpaceGet(i,&current_space_lvl));
        reallocs++;
      }

      /* copy data into free_space and free_space_lvl, then initialize lnk */
      PetscCheck(nzk != 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Empty row %" PetscInt_FMT " in ICC matrix factor",k);
      PetscCall(PetscIncompleteLLClean(am,am,nzk,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt));

      /* add the k-th row into il and jl */
      if (nzk > 1) {
        i     = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:am-1) */
        jl[k] = jl[i]; jl[i] = k;
        il[k] = ui[k] + 1;
      }
      uj_ptr[k]     = current_space->array;
      uj_lvl_ptr[k] = current_space_lvl->array;

      current_space->array               += nzk;
      current_space->local_used          += nzk;
      current_space->local_remaining     -= nzk;
      current_space_lvl->array           += nzk;
      current_space_lvl->local_used      += nzk;
      current_space_lvl->local_remaining -= nzk;

      ui[k+1] = ui[k] + nzk;
    }

    PetscCall(ISRestoreIndices(perm,&rip));
    PetscCall(PetscFree4(uj_ptr,uj_lvl_ptr,il,jl));

    /* destroy list of free space and other temporary array(s) */
    PetscCall(PetscMalloc1(ui[am]+1,&uj));
    PetscCall(PetscFreeSpaceContiguous_Cholesky(&free_space,uj,am,ui,udiag)); /* store matrix factor  */
    PetscCall(PetscIncompleteLLDestroy(lnk,lnkbt));
    PetscCall(PetscFreeSpaceDestroy(free_space_lvl));

  } /* end of case: levels>0 || (levels=0 && !perm_identity) */

  /* put together the new matrix in MATSEQSBAIJ format */
  PetscCall(MatSeqSBAIJSetPreallocation(fact,bs,MAT_SKIP_ALLOCATION,NULL));

  b    = (Mat_SeqSBAIJ*)(fact)->data;
  PetscCall(PetscFree2(b->imax,b->ilen));

  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = free_ij;

  PetscCall(PetscMalloc1(ui[am]+1,&b->a));

  b->j         = uj;
  b->i         = ui;
  b->diag      = udiag;
  b->free_diag = PETSC_TRUE;
  b->ilen      = NULL;
  b->imax      = NULL;
  b->row       = perm;
  b->col       = perm;

  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(PetscObjectReference((PetscObject)perm));

  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  PetscCall(PetscMalloc1(am+1,&b->solve_work));
  PetscCall(PetscLogObjectMemory((PetscObject)fact,ui[am]*(sizeof(PetscInt)+sizeof(MatScalar))));

  b->maxnz = b->nz = ui[am];

  fact->info.factor_mallocs   = reallocs;
  fact->info.fill_ratio_given = fill;
  if (ai[am] != 0) {
    fact->info.fill_ratio_needed = ((PetscReal)ui[am])/ai[am];
  } else {
    fact->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[am] != 0) {
    PetscReal af = fact->info.fill_ratio_needed;
    PetscCall(PetscInfo(A,"Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af));
    PetscCall(PetscInfo(A,"Run with -pc_factor_fill %g or use \n",(double)af));
    PetscCall(PetscInfo(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af));
  } else {
    PetscCall(PetscInfo(A,"Empty matrix\n"));
  }
#endif
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering;
  PetscFunctionReturn(0);
}

PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ_inplace(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqSBAIJ       *a = (Mat_SeqSBAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  PetscBool          perm_identity,free_ij = PETSC_TRUE;
  PetscInt           bs=A->rmap->bs,am=a->mbs;
  const PetscInt     *cols,*rip,*ai=a->i,*aj=a->j;
  PetscInt           reallocs=0,i,*ui;
  PetscInt           jmin,jmax,nzk,k,j,*jl,prow,*il,nextprow;
  PetscInt           nlnk,*lnk,*lnk_lvl=NULL,ncols,*cols_lvl,*uj,**uj_ptr,**uj_lvl_ptr;
  PetscReal          fill          =info->fill,levels=info->levels,ratio_needed;
  PetscFreeSpaceList free_space    =NULL,current_space=NULL;
  PetscFreeSpaceList free_space_lvl=NULL,current_space_lvl=NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  /*
   This code originally uses Modified Sparse Row (MSR) storage
   (see page 85, "Iterative Methods ..." by Saad) for the output matrix B - bad choice!
   Then it is rewritten so the factor B takes seqsbaij format. However the associated
   MatCholeskyFactorNumeric_() have not been modified for the cases of bs>1,
   thus the original code in MSR format is still used for these cases.
   The code below should replace MatICCFactorSymbolic_SeqSBAIJ_MSR() whenever
   MatCholeskyFactorNumeric_() is modified for using sbaij symbolic factor.
  */
  if (bs > 1) {
    PetscCall(MatICCFactorSymbolic_SeqSBAIJ_MSR(fact,A,perm,info));
    PetscFunctionReturn(0);
  }

  /* check whether perm is the identity mapping */
  PetscCall(ISIdentity(perm,&perm_identity));
  PetscCheck(perm_identity,PETSC_COMM_SELF,PETSC_ERR_SUP,"Matrix reordering is not supported for sbaij matrix. Use aij format");
  a->permute = PETSC_FALSE;

  /* special case that simply copies fill pattern */
  if (!levels) {
    /* reuse the column pointers and row offsets for memory savings */
    ui           = a->i;
    uj           = a->j;
    free_ij      = PETSC_FALSE;
    ratio_needed = 1.0;
  } else { /* case: levels>0 */
    PetscCall(ISGetIndices(perm,&rip));

    /* initialization */
    PetscCall(PetscMalloc1(am+1,&ui));
    ui[0] = 0;

    /* jl: linked list for storing indices of the pivot rows
       il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
    PetscCall(PetscMalloc4(am,&uj_ptr,am,&uj_lvl_ptr,am,&il,am,&jl));
    for (i=0; i<am; i++) {
      jl[i] = am; il[i] = 0;
    }

    /* create and initialize a linked list for storing column indices of the active row k */
    nlnk = am + 1;
    PetscCall(PetscIncompleteLLCreate(am,am,nlnk,lnk,lnk_lvl,lnkbt));

    /* initial FreeSpace size is fill*(ai[am]+1) */
    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,ai[am]+1),&free_space));

    current_space = free_space;

    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,ai[am]+1),&free_space_lvl));

    current_space_lvl = free_space_lvl;

    for (k=0; k<am; k++) {  /* for each active row k */
      /* initialize lnk by the column indices of row rip[k] */
      nzk   = 0;
      ncols = ai[rip[k]+1] - ai[rip[k]];
      cols  = aj+ai[rip[k]];
      PetscCall(PetscIncompleteLLInit(ncols,cols,am,rip,&nlnk,lnk,lnk_lvl,lnkbt));
      nzk  += nlnk;

      /* update lnk by computing fill-in for each pivot row to be merged in */
      prow = jl[k]; /* 1st pivot row */

      while (prow < k) {
        nextprow = jl[prow];

        /* merge prow into k-th row */
        jmin     = il[prow] + 1; /* index of the 2nd nzero entry in U(prow,k:am-1) */
        jmax     = ui[prow+1];
        ncols    = jmax-jmin;
        i        = jmin - ui[prow];
        cols     = uj_ptr[prow] + i; /* points to the 2nd nzero entry in U(prow,k:am-1) */
        j        = *(uj_lvl_ptr[prow] + i - 1);
        cols_lvl = uj_lvl_ptr[prow]+i;
        PetscCall(PetscICCLLAddSorted(ncols,cols,levels,cols_lvl,am,&nlnk,lnk,lnk_lvl,lnkbt,j));
        nzk     += nlnk;

        /* update il and jl for prow */
        if (jmin < jmax) {
          il[prow] = jmin;
          j        = *cols;
          jl[prow] = jl[j];
          jl[j]    = prow;
        }
        prow = nextprow;
      }

      /* if free space is not available, make more free space */
      if (current_space->local_remaining<nzk) {
        i    = am - k + 1; /* num of unfactored rows */
        i    = PetscMin(PetscIntMultTruncate(i,nzk), PetscIntMultTruncate(i,i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
        PetscCall(PetscFreeSpaceGet(i,&current_space));
        PetscCall(PetscFreeSpaceGet(i,&current_space_lvl));
        reallocs++;
      }

      /* copy data into free_space and free_space_lvl, then initialize lnk */
      PetscCall(PetscIncompleteLLClean(am,am,nzk,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt));

      /* add the k-th row into il and jl */
      if (nzk-1 > 0) {
        i     = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:am-1) */
        jl[k] = jl[i]; jl[i] = k;
        il[k] = ui[k] + 1;
      }
      uj_ptr[k]     = current_space->array;
      uj_lvl_ptr[k] = current_space_lvl->array;

      current_space->array               += nzk;
      current_space->local_used          += nzk;
      current_space->local_remaining     -= nzk;
      current_space_lvl->array           += nzk;
      current_space_lvl->local_used      += nzk;
      current_space_lvl->local_remaining -= nzk;

      ui[k+1] = ui[k] + nzk;
    }

    PetscCall(ISRestoreIndices(perm,&rip));
    PetscCall(PetscFree4(uj_ptr,uj_lvl_ptr,il,jl));

    /* destroy list of free space and other temporary array(s) */
    PetscCall(PetscMalloc1(ui[am]+1,&uj));
    PetscCall(PetscFreeSpaceContiguous(&free_space,uj));
    PetscCall(PetscIncompleteLLDestroy(lnk,lnkbt));
    PetscCall(PetscFreeSpaceDestroy(free_space_lvl));
    if (ai[am] != 0) {
      ratio_needed = ((PetscReal)ui[am])/((PetscReal)ai[am]);
    } else {
      ratio_needed = 0.0;
    }
  } /* end of case: levels>0 || (levels=0 && !perm_identity) */

  /* put together the new matrix in MATSEQSBAIJ format */
  PetscCall(MatSeqSBAIJSetPreallocation(fact,bs,MAT_SKIP_ALLOCATION,NULL));

  b = (Mat_SeqSBAIJ*)(fact)->data;

  PetscCall(PetscFree2(b->imax,b->ilen));

  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = free_ij;

  PetscCall(PetscMalloc1(ui[am]+1,&b->a));

  b->j             = uj;
  b->i             = ui;
  b->diag          = NULL;
  b->ilen          = NULL;
  b->imax          = NULL;
  b->row           = perm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */

  PetscCall(PetscObjectReference((PetscObject)perm));
  b->icol = perm;
  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(PetscMalloc1(am+1,&b->solve_work));

  b->maxnz = b->nz = ui[am];

  fact->info.factor_mallocs    = reallocs;
  fact->info.fill_ratio_given  = fill;
  fact->info.fill_ratio_needed = ratio_needed;
#if defined(PETSC_USE_INFO)
  if (ai[am] != 0) {
    PetscReal af = fact->info.fill_ratio_needed;
    PetscCall(PetscInfo(A,"Reallocs %" PetscInt_FMT " Fill ratio:given %g needed %g\n",reallocs,(double)fill,(double)af));
    PetscCall(PetscInfo(A,"Run with -pc_factor_fill %g or use \n",(double)af));
    PetscCall(PetscInfo(A,"PCFactorSetFill(pc,%g) for best performance.\n",(double)af));
  } else {
    PetscCall(PetscInfo(A,"Empty matrix\n"));
  }
#endif
  if (perm_identity) {
    fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace;
  } else {
    fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqSBAIJ_1_inplace;
  }
  PetscFunctionReturn(0);
}
