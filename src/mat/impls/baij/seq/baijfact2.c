#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: baijfact2.c,v 1.23 1999/01/24 19:57:49 bsmith Exp bsmith $";
#endif
/*
    Factorization code for BAIJ format. 
*/


#include "src/mat/impls/baij/seq/baij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/ilu.h"
#include "src/inline/dot.h"

/* ----------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_N"
int MatSolve_SeqBAIJ_N(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ *)A->data;
  IS              iscol=a->col,isrow=a->row;
  int             *r,*c,ierr,i,n=a->mbs,*vi,*ai=a->i,*aj=a->j;
  int             nz,bs=a->bs,bs2=a->bs2,*rout,*cout;
  MatScalar       *aa=a->a,*v;
  Scalar          *x,*b,*sum,*tmp,*lsum;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  PetscMemcpy(tmp,b + bs*(*r++), bs*sizeof(Scalar));
  for ( i=1; i<n; i++ ) {
    v   = aa + bs2*ai[i];
    vi  = aj + ai[i];
    nz  = a->diag[i] - ai[i];
    sum = tmp + bs*i;
    PetscMemcpy(sum,b+bs*(*r++),bs*sizeof(Scalar));
    while (nz--) {
      Kernel_v_gets_v_minus_A_times_w(bs,sum,v,tmp+bs*(*vi++));
      v += bs2;
    }
  }
  /* backward solve the upper triangular */
  lsum = a->solve_work + a->n;
  for ( i=n-1; i>=0; i-- ){
    v   = aa + bs2*(a->diag[i] + 1);
    vi  = aj + a->diag[i] + 1;
    nz  = ai[i+1] - a->diag[i] - 1;
    PetscMemcpy(lsum,tmp+i*bs,bs*sizeof(Scalar));
    while (nz--) {
      Kernel_v_gets_v_minus_A_times_w(bs,lsum,v,tmp+bs*(*vi++));
      v += bs2;
    }
    Kernel_w_gets_A_times_v(bs,lsum,aa+bs2*a->diag[i],tmp+i*bs);
    PetscMemcpy(x + bs*(*c--),tmp+i*bs,bs*sizeof(Scalar));
  }

  ierr = ISRestoreIndices(isrow,&rout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout); CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b); CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x); CHKERRQ(ierr);
  PLogFlops(2*(a->bs2)*(a->nz) - a->bs*a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_7"
int MatSolve_SeqBAIJ_7(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ *)A->data;
  IS              iscol=a->col,isrow=a->row;
  int             *r,*c,ierr,i,n=a->mbs,*vi,*ai=a->i,*aj=a->j,nz,idx,idt,idc,*rout,*cout;
  int             *diag = a->diag;
  MatScalar       *aa=a->a,*v;
  Scalar          sum1,sum2,sum3,sum4,sum5,sum6,sum7,x1,x2,x3,x4,x5,x6,x7;
  Scalar          *x,*b,*tmp;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  idx    = 7*(*r++); 
  tmp[0] = b[idx];   tmp[1] = b[1+idx]; 
  tmp[2] = b[2+idx]; tmp[3] = b[3+idx]; tmp[4] = b[4+idx];
  tmp[5] = b[5+idx]; tmp[6] = b[6+idx]; 

  for ( i=1; i<n; i++ ) {
    v     = aa + 49*ai[i];
    vi    = aj + ai[i];
    nz    = diag[i] - ai[i];
    idx   = 7*(*r++); 
    sum1  = b[idx];sum2 = b[1+idx];sum3 = b[2+idx];sum4 = b[3+idx];
    sum5  = b[4+idx];sum6 = b[5+idx];sum7 = b[6+idx];
    while (nz--) {
      idx   = 7*(*vi++);
      x1    = tmp[idx];  x2 = tmp[1+idx];x3 = tmp[2+idx];
      x4    = tmp[3+idx];x5 = tmp[4+idx];
      x6    = tmp[5+idx];x7 = tmp[6+idx];
      sum1 -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      sum2 -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      sum3 -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      sum4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      sum5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      sum6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      sum7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v += 49;
    }
    idx = 7*i;
    tmp[idx]   = sum1;tmp[1+idx] = sum2;
    tmp[2+idx] = sum3;tmp[3+idx] = sum4; tmp[4+idx] = sum5;
    tmp[5+idx] = sum6;tmp[6+idx] = sum7;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + 49*diag[i] + 49;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    idt  = 7*i;
    sum1 = tmp[idt];  sum2 = tmp[1+idt]; 
    sum3 = tmp[2+idt];sum4 = tmp[3+idt]; sum5 = tmp[4+idt];
    sum6 = tmp[5+idt];sum7 = tmp[6+idt]; 
    while (nz--) {
      idx   = 7*(*vi++);
      x1    = tmp[idx];   x2 = tmp[1+idx];
      x3    = tmp[2+idx]; x4 = tmp[3+idx]; x5 = tmp[4+idx];
      x6    = tmp[5+idx]; x7 = tmp[6+idx];
      sum1 -= v[0]*x1 + v[7]*x2  + v[14]*x3 + v[21]*x4 + v[28]*x5 + v[35]*x6 + v[42]*x7;
      sum2 -= v[1]*x1 + v[8]*x2  + v[15]*x3 + v[22]*x4 + v[29]*x5 + v[36]*x6 + v[43]*x7;
      sum3 -= v[2]*x1 + v[9]*x2  + v[16]*x3 + v[23]*x4 + v[30]*x5 + v[37]*x6 + v[44]*x7;
      sum4 -= v[3]*x1 + v[10]*x2 + v[17]*x3 + v[24]*x4 + v[31]*x5 + v[38]*x6 + v[45]*x7;
      sum5 -= v[4]*x1 + v[11]*x2 + v[18]*x3 + v[25]*x4 + v[32]*x5 + v[39]*x6 + v[46]*x7;
      sum6 -= v[5]*x1 + v[12]*x2 + v[19]*x3 + v[26]*x4 + v[33]*x5 + v[40]*x6 + v[47]*x7;
      sum7 -= v[6]*x1 + v[13]*x2 + v[20]*x3 + v[27]*x4 + v[34]*x5 + v[41]*x6 + v[48]*x7;
      v += 49;
    }
    idc = 7*(*c--);
    v   = aa + 49*diag[i];
    x[idc]   = tmp[idt]   = v[0]*sum1+v[7]*sum2+v[14]*sum3+
                                 v[21]*sum4+v[28]*sum5+v[35]*sum6+v[42]*sum7;
    x[1+idc] = tmp[1+idt] = v[1]*sum1+v[8]*sum2+v[15]*sum3+
                                 v[22]*sum4+v[29]*sum5+v[36]*sum6+v[43]*sum7;
    x[2+idc] = tmp[2+idt] = v[2]*sum1+v[9]*sum2+v[16]*sum3+
                                 v[23]*sum4+v[30]*sum5+v[37]*sum6+v[44]*sum7;
    x[3+idc] = tmp[3+idt] = v[3]*sum1+v[10]*sum2+v[17]*sum3+
                                 v[24]*sum4+v[31]*sum5+v[38]*sum6+v[45]*sum7;
    x[4+idc] = tmp[4+idt] = v[4]*sum1+v[11]*sum2+v[18]*sum3+
                                 v[25]*sum4+v[32]*sum5+v[39]*sum6+v[46]*sum7;
    x[5+idc] = tmp[5+idt] = v[5]*sum1+v[12]*sum2+v[19]*sum3+
                                 v[26]*sum4+v[33]*sum5+v[40]*sum6+v[47]*sum7;
    x[6+idc] = tmp[6+idt] = v[6]*sum1+v[13]*sum2+v[20]*sum3+
                                 v[27]*sum4+v[34]*sum5+v[41]*sum6+v[48]*sum7;
  }

  ierr = ISRestoreIndices(isrow,&rout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout); CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b); CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x); CHKERRQ(ierr);
  PLogFlops(2*49*(a->nz) - 7*a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_5"
int MatSolve_SeqBAIJ_5(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ *)A->data;
  IS              iscol=a->col,isrow=a->row;
  int             *r,*c,ierr,i,n=a->mbs,*vi,*ai=a->i,*aj=a->j,nz,idx,idt,idc,*rout,*cout;
  int             *diag = a->diag;
  MatScalar       *aa=a->a,*v;
  Scalar          *x,*b,sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5,*tmp;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  idx    = 5*(*r++); 
  tmp[0] = b[idx];   tmp[1] = b[1+idx]; 
  tmp[2] = b[2+idx]; tmp[3] = b[3+idx]; tmp[4] = b[4+idx];
  for ( i=1; i<n; i++ ) {
    v     = aa + 25*ai[i];
    vi    = aj + ai[i];
    nz    = diag[i] - ai[i];
    idx   = 5*(*r++); 
    sum1  = b[idx];sum2 = b[1+idx];sum3 = b[2+idx];sum4 = b[3+idx];
    sum5  = b[4+idx];
    while (nz--) {
      idx   = 5*(*vi++);
      x1    = tmp[idx];  x2 = tmp[1+idx];x3 = tmp[2+idx];
      x4    = tmp[3+idx];x5 = tmp[4+idx];
      sum1 -= v[0]*x1 + v[5]*x2 + v[10]*x3 + v[15]*x4 + v[20]*x5;
      sum2 -= v[1]*x1 + v[6]*x2 + v[11]*x3 + v[16]*x4 + v[21]*x5;
      sum3 -= v[2]*x1 + v[7]*x2 + v[12]*x3 + v[17]*x4 + v[22]*x5;
      sum4 -= v[3]*x1 + v[8]*x2 + v[13]*x3 + v[18]*x4 + v[23]*x5;
      sum5 -= v[4]*x1 + v[9]*x2 + v[14]*x3 + v[19]*x4 + v[24]*x5;
      v += 25;
    }
    idx = 5*i;
    tmp[idx]   = sum1;tmp[1+idx] = sum2;
    tmp[2+idx] = sum3;tmp[3+idx] = sum4; tmp[4+idx] = sum5;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + 25*diag[i] + 25;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    idt  = 5*i;
    sum1 = tmp[idt];  sum2 = tmp[1+idt]; 
    sum3 = tmp[2+idt];sum4 = tmp[3+idt]; sum5 = tmp[4+idt];
    while (nz--) {
      idx   = 5*(*vi++);
      x1    = tmp[idx];   x2 = tmp[1+idx];
      x3    = tmp[2+idx]; x4 = tmp[3+idx]; x5 = tmp[4+idx];
      sum1 -= v[0]*x1 + v[5]*x2 + v[10]*x3 + v[15]*x4 + v[20]*x5;
      sum2 -= v[1]*x1 + v[6]*x2 + v[11]*x3 + v[16]*x4 + v[21]*x5; 
      sum3 -= v[2]*x1 + v[7]*x2 + v[12]*x3 + v[17]*x4 + v[22]*x5;
      sum4 -= v[3]*x1 + v[8]*x2 + v[13]*x3 + v[18]*x4 + v[23]*x5;
      sum5 -= v[4]*x1 + v[9]*x2 + v[14]*x3 + v[19]*x4 + v[24]*x5;
      v += 25;
    }
    idc = 5*(*c--);
    v   = aa + 25*diag[i];
    x[idc]   = tmp[idt]   = v[0]*sum1+v[5]*sum2+v[10]*sum3+
                                 v[15]*sum4+v[20]*sum5;
    x[1+idc] = tmp[1+idt] = v[1]*sum1+v[6]*sum2+v[11]*sum3+
                                 v[16]*sum4+v[21]*sum5;
    x[2+idc] = tmp[2+idt] = v[2]*sum1+v[7]*sum2+v[12]*sum3+
                                 v[17]*sum4+v[22]*sum5;
    x[3+idc] = tmp[3+idt] = v[3]*sum1+v[8]*sum2+v[13]*sum3+
                                 v[18]*sum4+v[23]*sum5;
    x[4+idc] = tmp[4+idt] = v[4]*sum1+v[9]*sum2+v[14]*sum3+
                                 v[19]*sum4+v[24]*sum5;
  }

  ierr = ISRestoreIndices(isrow,&rout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout); CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b); CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x); CHKERRQ(ierr);
  PLogFlops(2*25*(a->nz) - 5*a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_4"
int MatSolve_SeqBAIJ_4(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *)A->data;
  IS              iscol=a->col,isrow=a->row;
  int             *r,*c,ierr,i,n=a->mbs,*vi,*ai=a->i,*aj=a->j,nz,idx,idt,idc,*rout,*cout;
  int             *diag = a->diag;
  MatScalar       *aa=a->a,*v;
  Scalar          *x,*b,sum1,sum2,sum3,sum4,x1,x2,x3,x4,*tmp;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  idx    = 4*(*r++); 
  tmp[0] = b[idx];   tmp[1] = b[1+idx]; 
  tmp[2] = b[2+idx]; tmp[3] = b[3+idx];
  for ( i=1; i<n; i++ ) {
    v     = aa + 16*ai[i];
    vi    = aj + ai[i];
    nz    = diag[i] - ai[i];
    idx   = 4*(*r++); 
    sum1  = b[idx];sum2 = b[1+idx];sum3 = b[2+idx];sum4 = b[3+idx];
    while (nz--) {
      idx   = 4*(*vi++);
      x1    = tmp[idx];x2 = tmp[1+idx];x3 = tmp[2+idx];x4 = tmp[3+idx];
      sum1 -= v[0]*x1 + v[4]*x2 + v[8]*x3  + v[12]*x4;
      sum2 -= v[1]*x1 + v[5]*x2 + v[9]*x3  + v[13]*x4;
      sum3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
      sum4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;
      v    += 16;
    }
    idx        = 4*i;
    tmp[idx]   = sum1;tmp[1+idx] = sum2;
    tmp[2+idx] = sum3;tmp[3+idx] = sum4;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + 16*diag[i] + 16;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    idt  = 4*i;
    sum1 = tmp[idt];  sum2 = tmp[1+idt]; 
    sum3 = tmp[2+idt];sum4 = tmp[3+idt];
    while (nz--) {
      idx   = 4*(*vi++);
      x1    = tmp[idx];   x2 = tmp[1+idx];
      x3    = tmp[2+idx]; x4 = tmp[3+idx];
      sum1 -= v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
      sum2 -= v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4; 
      sum3 -= v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
      sum4 -= v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
      v += 16;
    }
    idc      = 4*(*c--);
    v        = aa + 16*diag[i];
    x[idc]   = tmp[idt]   = v[0]*sum1+v[4]*sum2+v[8]*sum3+v[12]*sum4;
    x[1+idc] = tmp[1+idt] = v[1]*sum1+v[5]*sum2+v[9]*sum3+v[13]*sum4;
    x[2+idc] = tmp[2+idt] = v[2]*sum1+v[6]*sum2+v[10]*sum3+v[14]*sum4;
    x[3+idc] = tmp[3+idt] = v[3]*sum1+v[7]*sum2+v[11]*sum3+v[15]*sum4;
  }

  ierr = ISRestoreIndices(isrow,&rout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout); CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b); CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x); CHKERRQ(ierr);
  PLogFlops(2*16*(a->nz) - 4*a->n);
  PetscFunctionReturn(0);
}


/*
      Special case where the matrix was ILU(0) factored in the natural
   ordering. This eliminates the need for the column and row permutation.
*/
#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_4_NaturalOrdering"
int MatSolve_SeqBAIJ_4_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *)A->data;
  int             n=a->mbs,*ai=a->i,*aj=a->j;
  int             ierr,*diag = a->diag;
  MatScalar       *aa=a->a;
  Scalar          *x,*b;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 

#if defined(USE_FORTRAN_KERNEL_SOLVEBAIJBLAS)
  {
    static Scalar w[2000]; /* very BAD need to fix */
    fortransolvebaij4blas_(&n,x,ai,aj,diag,aa,b,w);
  }
#elif defined(USE_FORTRAN_KERNEL_SOLVEBAIJ)
  {
    static Scalar w[2000]; /* very BAD need to fix */
    fortransolvebaij4_(&n,x,ai,aj,diag,aa,b,w);
  }
#elif defined(USE_FORTRAN_KERNEL_SOLVEBAIJUNROLL)
  fortransolvebaij4unroll_(&n,x,ai,aj,diag,aa,b);
#else
  {
    Scalar    sum1,sum2,sum3,sum4,x1,x2,x3,x4;
    MatScalar *v;
    int       jdx,idt,idx,nz,*vi,i;

  /* forward solve the lower triangular */
  idx    = 0;
  x[0]   = b[0]; x[1] = b[1]; x[2] = b[2]; x[3] = b[3];
  for ( i=1; i<n; i++ ) {
    v     =  aa      + 16*ai[i];
    vi    =  aj      + ai[i];
    nz    =  diag[i] - ai[i];
    idx   +=  4;
    sum1  =  b[idx];sum2 = b[1+idx];sum3 = b[2+idx];sum4 = b[3+idx];
    while (nz--) {
      jdx   = 4*(*vi++);
      x1    = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];
      sum1 -= v[0]*x1 + v[4]*x2 + v[8]*x3  + v[12]*x4;
      sum2 -= v[1]*x1 + v[5]*x2 + v[9]*x3  + v[13]*x4;
      sum3 -= v[2]*x1 + v[6]*x2 + v[10]*x3 + v[14]*x4;
      sum4 -= v[3]*x1 + v[7]*x2 + v[11]*x3 + v[15]*x4;
      v    += 16;
    }
    x[idx]   = sum1;
    x[1+idx] = sum2;
    x[2+idx] = sum3;
    x[3+idx] = sum4;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + 16*diag[i] + 16;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    idt  = 4*i;
    sum1 = x[idt];  sum2 = x[1+idt]; 
    sum3 = x[2+idt];sum4 = x[3+idt];
    while (nz--) {
      idx   = 4*(*vi++);
      x1    = x[idx];   x2 = x[1+idx];x3    = x[2+idx]; x4 = x[3+idx];
      sum1 -= v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
      sum2 -= v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4; 
      sum3 -= v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
      sum4 -= v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
      v    += 16;
    }
    v        = aa + 16*diag[i];
    x[idt]   = v[0]*sum1 + v[4]*sum2 + v[8]*sum3  + v[12]*sum4;
    x[1+idt] = v[1]*sum1 + v[5]*sum2 + v[9]*sum3  + v[13]*sum4;
    x[2+idt] = v[2]*sum1 + v[6]*sum2 + v[10]*sum3 + v[14]*sum4;
    x[3+idt] = v[3]*sum1 + v[7]*sum2 + v[11]*sum3 + v[15]*sum4;
  }
  }
#endif

  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  PLogFlops(2*16*(a->nz) - 4*a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_5_NaturalOrdering"
int MatSolve_SeqBAIJ_5_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *)A->data;
  int             i,n=a->mbs,*vi,*ai=a->i,*aj=a->j,nz,idx,idt;
  int             ierr,*diag = a->diag,jdx;
  MatScalar       *aa=a->a,*v;
  Scalar          *x,*b,sum1,sum2,sum3,sum4,sum5,x1,x2,x3,x4,x5;;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
  /* forward solve the lower triangular */
  idx    = 0;
  x[0] = b[idx]; x[1] = b[1+idx]; x[2] = b[2+idx]; x[3] = b[3+idx];x[4] = b[4+idx];
  for ( i=1; i<n; i++ ) {
    v     =  aa + 25*ai[i];
    vi    =  aj + ai[i];
    nz    =  diag[i] - ai[i];
    idx   =  5*i;
    sum1  =  b[idx];sum2 = b[1+idx];sum3 = b[2+idx];sum4 = b[3+idx];sum5 = b[4+idx];
    while (nz--) {
      jdx   = 5*(*vi++);
      x1    = x[jdx];x2 = x[1+jdx];x3 = x[2+jdx];x4 = x[3+jdx];x5 = x[4+jdx];
      sum1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      sum2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      sum3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      sum4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      sum5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v    += 25;
    }
    x[idx]   = sum1;
    x[1+idx] = sum2;
    x[2+idx] = sum3;
    x[3+idx] = sum4;
    x[4+idx] = sum5;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + 25*diag[i] + 25;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    idt  = 5*i;
    sum1 = x[idt];  sum2 = x[1+idt]; 
    sum3 = x[2+idt];sum4 = x[3+idt]; sum5 = x[4+idt];
    while (nz--) {
      idx   = 5*(*vi++);
      x1    = x[idx];   x2 = x[1+idx];x3    = x[2+idx]; x4 = x[3+idx]; x5 = x[4+idx];
      sum1 -= v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
      sum2 -= v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
      sum3 -= v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
      sum4 -= v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
      sum5 -= v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
      v    += 25;
    }
    v        = aa + 25*diag[i];
    x[idt]   = v[0]*sum1 + v[5]*sum2 + v[10]*sum3  + v[15]*sum4 + v[20]*sum5;
    x[1+idt] = v[1]*sum1 + v[6]*sum2 + v[11]*sum3  + v[16]*sum4 + v[21]*sum5;
    x[2+idt] = v[2]*sum1 + v[7]*sum2 + v[12]*sum3  + v[17]*sum4 + v[22]*sum5;
    x[3+idt] = v[3]*sum1 + v[8]*sum2 + v[13]*sum3  + v[18]*sum4 + v[23]*sum5;
    x[4+idt] = v[4]*sum1 + v[9]*sum2 + v[14]*sum3  + v[19]*sum4 + v[24]*sum5;
  }

  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  PLogFlops(2*25*(a->nz) - 5*a->n);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_3"
int MatSolve_SeqBAIJ_3(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ *)A->data;
  IS              iscol=a->col,isrow=a->row;
  int             *r,*c,ierr,i,n=a->mbs,*vi,*ai=a->i,*aj=a->j,nz,idx,idt,idc,*rout,*cout;
  int             *diag = a->diag;
  MatScalar       *aa=a->a,*v;
  Scalar          *x,*b,sum1,sum2,sum3,x1,x2,x3,*tmp;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  idx    = 3*(*r++); 
  tmp[0] = b[idx]; tmp[1] = b[1+idx]; tmp[2] = b[2+idx];
  for ( i=1; i<n; i++ ) {
    v     = aa + 9*ai[i];
    vi    = aj + ai[i];
    nz    = diag[i] - ai[i];
    idx   = 3*(*r++); 
    sum1  = b[idx]; sum2 = b[1+idx]; sum3 = b[2+idx];
    while (nz--) {
      idx   = 3*(*vi++);
      x1    = tmp[idx]; x2 = tmp[1+idx]; x3 = tmp[2+idx];
      sum1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      sum2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      sum3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;
      v += 9;
    }
    idx = 3*i;
    tmp[idx] = sum1; tmp[1+idx] = sum2; tmp[2+idx] = sum3;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + 9*diag[i] + 9;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    idt  = 3*i;
    sum1 = tmp[idt]; sum2 = tmp[1+idt]; sum3 = tmp[2+idt];
    while (nz--) {
      idx   = 3*(*vi++);
      x1    = tmp[idx]; x2 = tmp[1+idx]; x3 = tmp[2+idx];
      sum1 -= v[0]*x1 + v[3]*x2 + v[6]*x3;
      sum2 -= v[1]*x1 + v[4]*x2 + v[7]*x3;
      sum3 -= v[2]*x1 + v[5]*x2 + v[8]*x3;
      v += 9;
    }
    idc = 3*(*c--);
    v   = aa + 9*diag[i];
    x[idc]   = tmp[idt]   = v[0]*sum1 + v[3]*sum2 + v[6]*sum3;
    x[1+idc] = tmp[1+idt] = v[1]*sum1 + v[4]*sum2 + v[7]*sum3;
    x[2+idc] = tmp[2+idt] = v[2]*sum1 + v[5]*sum2 + v[8]*sum3;
  }
  ierr = ISRestoreIndices(isrow,&rout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout); CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  PLogFlops(2*9*(a->nz) - 3*a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_2"
int MatSolve_SeqBAIJ_2(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ *)A->data;
  IS              iscol=a->col,isrow=a->row;
  int             *r,*c,ierr,i,n=a->mbs,*vi,*ai=a->i,*aj=a->j,nz,idx,idt,idc,*rout,*cout;
  int             *diag = a->diag;
  MatScalar       *aa=a->a,*v;
  Scalar          *x,*b,sum1,sum2,x1,x2,*tmp;

  PetscFunctionBegin;
  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  idx    = 2*(*r++); 
  tmp[0] = b[idx]; tmp[1] = b[1+idx];
  for ( i=1; i<n; i++ ) {
    v     = aa + 4*ai[i];
    vi    = aj + ai[i];
    nz    = diag[i] - ai[i];
    idx   = 2*(*r++); 
    sum1  = b[idx]; sum2 = b[1+idx];
    while (nz--) {
      idx   = 2*(*vi++);
      x1    = tmp[idx]; x2 = tmp[1+idx];
      sum1 -= v[0]*x1 + v[2]*x2;
      sum2 -= v[1]*x1 + v[3]*x2;
      v += 4;
    }
    idx = 2*i;
    tmp[idx] = sum1; tmp[1+idx] = sum2;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + 4*diag[i] + 4;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    idt  = 2*i;
    sum1 = tmp[idt]; sum2 = tmp[1+idt];
    while (nz--) {
      idx   = 2*(*vi++);
      x1    = tmp[idx]; x2 = tmp[1+idx];
      sum1 -= v[0]*x1 + v[2]*x2;
      sum2 -= v[1]*x1 + v[3]*x2;
      v += 4;
    }
    idc = 2*(*c--);
    v   = aa + 4*diag[i];
    x[idc]   = tmp[idt]   = v[0]*sum1 + v[2]*sum2;
    x[1+idc] = tmp[1+idt] = v[1]*sum1 + v[3]*sum2;
  }
  ierr = ISRestoreIndices(isrow,&rout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout); CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b); CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  PLogFlops(2*4*(a->nz) - 2*a->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqBAIJ_1"
int MatSolve_SeqBAIJ_1(Mat A,Vec bb,Vec xx)
{
  Mat_SeqBAIJ     *a=(Mat_SeqBAIJ *)A->data;
  IS              iscol=a->col,isrow=a->row;
  int             *r,*c,ierr,i,n=a->mbs,*vi,*ai=a->i,*aj=a->j,nz,*rout,*cout;
  int             *diag = a->diag;
  MatScalar       *aa=a->a,*v;
  Scalar          *x,*b,sum1,*tmp;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArray(bb,&b); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  for ( i=1; i<n; i++ ) {
    v     = aa + ai[i];
    vi    = aj + ai[i];
    nz    = diag[i] - ai[i];
    sum1  = b[*r++];
    while (nz--) {
      sum1 -= (*v++)*tmp[*vi++];
    }
    tmp[i] = sum1;
  }
  /* backward solve the upper triangular */
  for ( i=n-1; i>=0; i-- ){
    v    = aa + diag[i] + 1;
    vi   = aj + diag[i] + 1;
    nz   = ai[i+1] - diag[i] - 1;
    sum1 = tmp[i];
    while (nz--) {
      sum1 -= (*v++)*tmp[*vi++];
    }
    x[*c--] = tmp[i] = aa[diag[i]]*sum1;
  }

  ierr = ISRestoreIndices(isrow,&rout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout); CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b); CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x); CHKERRQ(ierr);
  PLogFlops(2*1*(a->nz) - a->n);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
/*
     This code is virtually identical to MatILUFactorSymbolic_SeqAIJ
   except that the data structure of Mat_SeqAIJ is slightly different.
   Not a good example of code reuse.
*/
extern int MatMissingDiag_SeqBAIJ(Mat);

#undef __FUNC__  
#define __FUNC__ "MatILUFactorSymbolic_SeqBAIJ"
int MatILUFactorSymbolic_SeqBAIJ(Mat A,IS isrow,IS iscol,MatILUInfo *info,Mat *fact)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data, *b;
  IS          isicol;
  int         *r,*ic, ierr, prow, n = a->mbs, *ai = a->i, *aj = a->j;
  int         *ainew,*ajnew, jmax,*fill, *xi, nz, *im,*ajfill,*flev;
  int         *dloc, idx, row,m,fm, nzf, nzi,len,  realloc = 0, dcount = 0;
  int         incrlev,nnz,i,bs = a->bs,bs2 = a->bs2, levels, diagonal_fill;
  PetscTruth  col_identity, row_identity;
  double      f;

  PetscFunctionBegin;
  if (info) {
    f             = info->fill;
    levels        = (int) info->levels;
    diagonal_fill = (int) info->diagonal_fill;
  } else {
    f             = 1.0;
    levels        = 0;
    diagonal_fill = 0;
  }
  ierr = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);

  /* special case that simply copies fill pattern */
  PetscValidHeaderSpecific(isrow,IS_COOKIE);
  PetscValidHeaderSpecific(iscol,IS_COOKIE);
  ierr = ISIdentity(isrow,&row_identity); CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity); CHKERRQ(ierr);
  if (levels == 0 && row_identity && col_identity) {
    ierr = MatDuplicate_SeqBAIJ(A,MAT_DO_NOT_COPY_VALUES,fact); CHKERRQ(ierr);
    (*fact)->factor = FACTOR_LU;
    b               = (Mat_SeqBAIJ *) (*fact)->data;
    if (!b->diag) {
      ierr = MatMarkDiag_SeqBAIJ(*fact); CHKERRQ(ierr);
    }
    ierr = MatMissingDiag_SeqBAIJ(*fact); CHKERRQ(ierr);
    b->row        = isrow;
    b->col        = iscol;
    b->icol       = isicol;
    b->solve_work = (Scalar *) PetscMalloc((b->m+1+b->bs)*sizeof(Scalar));CHKPTRQ(b->solve_work);
    /*
        Blocksize 4 and 5 a special faster solver for ILU(0) factorization 
        with natural ordering 
    */
    if (b->bs == 4) {
      (*fact)->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering;
      (*fact)->ops->solve           = MatSolve_SeqBAIJ_4_NaturalOrdering;
      PLogInfo(A,"MatILUFactorSymbolic_SeqBAIJ:Using special natural ordering factor and solve BS=4\n"); 
    } else if (b->bs == 5) {
      (*fact)->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering;
      (*fact)->ops->solve           = MatSolve_SeqBAIJ_5_NaturalOrdering;
      PLogInfo( A,"MatILUFactorSymbolic_SeqBAIJ:Using special natural ordering factor and solve BS=5\n"); 
    }
    PetscFunctionReturn(0);
  }

  ierr = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic); CHKERRQ(ierr);

  /* get new row pointers */
  ainew = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(ainew);
  ainew[0] = 0;
  /* don't know how many column pointers are needed so estimate */
  jmax = (int) (f*ai[n] + 1);
  ajnew = (int *) PetscMalloc( (jmax)*sizeof(int) ); CHKPTRQ(ajnew);
  /* ajfill is level of fill for each fill entry */
  ajfill = (int *) PetscMalloc( (jmax)*sizeof(int) ); CHKPTRQ(ajfill);
  /* fill is a linked list of nonzeros in active row */
  fill = (int *) PetscMalloc( (n+1)*sizeof(int)); CHKPTRQ(fill);
  /* im is level for each filled value */
  im = (int *) PetscMalloc( (n+1)*sizeof(int)); CHKPTRQ(im);
  /* dloc is location of diagonal in factor */
  dloc = (int *) PetscMalloc( (n+1)*sizeof(int)); CHKPTRQ(dloc);
  dloc[0]  = 0;
  for ( prow=0; prow<n; prow++ ) {

    /* copy prow into linked list */
    nzf        = nz  = ai[r[prow]+1] - ai[r[prow]];
    if (!nz) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,1,"Empty row in matrix");
    xi         = aj + ai[r[prow]];
    fill[n]    = n;
    fill[prow] = -1; /* marker for diagonal entry */
    while (nz--) {
      fm  = n;
      idx = ic[*xi++];
      do {
        m  = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m]   = idx;
      fill[idx] = fm;
      im[idx]   = 0;
    }

    /* make sure diagonal entry is included */
    if (diagonal_fill && fill[prow] == -1) {
      fm = n;
      while (fill[fm] < prow) fm = fill[fm];
      fill[prow] = fill[fm];  /* insert diagonal into linked list */
      fill[fm]   = prow;
      im[prow]   = 0;
      nzf++;
      dcount++;
    }

    nzi = 0;
    row = fill[n];
    while ( row < prow ) {
      incrlev = im[row] + 1;
      nz      = dloc[row];
      xi      = ajnew  + ainew[row] + nz + 1;
      flev    = ajfill + ainew[row] + nz + 1;
      nnz     = ainew[row+1] - ainew[row] - nz - 1;
      fm      = row;
      while (nnz-- > 0) {
        idx = *xi++;
        if (*flev + incrlev > levels) {
          flev++;
          continue;
        }
        do {
          m  = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          im[idx]   = *flev + incrlev;
          fill[m]   = idx;
          fill[idx] = fm;
          fm        = idx;
          nzf++;
        } else {
          if (im[idx] > *flev + incrlev) im[idx] = *flev+incrlev;
        }
        flev++;
      }
      row = fill[row];
      nzi++;
    }
    /* copy new filled row into permanent storage */
    ainew[prow+1] = ainew[prow] + nzf;
    if (ainew[prow+1] > jmax) {

      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      int maxadd = jmax;
      /* maxadd = (int) (((f*ai[n]+1)*(n-prow+5))/n); */
      if (maxadd < nzf) maxadd = (n-prow)*(nzf+1);
      jmax += maxadd;

      /* allocate a longer ajnew and ajfill */
      xi = (int *) PetscMalloc( jmax*sizeof(int) );CHKPTRQ(xi);
      PetscMemcpy(xi,ajnew,ainew[prow]*sizeof(int));
      PetscFree(ajnew);
      ajnew = xi;
      xi = (int *) PetscMalloc( jmax*sizeof(int) );CHKPTRQ(xi);
      PetscMemcpy(xi,ajfill,ainew[prow]*sizeof(int));
      PetscFree(ajfill);
      ajfill = xi;
      realloc++; /* count how many reallocations are needed */
    }
    xi          = ajnew + ainew[prow];
    flev        = ajfill + ainew[prow];
    dloc[prow]  = nzi;
    fm          = fill[n];
    while (nzf--) {
      *xi++   = fm;
      *flev++ = im[fm];
      fm      = fill[fm];
    }
    /* make sure row has diagonal entry */
    if (ajnew[ainew[prow]+dloc[prow]] != prow) {
      SETERRQ1(PETSC_ERR_MAT_LU_ZRPVT,1,"Row %d has missing diagonal in factored matrix\n\
    try running with -pc_ilu_nonzeros_along_diagonal or -pc_ilu_diagonal_fill",prow);
    }
  }
  PetscFree(ajfill); 
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  PetscFree(fill); PetscFree(im);

  {
    double af = ((double)ainew[n])/((double)ai[n]);
    PLogInfo(A,"MatILUFactorSymbolic_SeqBAIJ:Reallocs %d Fill ratio:given %g needed %g\n",
             realloc,f,af);
    PLogInfo(A,"MatILUFactorSymbolic_SeqBAIJ:Run with -pc_ilu_fill %g or use \n",af);
    PLogInfo(A,"MatILUFactorSymbolic_SeqBAIJ:PCILUSetFill(pc,%g);\n",af);
    PLogInfo(A,"MatILUFactorSymbolic_SeqBAIJ:for best performance.\n");
    if (diagonal_fill) {
      PLogInfo(A,"MatILUFactorSymbolic_SeqBAIJ:Detected and replace %d missing diagonals",dcount);
    }
  }

  /* put together the new matrix */
  ierr = MatCreateSeqBAIJ(A->comm,bs,bs*n,bs*n,0,PETSC_NULL,fact);CHKERRQ(ierr);
  PLogObjectParent(*fact,isicol);
  b = (Mat_SeqBAIJ *) (*fact)->data;
  PetscFree(b->imax);
  b->singlemalloc = 0;
  len = bs2*ainew[n]*sizeof(MatScalar);
  /* the next line frees the default space generated by the Create() */
  PetscFree(b->a); PetscFree(b->ilen);
  b->a          = (MatScalar *) PetscMalloc( len ); CHKPTRQ(b->a);
  b->j          = ajnew;
  b->i          = ainew;
  for ( i=0; i<n; i++ ) dloc[i] += ainew[i];
  b->diag       = dloc;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  b->icol       = isicol;
  b->solve_work = (Scalar *) PetscMalloc( (bs*n+bs)*sizeof(Scalar));CHKPTRQ(b->solve_work);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate dloc, solve_work, new a, new j */
  PLogObjectMemory(*fact,(ainew[n]-n)*(sizeof(int))+bs2*ainew[n]*sizeof(Scalar));
  b->maxnz          = b->nz = ainew[n];
  (*fact)->factor   = FACTOR_LU;

  (*fact)->info.factor_mallocs    = realloc;
  (*fact)->info.fill_ratio_given  = f;
  (*fact)->info.fill_ratio_needed = ((double)ainew[n])/((double)ai[prow]);

  PetscFunctionReturn(0); 
}




