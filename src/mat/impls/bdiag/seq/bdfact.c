
/* Block diagonal matrix format - factorization and triangular solves */

#include "src/mat/impls/bdiag/seq/bdiag.h"
#include "src/inline/ilu.h"

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_SeqBDiag"
PetscErrorCode MatILUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,MatFactorInfo *info,Mat *B)
{
  PetscTruth     idn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->m != A->n) SETERRQ(PETSC_ERR_SUP,"Matrix must be square");
  if (isrow) {
    ierr = ISIdentity(isrow,&idn);CHKERRQ(ierr);
    if (!idn) SETERRQ(PETSC_ERR_SUP,"Only identity row permutation supported");
  }
  if (iscol) {
    ierr = ISIdentity(iscol,&idn);CHKERRQ(ierr);
    if (!idn) SETERRQ(PETSC_ERR_SUP,"Only identity column permutation supported");
  }
  if (info->levels != 0) {
    SETERRQ(PETSC_ERR_SUP,"Only ILU(0) is supported");
  }
  ierr = MatConvert(A,MATSAME,B);CHKERRQ(ierr);

  /* Must set to zero for repeated calls with different nonzero structure */
  (*B)->factor = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactor_SeqBDiag"
PetscErrorCode MatILUFactor_SeqBDiag(Mat A,IS isrow,IS iscol,MatFactorInfo *info)
{
  PetscTruth     idn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* For now, no fill is allocated in symbolic factorization phase, so we
     directly use the input matrix for numeric factorization. */
  if (A->m != A->n) SETERRQ(PETSC_ERR_SUP,"Matrix must be square");
  if (isrow) {
    ierr = ISIdentity(isrow,&idn);CHKERRQ(ierr);
    if (!idn) SETERRQ(PETSC_ERR_SUP,"Only identity row permutation supported");
  }
  if (iscol) {
    ierr = ISIdentity(iscol,&idn);CHKERRQ(ierr);
    if (!idn) SETERRQ(PETSC_ERR_SUP,"Only identity column permutation supported");
  }
  if (info->levels != 0) SETERRQ(PETSC_ERR_SUP,"Only ILU(0) is supported");
  ierr = MatLUFactorNumeric(A,info,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBDiag_N"
PetscErrorCode MatLUFactorNumeric_SeqBDiag_N(Mat A,MatFactorInfo *info,Mat *B)
{
  Mat            C = *B;
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)C->data,*a1 = (Mat_SeqBDiag*)A->data;
  PetscInt       k,d,d2,dgk,elim_row,elim_col,bs = A->bs,knb,knb2,bs2 = bs*bs;
  PetscErrorCode ierr;
  PetscInt       dnum,nd = a->nd,mblock = a->mblock,nblock = a->nblock;
  PetscInt       *diag = a->diag, m = A->m,mainbd = a->mainbd,*dgptr,len,i;
  PetscScalar    **dv = a->diagv,*dd = dv[mainbd],*v_work;
  PetscScalar    *multiplier;

  PetscFunctionBegin;
  /* Copy input matrix to factored matrix if we've already factored the
     matrix before AND the nonzero structure remains the same.  This is done
     in symbolic factorization the first time through, but there's no symbolic
     factorization for successive calls with same matrix sparsity structure. */
  if (C->factor == FACTOR_LU) {
    for (i=0; i<a->nd; i++) {
      len = a->bdlen[i] * bs2 * sizeof(PetscScalar);
      d   = diag[i];
      if (d > 0) {
        ierr = PetscMemcpy(dv[i]+bs2*d,a1->diagv[i]+bs2*d,len);CHKERRQ(ierr);
      } else {
        ierr = PetscMemcpy(dv[i],a1->diagv[i],len);CHKERRQ(ierr);
      }
    }
  }

  if (!a->pivot) {
    ierr = PetscMalloc((m+1)*sizeof(PetscInt),&a->pivot);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(C,m*sizeof(PetscInt));CHKERRQ(ierr);
  }
  ierr       = PetscMalloc((bs2+bs+1)*sizeof(PetscScalar),&v_work);CHKERRQ(ierr);
  multiplier = v_work + bs;
  ierr       = PetscMalloc((mblock+nblock+1)*sizeof(PetscInt),&dgptr);CHKERRQ(ierr);
  ierr       = PetscMemzero(dgptr,(mblock+nblock)*sizeof(PetscInt));CHKERRQ(ierr);
  for (k=0; k<nd; k++) dgptr[diag[k]+mblock] = k+1;
  for (k=0; k<mblock; k++) { /* k = block pivot_row */
    knb = k*bs; knb2 = knb*bs;
    /* invert the diagonal block */
    ierr = Kernel_A_gets_inverse_A(bs,dd+knb2,a->pivot+knb,v_work);CHKERRQ(ierr);
    for (d=mainbd-1; d>=0; d--) {
      elim_row = k + diag[d];
      if (elim_row < mblock) { /* sweep down */
        /* dv[d][knb2]: test if entire block is zero? */
        Kernel_A_gets_A_times_B(bs,&dv[d][elim_row*bs2],dd+knb2,multiplier); 
        for (d2=d+1; d2<nd; d2++) {
          elim_col = elim_row - diag[d2];
          if (elim_col >=0 && elim_col < nblock) {
            dgk = k - elim_col;
            if ((dnum = dgptr[dgk+mblock])) {
              Kernel_A_gets_A_minus_B_times_C(bs,&dv[d2][elim_row*bs2],
                             &dv[d][elim_row*bs2],&dv[dnum-1][knb2]);
            }
          }
        }
      }
    }
  }
  ierr = PetscFree(dgptr);CHKERRQ(ierr);
  ierr = PetscFree(v_work);CHKERRQ(ierr);
  if (!a->solvework) {
    ierr = PetscMalloc(bs*sizeof(PetscScalar),&a->solvework);CHKERRQ(ierr);
  }
  C->factor = FACTOR_LU;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqBDiag_1"
PetscErrorCode MatLUFactorNumeric_SeqBDiag_1(Mat A,MatFactorInfo *info,Mat *B)
{
  Mat            C = *B;
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)C->data,*a1 = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       k,d,d2,dgk,elim_row,elim_col,dnum,nd = a->nd,i,len;
  PetscInt       *diag = a->diag,n = A->n,m = A->m,mainbd = a->mainbd,*dgptr;
  PetscScalar    **dv = a->diagv,*dd = dv[mainbd],mult;

  PetscFunctionBegin;
  /* Copy input matrix to factored matrix if we've already factored the
     matrix before AND the nonzero structure remains the same.  This is done
     in symbolic factorization the first time through, but there's no symbolic
     factorization for successive calls with same matrix sparsity structure. */
  if (C->factor == FACTOR_LU) {
    for (i=0; i<nd; i++) {
      len = a->bdlen[i] * sizeof(PetscScalar);
      d   = diag[i];
      if (d > 0) {
        ierr = PetscMemcpy(dv[i]+d,a1->diagv[i]+d,len);CHKERRQ(ierr);
      } else {
        ierr = PetscMemcpy(dv[i],a1->diagv[i],len);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscMalloc((m+n+1)*sizeof(PetscInt),&dgptr);CHKERRQ(ierr);
  ierr  = PetscMemzero(dgptr,(m+n)*sizeof(PetscInt));CHKERRQ(ierr);
  for (k=0; k<nd; k++) dgptr[diag[k]+m] = k+1;
  for (k=0; k<m; k++) { /* k = pivot_row */
    dd[k] = 1.0/dd[k];
    for (d=mainbd-1; d>=0; d--) {
      elim_row = k + diag[d];
      if (elim_row < m) { /* sweep down */
        if (dv[d][elim_row] != 0.0) {
          dv[d][elim_row] *= dd[k];
          mult = dv[d][elim_row];
          for (d2=d+1; d2<nd; d2++) {
            elim_col = elim_row - diag[d2];
            dgk = k - elim_col;
            if (elim_col >=0 && elim_col < n) {
              if ((dnum = dgptr[dgk+m])) {
                dv[d2][elim_row] -= mult * dv[dnum-1][k];
              }
            }
          }
        }
      }
    }
  }
  ierr = PetscFree(dgptr);CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqBDiag_1"
PetscErrorCode MatSolve_SeqBDiag_1(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,d,loc,mainbd = a->mainbd;
  PetscInt       n = A->n,m = A->m,*diag = a->diag,col;
  PetscScalar    *x,*y,*dd = a->diagv[mainbd],sum,**dv = a->diagv;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  /* forward solve the lower triangular part */
  for (i=0; i<m; i++) {
    sum = x[i];
    for (d=0; d<mainbd; d++) {
      loc = i - diag[d];
      if (loc >= 0) sum -= dv[d][i] * y[loc];
    }
    y[i] = sum;
  }
  /* backward solve the upper triangular part */
  for (i=m-1; i>=0; i--) {
    sum = y[i];
    for (d=mainbd+1; d<a->nd; d++) {
      col = i - diag[d];
      if (col < n) sum -= dv[d][i] * y[col];
    }
    y[i] = sum*dd[i];
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz - A->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqBDiag_2"
PetscErrorCode MatSolve_SeqBDiag_2(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       i,d,loc,mainbd = a->mainbd;
  PetscInt       mblock = a->mblock,nblock = a->nblock,inb,inb2;
  PetscErrorCode ierr;
  PetscInt       m = A->m,*diag = a->diag,col;
  PetscScalar    *x,*y,*dd = a->diagv[mainbd],**dv = a->diagv,*dvt;
  PetscScalar    w0,w1,sum0,sum1;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,m*sizeof(PetscScalar));CHKERRQ(ierr);

  /* forward solve the lower triangular part */
  if (mainbd != 0) {
    inb = 0;
    for (i=0; i<mblock; i++) {
      sum0 = sum1 = 0.0;
      for (d=0; d<mainbd; d++) {
        loc = 2*(i - diag[d]);
        if (loc >= 0) {
          dvt = &dv[d][4*i]; 
          w0 = y[loc]; w1 = y[loc+1];
          sum0 += dvt[0]*w0 + dvt[2]*w1;
          sum1 += dvt[1]*w0 + dvt[3]*w1;
        }
      }
      y[inb] -= sum0; y[inb+1] -= sum1; 

      inb += 2;
    }
  }
  /* backward solve the upper triangular part */
  inb = 2*(mblock-1); inb2 = 2*inb;
  for (i=mblock-1; i>=0; i--) {
    sum0 = y[inb]; sum1 = y[inb+1];
    for (d=mainbd+1; d<a->nd; d++) {
      col = 2*(i - diag[d]);
      if (col < 2*nblock) {
        dvt = &dv[d][4*i]; 
        w0 = y[col]; w1 = y[col+1];
        sum0 -= dvt[0]*w0 + dvt[2]*w1;
        sum1 -= dvt[1]*w0 + dvt[3]*w1;
      }
    }
    dvt = dd+inb2;
    y[inb]   = dvt[0]*sum0 + dvt[2]*sum1;
    y[inb+1] = dvt[1]*sum0 + dvt[3]*sum1;
    inb -= 2; inb2 -= 4;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz - A->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqBDiag_3"
PetscErrorCode MatSolve_SeqBDiag_3(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       i,d,loc,mainbd = a->mainbd;
  PetscInt       mblock = a->mblock,nblock = a->nblock,inb,inb2;
  PetscErrorCode ierr;
  PetscInt       m = A->m,*diag = a->diag,col;
  PetscScalar    *x,*y,*dd = a->diagv[mainbd],**dv = a->diagv,*dvt;
  PetscScalar    w0,w1,w2,sum0,sum1,sum2;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,m*sizeof(PetscScalar));CHKERRQ(ierr);

  /* forward solve the lower triangular part */
  if (mainbd != 0) {
    inb = 0;
    for (i=0; i<mblock; i++) {
      sum0 = sum1 = sum2 = 0.0;
      for (d=0; d<mainbd; d++) {
        loc = 3*(i - diag[d]);
        if (loc >= 0) {
          dvt = &dv[d][9*i]; 
          w0 = y[loc]; w1 = y[loc+1]; w2 = y[loc+2];
          sum0 += dvt[0]*w0 + dvt[3]*w1 + dvt[6]*w2;
          sum1 += dvt[1]*w0 + dvt[4]*w1 + dvt[7]*w2;
          sum2 += dvt[2]*w0 + dvt[5]*w1 + dvt[8]*w2;
        }
      }
      y[inb] -= sum0; y[inb+1] -= sum1; y[inb+2] -= sum2;
      inb += 3;
    }
  }
  /* backward solve the upper triangular part */
  inb = 3*(mblock-1); inb2 = 3*inb;
  for (i=mblock-1; i>=0; i--) {
    sum0 = y[inb]; sum1 = y[inb+1]; sum2 =  y[inb+2];
    for (d=mainbd+1; d<a->nd; d++) {
      col = 3*(i - diag[d]);
      if (col < 3*nblock) {
        dvt = &dv[d][9*i]; 
        w0 = y[col]; w1 = y[col+1];w2 = y[col+2];
        sum0 -= dvt[0]*w0 + dvt[3]*w1 + dvt[6]*w2;
        sum1 -= dvt[1]*w0 + dvt[4]*w1 + dvt[7]*w2;
        sum2 -= dvt[2]*w0 + dvt[5]*w1 + dvt[8]*w2;
      }
    }
    dvt = dd+inb2;
    y[inb]   = dvt[0]*sum0 + dvt[3]*sum1 + dvt[6]*sum2;
    y[inb+1] = dvt[1]*sum0 + dvt[4]*sum1 + dvt[7]*sum2;
    y[inb+2] = dvt[2]*sum0 + dvt[5]*sum1 + dvt[8]*sum2;
    inb -= 3; inb2 -= 9;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz - A->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqBDiag_4"
PetscErrorCode MatSolve_SeqBDiag_4(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       i,d,loc,mainbd = a->mainbd;
  PetscInt       mblock = a->mblock,nblock = a->nblock,inb,inb2;
  PetscErrorCode ierr;
  PetscInt       m = A->m,*diag = a->diag,col;
  PetscScalar    *x,*y,*dd = a->diagv[mainbd],**dv = a->diagv,*dvt;
  PetscScalar    w0,w1,w2,w3,sum0,sum1,sum2,sum3;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,m*sizeof(PetscScalar));CHKERRQ(ierr);

  /* forward solve the lower triangular part */
  if (mainbd != 0) {
    inb = 0;
    for (i=0; i<mblock; i++) {
      sum0 = sum1 = sum2 = sum3 = 0.0;
      for (d=0; d<mainbd; d++) {
        loc = 4*(i - diag[d]);
        if (loc >= 0) {
          dvt = &dv[d][16*i]; 
          w0 = y[loc]; w1 = y[loc+1]; w2 = y[loc+2];w3 = y[loc+3];
          sum0 += dvt[0]*w0 + dvt[4]*w1 + dvt[8]*w2  + dvt[12]*w3;
          sum1 += dvt[1]*w0 + dvt[5]*w1 + dvt[9]*w2  + dvt[13]*w3;
          sum2 += dvt[2]*w0 + dvt[6]*w1 + dvt[10]*w2 + dvt[14]*w3;
          sum3 += dvt[3]*w0 + dvt[7]*w1 + dvt[11]*w2 + dvt[15]*w3;
        }
      }
      y[inb] -= sum0; y[inb+1] -= sum1; y[inb+2] -= sum2;y[inb+3] -= sum3; 
      inb += 4;
    }
  }
  /* backward solve the upper triangular part */
  inb = 4*(mblock-1); inb2 = 4*inb;
  for (i=mblock-1; i>=0; i--) {
    sum0 = y[inb]; sum1 = y[inb+1]; sum2 =  y[inb+2]; sum3 =  y[inb+3];
    for (d=mainbd+1; d<a->nd; d++) {
      col = 4*(i - diag[d]);
      if (col < 4*nblock) {
        dvt = &dv[d][16*i]; 
        w0 = y[col]; w1 = y[col+1];w2 = y[col+2];w3 = y[col+3];
        sum0 -= dvt[0]*w0 + dvt[4]*w1 + dvt[8]*w2  + dvt[12]*w3;
        sum1 -= dvt[1]*w0 + dvt[5]*w1 + dvt[9]*w2  + dvt[13]*w3;
        sum2 -= dvt[2]*w0 + dvt[6]*w1 + dvt[10]*w2 + dvt[14]*w3;
        sum3 -= dvt[3]*w0 + dvt[7]*w1 + dvt[11]*w2 + dvt[15]*w3;
      }
    }
    dvt = dd+inb2;
    y[inb]   = dvt[0]*sum0 + dvt[4]*sum1 + dvt[8]*sum2 + dvt[12]*sum3;
    y[inb+1] = dvt[1]*sum0 + dvt[5]*sum1 + dvt[9]*sum2 + dvt[13]*sum3;
    y[inb+2] = dvt[2]*sum0 + dvt[6]*sum1 + dvt[10]*sum2 + dvt[14]*sum3;
    y[inb+3] = dvt[3]*sum0 + dvt[7]*sum1 + dvt[11]*sum2 + dvt[15]*sum3;
    inb -= 4; inb2 -= 16;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz - A->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqBDiag_5"
PetscErrorCode MatSolve_SeqBDiag_5(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       i,d,loc,mainbd = a->mainbd;
  PetscInt       mblock = a->mblock,nblock = a->nblock,inb,inb2;
  PetscErrorCode ierr;
  PetscInt       m = A->m,*diag = a->diag,col;
  PetscScalar    *x,*y,*dd = a->diagv[mainbd],**dv = a->diagv,*dvt;
  PetscScalar    w0,w1,w2,w3,w4,sum0,sum1,sum2,sum3,sum4;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,m*sizeof(PetscScalar));CHKERRQ(ierr);

  /* forward solve the lower triangular part */
  if (mainbd != 0) {
    inb = 0;
    for (i=0; i<mblock; i++) {
      sum0 = sum1 = sum2 = sum3 = sum4 = 0.0;
      for (d=0; d<mainbd; d++) {
        loc = 5*(i - diag[d]);
        if (loc >= 0) {
          dvt = &dv[d][25*i]; 
          w0 = y[loc]; w1 = y[loc+1]; w2 = y[loc+2];w3 = y[loc+3];w4 = y[loc+4];
          sum0 += dvt[0]*w0 + dvt[5]*w1 + dvt[10]*w2 + dvt[15]*w3 + dvt[20]*w4;
          sum1 += dvt[1]*w0 + dvt[6]*w1 + dvt[11]*w2 + dvt[16]*w3 + dvt[21]*w4;
          sum2 += dvt[2]*w0 + dvt[7]*w1 + dvt[12]*w2 + dvt[17]*w3 + dvt[22]*w4;
          sum3 += dvt[3]*w0 + dvt[8]*w1 + dvt[13]*w2 + dvt[18]*w3 + dvt[23]*w4;
          sum4 += dvt[4]*w0 + dvt[9]*w1 + dvt[14]*w2 + dvt[19]*w3 + dvt[24]*w4;
        }
      }
      y[inb]   -= sum0; y[inb+1] -= sum1; y[inb+2] -= sum2;y[inb+3] -= sum3; 
      y[inb+4] -= sum4;
      inb += 5;
    }
  }
  /* backward solve the upper triangular part */
  inb = 5*(mblock-1); inb2 = 5*inb;
  for (i=mblock-1; i>=0; i--) {
    sum0 = y[inb];sum1 = y[inb+1];sum2 = y[inb+2];sum3 = y[inb+3];sum4 = y[inb+4];
    for (d=mainbd+1; d<a->nd; d++) {
      col = 5*(i - diag[d]);
      if (col < 5*nblock) {
        dvt = &dv[d][25*i]; 
        w0 = y[col]; w1 = y[col+1];w2 = y[col+2];w3 = y[col+3];w4 = y[col+4];
        sum0 -= dvt[0]*w0 + dvt[5]*w1 + dvt[10]*w2 + dvt[15]*w3 + dvt[20]*w4;
        sum1 -= dvt[1]*w0 + dvt[6]*w1 + dvt[11]*w2 + dvt[16]*w3 + dvt[21]*w4;
        sum2 -= dvt[2]*w0 + dvt[7]*w1 + dvt[12]*w2 + dvt[17]*w3 + dvt[22]*w4;
        sum3 -= dvt[3]*w0 + dvt[8]*w1 + dvt[13]*w2 + dvt[18]*w3 + dvt[23]*w4;
        sum4 -= dvt[4]*w0 + dvt[9]*w1 + dvt[14]*w2 + dvt[19]*w3 + dvt[24]*w4;
      }
    }
    dvt = dd+inb2;
    y[inb]   = dvt[0]*sum0 + dvt[5]*sum1 + dvt[10]*sum2 + dvt[15]*sum3 
               + dvt[20]*sum4;
    y[inb+1] = dvt[1]*sum0 + dvt[6]*sum1 + dvt[11]*sum2 + dvt[16]*sum3 
               + dvt[21]*sum4;
    y[inb+2] = dvt[2]*sum0 + dvt[7]*sum1 + dvt[12]*sum2 + dvt[17]*sum3 
               + dvt[22]*sum4;
    y[inb+3] = dvt[3]*sum0 + dvt[8]*sum1 + dvt[13]*sum2 + dvt[18]*sum3 
               + dvt[23]*sum4;
    y[inb+4] = dvt[4]*sum0 + dvt[9]*sum1 + dvt[14]*sum2 + dvt[19]*sum3 
               + dvt[24]*sum4;
    inb -= 5; inb2 -= 25;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz - A->n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqBDiag_N"
PetscErrorCode MatSolve_SeqBDiag_N(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       i,d,loc,mainbd = a->mainbd;
  PetscInt       mblock = a->mblock,nblock = a->nblock,inb,inb2;
  PetscErrorCode ierr;
  PetscInt       bs = A->bs,m = A->m,*diag = a->diag,col,bs2 = bs*bs;
  PetscScalar    *x,*y,*dd = a->diagv[mainbd],**dv = a->diagv;
  PetscScalar    *work = a->solvework;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,m*sizeof(PetscScalar));CHKERRQ(ierr);

  /* forward solve the lower triangular part */
  if (mainbd != 0) {
    inb = 0;
    for (i=0; i<mblock; i++) {
      for (d=0; d<mainbd; d++) {
        loc = i - diag[d];
        if (loc >= 0) {
          Kernel_v_gets_v_minus_A_times_w(bs,y+inb,&dv[d][i*bs2],y+loc*bs);
        }
      }
      inb += bs;
    }
  }
  /* backward solve the upper triangular part */
  inb = bs*(mblock-1); inb2 = inb*bs;
  for (i=mblock-1; i>=0; i--) {
    for (d=mainbd+1; d<a->nd; d++) {
      col = i - diag[d];
      if (col < nblock) {
        Kernel_v_gets_v_minus_A_times_w(bs,y+inb,&dv[d][inb2],y+col*bs);
      }
    }
    Kernel_w_gets_A_times_v(bs,y+inb,dd+inb2,work);  
    ierr = PetscMemcpy(y+inb,work,bs*sizeof(PetscScalar));CHKERRQ(ierr);
    inb -= bs; inb2 -= bs2;
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscLogFlops(2*a->nz - A->n);
  PetscFunctionReturn(0);
}






