#ifndef lint
static char vcid[] = "$Id: bdfact.c,v 1.29 1996/04/29 02:30:15 bsmith Exp bsmith $";
#endif

/* Block diagonal matrix format - factorization and triangular solves */

#include "bdiag.h"
#include "src/inline/ilu.h"
#include "pinclude/plapack.h"

/* 
   BlockMatMult_Private - Computes C -= A*B, where
       A is nrow X nrow,
       B and C are nrow X ncol,
       All matrices are dense, stored by columns, where nrow is the 
           number of allocated rows for each matrix.
 */

#define BMatMult(nrow,ncol,A,B,C) BlockMatMult_Private(nrow,ncol,A,B,C)
static int BlockMatMult_Private(int nrow,int ncol,Scalar *A,Scalar *B,Scalar *C)
{
  Scalar B_i, *Apt;
  int    i, j, k, jnr;

  if (ncol == 1) {
    Apt = A;
    for (i=0; i<nrow; i++) {
      B_i = B[i];
      for (k=0; k<nrow; k++) C[k] -= Apt[k] * B_i;
      Apt += nrow;
    }
  } else {
    for (j=0; j<ncol; j++) {
      Apt = A;
      jnr = j*nrow;
      for (i=0; i<nrow; i++) {
        B_i = B[i+jnr];
        for (k=0; k<nrow; k++) C[k+jnr] -= Apt[k] * B_i;
        Apt += nrow;
      }
    }
  }
  PLogFlops(2*nrow*nrow*ncol);
  return 0;
}

int MatLUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,double f,Mat *B)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  if (a->m != a->n) SETERRQ(1,"MatLUFactorSymbolic_SeqBDiag:Matrix must be square");
  if (isrow || iscol)
    SETERRQ(1,"MatLUFactorSymbolic_SeqBDiag:permutations not supported");
  SETERRQ(1,"MatLUFactorSymbolic_SeqBDiag:Not written");
}

int MatILUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,double f,
                                  int levels,Mat *B)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  if (a->m != a->n) SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Matrix must be square");
  if (isrow || iscol)
    SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:permutations not supported");
  if (levels != 0)
    SETERRQ(1,"MatLUFactorSymbolic_SeqBDiag:Only ILU(0) is supported");
  return MatConvert(A,MATSAME,B);
}

int MatLUFactor_SeqBDiag(Mat A,IS isrow,IS iscol,double f)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  /* For now, no fill is allocated in symbolic factorization phase, so we
     directly use the input matrix for numeric factorization. */
  if (a->m != a->n) SETERRQ(1,"MatLUFactor_SeqBDiag:Matrix must be square");
  if (isrow || iscol) PLogInfo(A,
    "MatLUFactor_SeqBDiag: row and col permutations not supported\n");
  PLogInfo(A,"MatLUFactor_SeqBDiag:Only ILU(0) is supported\n");
  return MatLUFactorNumeric(A,&A);
}

int MatILUFactor_SeqBDiag(Mat A,IS isrow,IS iscol,double f,int level)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  /* For now, no fill is allocated in symbolic factorization phase, so we
     directly use the input matrix for numeric factorization. */
  if (a->m != a->n) SETERRQ(1,"MatILUFactor_SeqBDiag:Matrix must be square");
  if (isrow || iscol) PLogInfo(A,
    "MatILUFactor_SeqBDiag: row and col permutations not supported\n");
  if (level != 0)
    PLogInfo(A,"MatILUFactor_SeqBDiag:Only ILU(0) is supported\n");
  return MatLUFactorNumeric(A,&A);
}

/* --------------------------------------------------------------------------*/
int MatLUFactorNumeric_SeqBDiag_N(Mat A,Mat *B)
{
  Mat          C = *B;
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) C->data;
  int          k, d, d2, dgk, elim_row, elim_col, nb = a->nb, knb, knb2, nb2;
  int          dnum,nd = a->nd, mblock = a->mblock, nblock = a->nblock,ierr;
  int          *diag = a->diag,  m = a->m, mainbd = a->mainbd, *dgptr;
  Scalar       **dv = a->diagv, *dd = dv[mainbd],  *v_work;
  Scalar       *multiplier;

  /* Notes: 
      - We're using only B in this routine (A remains untouched).
      - The nb>1 case performs block LU, which is functionally the same as
        the nb=1 case, except that we use factorization, triangular solve,
        and matrix-matrix products within the dense subblocks.
      - Pivoting is not employed for the case nb=1; for the case nb>1
        pivoting is used only within the dense subblocks. 
   */
  if (!a->pivot) {
    a->pivot = (int *) PetscMalloc(m*sizeof(int)); CHKPTRQ(a->pivot);
    PLogObjectMemory(C,m*sizeof(int));
  }
  v_work = (Scalar *) PetscMalloc((nb*nb+nb)*sizeof(Scalar)); CHKPTRQ(v_work);
  multiplier = v_work + nb;
  nb2 = nb*nb;
  dgptr = (int *) PetscMalloc((mblock+nblock)*sizeof(int)); CHKPTRQ(dgptr);
  PetscMemzero(dgptr,(mblock+nblock)*sizeof(int));
  for ( k=0; k<nd; k++ ) dgptr[diag[k]+mblock] = k+1;
  for ( k=0; k<mblock; k++ ) { /* k = block pivot_row */
    knb = k*nb; knb2 = knb*nb;
    /* invert the diagonal block */
    Kernel_A_gets_inverse_A(nb,dd+knb2,a->pivot+knb,v_work);
    for ( d=mainbd-1; d>=0; d-- ) {
      elim_row = k + diag[d];
      if (elim_row < mblock) { /* sweep down */
        /* dv[d][knb2]: test if entire block is zero? */
        Kernel_A_gets_A_times_B(nb,&dv[d][elim_row*nb2],dd+knb2,multiplier); 
        for ( d2=d+1; d2<nd; d2++ ) {
          elim_col = elim_row - diag[d2];
          if (elim_col >=0 && elim_col < nblock) {
            dgk = k - elim_col;
            if ((dnum = dgptr[dgk+mblock])) {
              BMatMult(nb,nb,&dv[d][elim_row*nb2],
                              &dv[dnum-1][knb2],&dv[d2][elim_row*nb2]);
            }
          }
        }
      }
    }
  }
  PetscFree(dgptr);
  C->factor = FACTOR_LU;
  return 0;
}

int MatLUFactorNumeric_SeqBDiag_1(Mat A,Mat *B)
{
  Mat          C = *B;
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) C->data;
  int          k, d, d2, dgk, elim_row, elim_col;
  int          dnum,nd = a->nd;
  int          *diag = a->diag, n = a->n, m = a->m, mainbd = a->mainbd, *dgptr;
  Scalar       **dv = a->diagv, *dd = dv[mainbd], mult;

  /* Notes: 
      - We're using only B in this routine (A remains untouched).
      - The nb>1 case performs block LU, which is functionally the same as
        the nb=1 case, except that we use factorization, triangular solve,
        and matrix-matrix products within the dense subblocks.
      - Pivoting is not employed for the case nb=1; for the case nb>1
        pivoting is used only within the dense subblocks. 
   */
  dgptr = (int *) PetscMalloc((m+n)*sizeof(int)); CHKPTRQ(dgptr);
  PetscMemzero(dgptr,(m+n)*sizeof(int));
  for ( k=0; k<nd; k++ ) dgptr[diag[k]+m] = k+1;
  for ( k=0; k<m; k++ ) { /* k = pivot_row */
    dd[k] = 1.0/dd[k];
    for ( d=mainbd-1; d>=0; d-- ) {
      elim_row = k + diag[d];
      if (elim_row < m) { /* sweep down */
        if (dv[d][elim_row] != 0) {
          dv[d][elim_row] *= dd[k];
          mult = dv[d][elim_row];
          for ( d2=d+1; d2<nd; d2++ ) {
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
  PetscFree(dgptr);
  C->factor = FACTOR_LU;
  return 0;
}

/* -----------------------------------------------------------------*/

int MatSolve_SeqBDiag_1(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, d, loc, ierr, mainbd = a->mainbd;
  int          n = a->n, m = a->m, *diag = a->diag, col;
  Scalar       *x, *y, *dd = a->diagv[mainbd], sum, **dv = a->diagv;

  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y); CHKERRQ(ierr);
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
  for ( i=m-1; i>=0; i-- ) {
    sum = y[i];
    for (d=mainbd+1; d<a->nd; d++) {
      col = i - diag[d];
      if (col < n) sum -= dv[d][i] * y[col];
    }
    y[i] = sum*dd[i];
  }
  PLogFlops(2*a->nz - a->n);
  return 0;
}

int MatSolve_SeqBDiag_N(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, d, loc, ierr, mainbd = a->mainbd;
  int          mblock = a->mblock, nblock = a->nblock, inb, inb2,_One = 1;
  int          nb = a->nb, m = a->m, *diag = a->diag, col;
  Scalar       *x, *y, *dd = a->diagv[mainbd], **dv = a->diagv;
  Scalar       work[25],_DZero = 0.0,_DOne = 1.0;

  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y); CHKERRQ(ierr);
  if (nb > 25) SETERRQ(1,"Blocks must be smaller then 25");
  PetscMemcpy(y,x,m*sizeof(Scalar));

  /* forward solve the lower triangular part */
  if (mainbd != 0) {
    for (i=0; i<mblock; i++) {
      inb = i*nb;
      for (d=0; d<mainbd; d++) {
        loc = i - diag[d];
        if (loc >= 0) BMatMult(nb,1,&dv[d][i*nb*nb],&y[loc*nb],&y[inb]);
      }
    }
  }
  /* backward solve the upper triangular part */
  for ( i=mblock-1; i>=0; i-- ) {
    inb = i*nb; inb2 = inb*nb;
    for (d=mainbd+1; d<a->nd; d++) {
      col = i - diag[d];
      if (col < nblock) BMatMult(nb,1,&dv[d][inb2],&y[col*nb],&y[inb]);
    }
    LAgemv_("N",&nb,&nb,&_DOne,&dd[inb2],&nb,&y[inb],&_One,&_DZero,
               work,&_One);
    PetscMemcpy(&(y[inb]),work,nb*sizeof(Scalar));
  }
  return 0;
}

