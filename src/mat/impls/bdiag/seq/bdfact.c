#ifndef lint
static char vcid[] = "$Id: bdfact.c,v 1.11 1995/10/06 13:48:16 curfman Exp curfman $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "pinclude/plapack.h"

/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/

int MatLUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,double f,Mat *fact)
{
  if (a->m != a->n) SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Matrix must be square");
  if (isrow || iscol) PLogInfo((PetscObject)A,
    "MatLUFactorSymbolic_SeqBDiag: Row and col permutations not supported.\n");
  PLogInfo((PetscObject)A,
    "MatLUFactorSymbolic_SeqBDiag: Currently no fill is computed!\n");
  return MatConvert(A,MATSAME,fact);
}

int MatILUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,double f,
                                  int levels,Mat *fact)
{
  if (a->m != a->n) SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Matrix must be square");
  if (isrow || iscol) PLogInfo((PetscObject)A,
    "MatILUFactorSymbolic_SeqBDiag: row and col permutations not supported.\n");
  if (levels != 0)
    SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Only ILU(0) is supported");
  return MatConvert(A,MATSAME,fact);
}

int MatLUFactorNumeric_SeqBDiag(Mat A,Mat *B)
{
  Mat          C = *B;
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) C->data;
  int          info, i, k, d, d2, dgk, elim_row, elim_col, nb = a->nb;
  int          dnum,  nd = a->nd;
  int          *diag = a->diag, n = a->n, m = a->m, mainbd = a->mainbd, *dgptr;
  Scalar       *submat, **dv = a->diagv, *dd = dv[mainbd], mult;

  if (nb == 1) {
    dgptr = (int *) PETSCMALLOC((m+n+1)*sizeof(int)); CHKPTRQ(dgptr);
    PetscZero(dgptr,(m+n+1)*sizeof(int));
    for ( i=0; i<nd; i++ ) dgptr[diag[i]+m] = i+1;
    for ( k=0; k<m; k++ ) { /* k = pivot_row */
      dd[k] = 1.0/dd[k];
      for ( d=mainbd-1; d>=0; d-- ) {
        elim_row = k + diag[d];
        if (elim_row < m) { /* sweep down */
          if (dv[d][k] != 0) {
            dv[d][k] *= dd[k];
            mult = dv[d][k];
            for ( d2=d+1; d2<nd; d2++ ) {
              elim_col = elim_row - diag[d2];
              if (elim_col >=0 && elim_col < n) {
                dgk = k - elim_col;
                if (dgk > 0) SETERRQ(1,
                   "MatLUFactorNumeric_SeqBDiag:bad elimination column");
                if (dnum = dgptr[dgk+m]) {
                  if (diag[d2] > 0) dv[d2][elim_col] -= mult * dv[dnum-1][k];
                  else              dv[d2][elim_row] -= mult * dv[dnum-1][k];
                }
              }
            }
          }
        }
      }
    }
    PETSCFREE(dgptr);
  } 
  else {
    if (a->nd != 1 || a->diag[0] !=0) SETERRQ(1,
      "MatLUFactorNumeric_SeqBDiag:factoriz only for main diagonal");
    if (!a->pivots) {
      a->pivots = (int *) PETSCMALLOC(a->m*sizeof(int)); CHKPTRQ(a->pivots);
    }
    submat = a->diagv[0];
    for (i=0; i<a->bdlen[0]; i++) {
      LAgetrf_(&nb,&nb,&submat[i*nb*nb],&nb,&a->pivots[i*nb],&info);
      if (info) SETERRQ(1,"MatLUFactorNumeric_SeqBDiag:Bad LU factorization");
    }
  }
  C->factor = FACTOR_LU;
  return 0;
}

int MatLUFactor_SeqBDiag(Mat A,IS row,IS col,double f)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, ierr;
  Mat          C;

  ierr = MatLUFactorSymbolic_SeqBDiag(A,row,col,f,&C); CHKERRQ(ierr);
  ierr = MatLUFactorNumeric_SeqBDiag(A,&C); CHKERRQ(ierr);

  /* free all the data structures from mat */
  if (!a->user_alloc) { /* Free the actual diagonals */
    for (i=0; i<a->nd; i++) PETSCFREE( a->diagv[i] );
  } /* What to do for user allocation of diags?? */
  if (a->pivots) PETSCFREE(a->pivots);
  PETSCFREE(a->diagv); PETSCFREE(a->diag);
  PETSCFREE(a->colloc); PETSCFREE(a->dvalue);
  PETSCFREE(a);
  PetscMemcpy(A,C,sizeof(struct _Mat));
  PETSCHEADERDESTROY(C);

  return 0;
}

int MatSolve_SeqBDiag(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          one = 1, info, i, d, loc, ierr, mainbd = a->mainbd;
  int          nb = a->nb, m = a->m;
  Scalar       *x, *y, *dd = a->diagv[mainbd], sum;
  Scalar       *submat;

  if (A->factor != FACTOR_LU) SETERRQ(1,"MatSolve_SeqBDiag:Not for unfactored matrix.");
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y); CHKERRQ(ierr);

  if (nb == 1) {
    for (i=0; i<m; i++) {
      sum = x[i];
      for (d=0; d<mainbd; d++) {
        loc = i - a->diag[d];
        if (loc >= 0) sum -= a->diagv[d][loc] * y[loc];
      }
      y[i] = sum*dd[i];
    }
  } else {
    if (a->nd != 1 || a->diag[0] !=0) SETERRQ(1,
      "MatSolve_SeqBDiag:Block triangular solves only for main diag");
    PetscMemcpy(y,x,m*sizeof(Scalar));
    submat = a->diagv[0];
    for (i=0; i<a->bdlen[0]; i++) {
      LAgetrs_("N",&nb,&one,&submat[i*nb*nb],&nb,&a->pivots[i*nb],
               y+i*nb,&nb,&info);
      if (info) SETERRQ(1,"MatSolve_SeqBDiag:Bad triangular solve");
    }
  }
  return 0;
}

int MatSolveTrans_SeqBDiag(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          one = 1, info, i, ierr, nb = a->nb, m = a->m;
  Scalar       *x, *y, *submat;

  if (a->nd != 1 || a->diag[0] !=0) SETERRQ(1,
    "MatSolveTrans_SeqBDiag:Triangular solves only for main diag");
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y); CHKERRQ(ierr);
  PetscMemcpy(y,x,m*sizeof(Scalar));
  if (A->factor == FACTOR_LU) {
    submat = a->diagv[0];
    for (i=0; i<a->bdlen[0]; i++) {
      LAgetrs_("T",&nb,&one,&submat[i*nb*nb],&nb,&a->pivots[i*nb],y+i*nb,&nb,&info);
    }
  }
  else SETERRQ(1,"MatSolveTrans_SeqBDiag:Matrix must be factored to solve");
  if (info) SETERRQ(1,"MatSolveTrans_SeqBDiag:Bad solve");
  return 0;
}

int MatSolveAdd_SeqBDiag(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          one = 1, info, ierr, i, nb = a->nb, m = a->m;
  Scalar       *x, *y, sone = 1.0, *submat;
  Vec          tmp = 0;

  if (a->nd != 1 || a->diag[0] !=0) SETERRQ(1,
    "MatSolveAdd_SeqBDiag:Triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(A,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PetscMemcpy(y,x,m*sizeof(Scalar));
  if (A->factor == FACTOR_LU) {
    submat = a->diagv[0];
    for (i=0; i<a->bdlen[0]; i++) {
      LAgetrs_("N",&nb,&one,&submat[i*nb*nb],&nb,&a->pivots[i*nb],y+i*nb,&nb,&info);
    }
  }
  if (info) SETERRQ(1,"MatSolveAdd_SeqBDiag:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
int MatSolveTransAdd_SeqBDiag(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqBDiag  *a = (Mat_SeqBDiag *) A->data;
  int           one = 1, info,ierr, i, nb = a->nb, m = a->m;
  Scalar        *x, *y, sone = 1.0, *submat;
  Vec           tmp;

  if (a->nd != 1 || a->diag[0] !=0) SETERRQ(1,
    "MatSolveTransAdd_SeqBDiag:Triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(A,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PetscMemcpy(y,x,m*sizeof(Scalar));
  if (A->factor == FACTOR_LU) {
    submat = a->diagv[0];
    for (i=0; i<a->bdlen[0]; i++) {
      LAgetrs_("T",&nb,&one,&submat[i*nb*nb],&nb,&a->pivots[i*nb],y+i*nb,&nb,&info);
    }
  }
  if (info) SETERRQ(1,"MatSolveTransAdd_SeqBDiag:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
