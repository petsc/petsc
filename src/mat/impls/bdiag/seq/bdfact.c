#ifndef lint
static char vcid[] = "$Id: bdfact.c,v 1.9 1995/10/05 01:31:43 curfman Exp curfman $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "pinclude/plapack.h"

/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/

int MatLUFactorSymbolic_SeqBDiag(Mat A,IS row,IS col,double f,Mat *fact)
{
  return MatConvert(A,MATSAME,fact);
}

int MatILUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,double f,
                                  int levels,Mat *fact)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          ierr, n = a->m;

  if (n != a->n) SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Matrix must be square");
  if (isrow || iscol) PLogInfo((PetscObject)A,
    "MatILUFactorSymbolic_SeqBDiag: row and col permutations not supported.\n");
  if (levels != 0)
    SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Only ILU(0) is supported");
  ierr = MatConvert(A,MATSAME,fact); CHKERRQ(ierr);
  return 0;
}

int MatLUFactorNumeric_SeqBDiag(Mat A,Mat *B)
{
  Mat          C = *B;
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) C->data;
  int          info, i, k, d, d2, dgk, pivot_row, col, nb = a->nb, nd = a->nd;
  int          *diag = a->diag, ds, m = a->m, mainbd = a->mainbd;
  Scalar       *submat, **dv = a->diagv, *dd = dv[mainbd], mult;

  if (nb == 1) {
    for ( k=0; k<m; k++ ) {
      dd[k] = 1.0/dd[k];
      for ( d=mainbd-1; d>=0; d-- ) { /*      for ( d=0; d<mainbd; d++ ) { */
        pivot_row = k + diag[d];
        if (diag[d] >= k && pivot_row < m) {
          if (dv[d][k] != 0) { /* nonzero pivot */
            dv[d][k] *= dd[k];
            mult = dv[d][k];
            for ( d2=d+1; d2<nd; d2++ ) {
              col = pivot_row - diag[d2];
              dgk = k - col;
              if (diag[d2] > 0) { /* lower triangle */
                if (dgk > 0) { /* lower triangle */
                  for ( ds=0; ds<mainbd; ds++ ) {
                    if (diag[ds] <= dgk) {
                      if (diag[ds] == dgk) dv[d2][col] -= mult * dv[ds][col];
                      break;
                    }
                  }
                } else { /* upper triangle */
                  for ( ds=mainbd; ds<nd; ds++ ) {
                    if (diag[ds] <= dgk) {
                      if (diag[ds] == dgk) dv[d2][col] -= mult * dv[ds][k];
                      break;
                    }
                  }
                }
              } else { /* upper triangle */
                if (dgk > 0) { /* lower triangle */
                  for ( ds=0; ds<mainbd; ds++ ) {
                    if (diag[ds] <= dgk) {
                      if (diag[ds] == dgk) dv[d2][pivot_row] -= mult * dv[ds][col];
                      break;
                    }
                  }
                } else { /* upper triangle */
                  for ( ds=mainbd; ds<nd; ds++ ) {
                    if (diag[ds] <= dgk) {
                      if (diag[ds] == dgk) dv[d2][pivot_row] -= mult * dv[ds][k];
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
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
  Scalar       *x, *y, *dvmain = a->diagv[mainbd], sum;
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
      y[i] = sum/dvmain[i];
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
