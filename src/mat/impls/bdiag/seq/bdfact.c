#ifndef lint
static char vcid[] = "$Id: bdfact.c,v 1.14 1995/10/11 03:53:08 curfman Exp curfman $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "pinclude/plapack.h"

/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot. */

/* 
   BlockMatMult_Private - Computes y -= A*v for square A, stored by columns.
 */

#define BMatMult(n,A,v,y) BlockMatMult_Private(n,A,v,y)
static int BlockMatMult_Private(int n,Scalar *A,Scalar *v,Scalar *y)
{
  Scalar v_i, *Apt = A;
  int    i, j;

  for (i=0; i<n; i++) {
    v_i = v[i];
    for (j=0; j<n; j++) y[j] -= Apt[j] * v_i;
    Apt += n;
  }
 return 0;
}

int MatLUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,double f,Mat *B)
{
  int          ierr;
  PLogInfo((PetscObject)A,"MatLUFactorSymbolic_SeqBDiag: Currently no fill.\n");
  ierr = MatConvert(A,MATSAME,B); CHKERRQ(ierr);
  return 0;
}

int MatILUFactorSymbolic_SeqBDiag(Mat A,IS isrow,IS iscol,double f,
                                  int levels,Mat *B)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          ierr;

  if (a->m != a->n) SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Matrix must be square");
  if (isrow || iscol) PLogInfo((PetscObject)A,
    "MatILUFactorSymbolic_SeqBDiag: row and col permutations not supported.\n");
  if (levels != 0)
    SETERRQ(1,"MatILUFactorSymbolic_SeqBDiag:Only ILU(0) is supported");
  ierr = MatConvert(A,MATSAME,B); CHKERRQ(ierr);
  return 0;
}

int MatLUFactorNumeric_SeqBDiag(Mat A,Mat *B)
{
  Mat          C = *B;
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) C->data;
  int          info, k, d, d2, dgk, elim_row, elim_col, nb = a->nb, knb, knb2, nb2;
  int          dnum,  nd = a->nd, mbk = a->mblock, nbk = a->nblock, **pivot, ierr;
  int          *diag = a->diag, n = a->n, m = a->m, mainbd = a->mainbd, *dgptr;
  Scalar       **dv = a->diagv, *dd = dv[mainbd], mult;

  if (nb == 1) {
    dgptr = (int *) PETSCMALLOC((m+n)*sizeof(int)); CHKPTRQ(dgptr);
    PetscZero(dgptr,(m+n)*sizeof(int));
    for ( k=0; k<nd; k++ ) dgptr[diag[k]+m] = k+1;
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
                if ((dnum = dgptr[dgk+m])) {
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
    if (!a->pivot) {
      a->pivot = (int **) PETSCMALLOC(nd*sizeof(int*)); CHKPTRQ(a->pivot);
      for (d=0; d<nd; d++)
        a->pivot[d] = (int *) PETSCMALLOC(m*sizeof(int)); CHKPTRQ(a->pivot[d]);
    }
    pivot = a->pivot;
    nb2 = nb*nb;
    dgptr = (int *) PETSCMALLOC((mbk+nbk)*sizeof(int)); CHKPTRQ(dgptr);
    PetscZero(dgptr,(mbk+nbk)*sizeof(int));
    for ( k=0; k<nd; k++ ) dgptr[diag[k]+mbk] = k+1;
    for ( k=0; k<mbk; k++ ) { /* k = block pivot_row */
      knb = k*nb; knb2 = knb*nb;
      LAgetf2_(&nb,&nb,&dv[0][knb2],&nb,&pivot[0][knb],&info);
      if (info) SETERRQ(1,"MatLUFactorNumeric_SeqBDiag:Bad LU factorization");
      for ( d=mainbd-1; d>=0; d-- ) {
        elim_row = k + diag[d];
        if (elim_row < mbk) { /* sweep down */
    /*    if (dv[d][k] != 0) test if entire block is zero? */
            LAgetrs_("N",&nb,&nb,&dv[0][knb2],&nb,&pivot[d][knb],
                     &dv[d][knb2],&nb,&info);
            if (info) SETERRQ(1,"MatLUFactorNumeric_SeqBDiag:Bad triangular solve");
            for ( d2=d+1; d2<nd; d2++ ) {
              elim_col = elim_row - diag[d2];
              if (elim_col >=0 && elim_col < nbk) {
                dgk = k - elim_col;
                if (dgk > 0) SETERRQ(1,
                   "MatLUFactorNumeric_SeqBDiag:Bad elimination column");
                if ((dnum = dgptr[dgk+mbk])) {
                  if (diag[d2] > 0) {
                    ierr = BMatMult(nb,&dv[d][knb2],&dv[dnum-1][knb2],
                                    &dv[d2][elim_col*nb2]); CHKERRQ(ierr);
                  } else {
                    ierr = BMatMult(nb,&dv[d][knb2],&dv[dnum-1][knb2],
                                    &dv[d2][elim_row*nb2]); CHKERRQ(ierr);
                  }
                }
              }
            }
     /*   }  */
        }
      }
    }
    PETSCFREE(dgptr);
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
  if (a->pivot) {
    for (i=0; i<a->nd; i++) PETSCFREE( a->pivot[i] );
    PETSCFREE(a->pivot);
  }
  if (a->pivot) PETSCFREE(a->pivot);
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
  int          mbk = a->mblock, nbk = a->nblock, inb, inb2;
  int          nb = a->nb, n = a->n, m = a->m, *diag = a->diag, col;
  Scalar       *x, *y, *dd = a->diagv[mainbd], sum, **dv = a->diagv;

  if (A->factor != FACTOR_LU) SETERRQ(1,"MatSolve_SeqBDiag:Not for unfactored matrix.");
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y); CHKERRQ(ierr);

  if (nb == 1) {
    /* forward solve the lower triangular part */
    for (i=0; i<m; i++) {
      sum = x[i];
      for (d=0; d<mainbd; d++) {
        loc = i - diag[d];
        if (loc >= 0) sum -= dv[d][loc] * y[loc];
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
  } else {
    PetscMemcpy(y,x,m*sizeof(Scalar));

    /* forward solve the lower triangular part */
    for (i=0; i<mbk; i++) {
      inb = i*nb;
      for (d=0; d<mainbd; d++) {
        loc = i - diag[d];
        if (loc >= 0) {
          ierr = BMatMult(nb,&dv[d][loc*nb*nb],&y[loc*nb],&y[inb]); CHKERRQ(ierr);
        }
      }
    }
    /* backward solve the upper triangular part */
    for ( i=mbk-1; i>=0; i-- ) {
      inb = i*nb; inb2 = inb*nb;
      for (d=mainbd+1; d<a->nd; d++) {
        col = i - diag[d];
        if (col < nbk) {
          ierr = BMatMult(nb,&dv[d][inb2],&y[col*nb],&y[inb]); CHKERRQ(ierr);
        }
      }
      LAgetrs_("N",&nb,&one,&dv[i][inb2],&nb,&a->pivot[i][inb],
               &y[inb],&nb,&info);
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
      LAgetrs_("T",&nb,&one,&submat[i*nb*nb],&nb,&a->pivot[0][i*nb],y+i*nb,&nb,&info);
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
      LAgetrs_("N",&nb,&one,&submat[i*nb*nb],&nb,&a->pivot[0][i*nb],y+i*nb,&nb,&info);
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
      LAgetrs_("T",&nb,&one,&submat[i*nb*nb],&nb,&a->pivot[0][i*nb],y+i*nb,&nb,&info);
    }
  }
  if (info) SETERRQ(1,"MatSolveTransAdd_SeqBDiag:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
