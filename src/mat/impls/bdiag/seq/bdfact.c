#ifndef lint
static char vcid[] = "$Id: bdfact.c,v 1.6 1995/09/12 03:25:38 bsmith Exp bsmith $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "pinclude/plapack.h"

/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/

int MatLUFactor_SeqBDiag(Mat matin,IS row,IS col,double f)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int    info, i, nb = mat->nb;
  Scalar *submat;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatLUFactor_SeqBDiag: Currently supports factoriz only for main diagonal");
  if (!mat->pivots) {
    mat->pivots = (int *) PETSCMALLOC( mat->m*sizeof(int) );
    CHKPTRQ(mat->pivots);
  }
  submat = mat->diagv[0];
  for (i=0; i<mat->bdlen[0]; i++) {
    LAgetrf_(&nb,&nb,&submat[i*nb*nb],&nb,&mat->pivots[i*nb],&info);
    if (info) SETERRQ(1,"MatLUFactor_SeqBDiag:Bad LU factorization");
  }
  matin->factor = FACTOR_LU;
  return 0;
}

int MatLUFactorSymbolic_SeqBDiag(Mat matin,IS row,IS col,double f,
                                     Mat *fact)
{
  int ierr;
  ierr = MatConvert(matin,MATSAME,fact); CHKERRQ(ierr);
  return 0;
}

int MatLUFactorNumeric_SeqBDiag(Mat matin,Mat *fact)
{
  return MatLUFactor(*fact,0,0,1.0);
}

int MatSolve_SeqBDiag(Mat matin,Vec xx,Vec yy)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int    one = 1, info, i, nb = mat->nb;
  Scalar *x, *y;
  Scalar *submat;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolve_SeqBDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    submat = mat->diagv[0];
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "N", &nb, &one, &submat[i*nb*nb], &nb, &mat->pivots[i*nb], 
              y+i*nb, &nb, &info );
    }
  }
  else SETERRQ(1,"MatSolve_SeqBDiag:Matrix must be factored to solve");
  if (info) SETERRQ(1,"MatSolve_SeqBDiag:Bad solve");
  return 0;
}

int MatSolveTrans_SeqBDiag(Mat matin,Vec xx,Vec yy)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int    one = 1, info, i, nb = mat->nb;
  Scalar *x, *y;
  Scalar *submat;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolveTrans_SeqBDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    submat = mat->diagv[0];
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "T", &nb, &one, &submat[i*nb*nb], &nb, &mat->pivots[i*nb], 
              y+i*nb, &nb, &info );
    }
  }
  else SETERRQ(1,"MatSolveTrans_SeqBDiag:Matrix must be factored to solve");
  if (info) SETERRQ(1,"MatSolveTrans_SeqBDiag:Bad solve");
  return 0;
}

int MatSolveAdd_SeqBDiag(Mat matin,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int    one = 1, info, ierr, i, nb = mat->nb;
  Scalar *x, *y, sone = 1.0;
  Vec    tmp = 0;
  Scalar *submat;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolveAdd_SeqBDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(matin,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    submat = mat->diagv[0];
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "N", &nb, &one, &submat[i*nb*nb], &nb, &mat->pivots[i*nb], 
              y+i*nb, &nb, &info );
    }
  }
  if (info) SETERRQ(1,"MatSolveAdd_SeqBDiag:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
int MatSolveTransAdd_SeqBDiag(Mat matin,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqBDiag  *mat = (Mat_SeqBDiag *) matin->data;
  int     one = 1, info,ierr, i, nb = mat->nb;
  Scalar  *x, *y, sone = 1.0;
  Vec     tmp;
  Scalar *submat;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolveTransAdd_SeqBDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(matin,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    submat = mat->diagv[0];
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "T", &nb, &one, &submat[i*nb*nb], &nb, &mat->pivots[i*nb], 
              y+i*nb, &nb, &info );
    }
  }
  if (info) SETERRQ(1,"MatSolveTransAdd_SeqBDiag:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
