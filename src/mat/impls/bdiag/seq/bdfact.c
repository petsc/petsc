#ifndef lint
static char vcid[] = "$Id: bdiag.c,v 1.32 1995/07/28 04:22:17 bsmith Exp $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "plapack.h"

/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/

int MatLUFactor_BDiag(Mat matin,IS row,IS col,double f)
{
  Mat_BDiag *mat = (Mat_BDiag *) matin->data;
  int    info, i, nb = mat->nb;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatLUFactor_BDiag: Currently supports factoriz only for main diagonal");
  if (!mat->pivots) {
    mat->pivots = (int *) PETSCMALLOC( mat->m*sizeof(int) );
    CHKPTRQ(mat->pivots);
  }
  for (i=0; i<mat->bdlen[0]; i++) {
    LAgetrf_(&nb,&nb,mat->diagv[0]+i*nb*nb,&nb,mat->pivots+i*nb,&info);
    if (info) SETERRQ(1,"MatLUFactor_BDiag:Bad LU factorization");
  }
  matin->factor = FACTOR_LU;
  return 0;
}

int MatLUFactorSymbolic_BDiag(Mat matin,IS row,IS col,double f,
                                     Mat *fact)
{
  int ierr;
  ierr = MatConvert(matin,MATSAME,fact); CHKERRQ(ierr);
  return 0;
}

int MatLUFactorNumeric_BDiag(Mat matin,Mat *fact)
{
  return MatLUFactor(*fact,0,0,1.0);
}

int MatSolve_BDiag(Mat matin,Vec xx,Vec yy)
{
  Mat_BDiag *mat = (Mat_BDiag *) matin->data;
  int    one = 1, info, i, nb = mat->nb;
  Scalar *x, *y;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolve_BDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "N", &nb, &one, mat->diagv[0]+i*nb*nb, &nb, mat->pivots+i*nb,
              y+i*nb, &nb, &info );
    }
  }
  else SETERRQ(1,"MatSolve_BDiag:Matrix must be factored to solve");
  if (info) SETERRQ(1,"MatSolve_BDiag:Bad solve");
  return 0;
}

int MatSolveTrans_BDiag(Mat matin,Vec xx,Vec yy)
{
  Mat_BDiag *mat = (Mat_BDiag *) matin->data;
  int    one = 1, info, i, nb = mat->nb;
  Scalar *x, *y;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolveTrans_BDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "T", &nb, &one, mat->diagv[0]+i*nb*nb, &nb, mat->pivots+i*nb,
              y+i*nb, &nb, &info );
    }
  }
  else SETERRQ(1,"MatSolveTrans_BDiag:Matrix must be factored to solve");
  if (info) SETERRQ(1,"MatSolveTrans_BDiag:Bad solve");
  return 0;
}

int MatSolveAdd_BDiag(Mat matin,Vec xx,Vec zz,Vec yy)
{
  Mat_BDiag *mat = (Mat_BDiag *) matin->data;
  int    one = 1, info, ierr, i, nb = mat->nb;
  Scalar *x, *y, sone = 1.0;
  Vec    tmp = 0;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolveAdd_BDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "N", &nb, &one, mat->diagv[0]+i*nb*nb, &nb, mat->pivots+i*nb,
              y+i*nb, &nb, &info );
    }
  }
  if (info) SETERRQ(1,"MatSolveAdd_BDiag:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
int MatSolveTransAdd_BDiag(Mat matin,Vec xx,Vec zz,Vec yy)
{
  Mat_BDiag  *mat = (Mat_BDiag *) matin->data;
  int     one = 1, info,ierr, i, nb = mat->nb;
  Scalar  *x, *y, sone = 1.0;
  Vec     tmp;

  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatSolveTransAdd_BDiag: Currently supports triangular solves only for main diag");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    for (i=0; i<mat->bdlen[0]; i++) {
      LAgetrs_( "T", &nb, &one, mat->diagv[0]+i*nb*nb, &nb, mat->pivots+i*nb,
              y+i*nb, &nb, &info );
    }
  }
  if (info) SETERRQ(1,"MatSolveTransAdd_BDiag:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
