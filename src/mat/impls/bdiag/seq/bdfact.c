#ifndef lint
static char vcid[] = "$Id: bdfact.c,v 1.1 1995/07/28 21:21:21 curfman Exp curfman $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "plapack.h"

/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/

int MatLUFactor_BDiag(Mat matin,IS row,IS col,double f)
{
  Mat_BDiag *mat = (Mat_BDiag *) matin->data;
  int    info, i, nb = mat->nb, j, k, *piv;
  Scalar *submat, *smi;

  MatView(matin,STDOUT_VIEWER);
  if (mat->nd != 1 || mat->diag[0] !=0) SETERRQ(1,
    "MatLUFactor_BDiag: Currently supports factoriz only for main diagonal");
  if (!mat->pivots) {
    mat->pivots = (int *) PETSCMALLOC( mat->m*sizeof(int) );
    CHKPTRQ(mat->pivots);
  }
  submat = mat->diagv[0];
  for (i=0; i<mat->bdlen[0]; i++) {
    smi = &submat[i*nb*nb];
    piv = &mat->pivots[i*nb];
    for (j=0; j<nb; j++) {
      printf("block number=%d, i=%d Before factoriz\n",i,j);
      for (k=0; k<nb; k++) {
         printf("  i=%d, j=%d, Aij=%g\n",j,k,smi[j*nb+k]);
      }
    }
    LAgetrf_(&nb,&nb,smi,&nb,piv,&info);
/*    LAgetrf_(&nb,&nb,&submat[i*nb*nb],&nb,&mat->pivots[i*nb],&info); */
    for (j=0; j<nb; j++) {
      printf("block number=%d, i=%d, pivot=%d\n",i,j,piv[j]);
      for (k=0; k<nb; k++) {
         printf("   i=%d, j=%d, Aij=%g\n",j,k,smi[j*nb+k]);
      }
    }
    printf("bd = %d, LAPACK info = %d\n",i,info);
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
      printf("bd = %d,LAPACK info = %d\n",i,info);
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
