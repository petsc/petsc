/*$Id: pxxt.c,v 1.8 2001/08/07 03:02:49 balay Exp $*/

/* 
        Provides an interface to the Tufo-Fischer parallel direct solver

*/
#include "src/mat/impls/aij/mpi/mpiaij.h"

#if !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#include "src/contrib/libtfs/xxt.h"

typedef struct {
  xxt_ADT xxt;
  Vec     b,xd,xo;
  int     nd;
} Mat_MPIAIJ_XXT;


EXTERN int MatDestroy_MPIAIJ(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_XXT"
int MatDestroy_MPIAIJ_XXT(Mat A)
{
  Mat_MPIAIJ_XXT *xxt = (Mat_MPIAIJ_XXT*)A->spptr;
  int            ierr;

  PetscFunctionBegin;
  /* free the XXT datastructures */
  ierr = XXT_free(xxt->xxt);CHKERRQ(ierr); 
  ierr = PetscFree(xxt);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJ_XXT"
int MatSolve_MPIAIJ_XXT(Mat A,Vec b,Vec x)
{
  Mat_MPIAIJ_XXT *xxt = (Mat_MPIAIJ_XXT*)A->spptr;
  PetscScalar    *bb,*xx;
  int            ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = XXT_solve(xxt->xxt,xx,bb);CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_MPIAIJ_XXT"
int MatLUFactorNumeric_MPIAIJ_XXT(Mat A,Mat *F)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "LocalMult"
static int LocalMult_XXT(Mat A,PetscScalar *xin,PetscScalar *xout)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data; 
  int            ierr;
  Mat_MPIAIJ_XXT *xxt = (Mat_MPIAIJ_XXT*)A->spptr;

  PetscFunctionBegin;
  ierr = VecPlaceArray(xxt->b,xout);CHKERRQ(ierr);
  ierr = VecPlaceArray(xxt->xd,xin);CHKERRQ(ierr);
  ierr = VecPlaceArray(xxt->xo,xin+xxt->nd);CHKERRQ(ierr);
  ierr = MatMult(a->A,xxt->xd,xxt->b);CHKERRQ(ierr);
  ierr = MatMultAdd(a->B,xxt->xo,xxt->b,xxt->b);CHKERRQ(ierr);
  /*
  PetscRealView(a->A->n+a->B->n,xin,PETSC_VIEWER_STDOUT_WORLD);
  PetscRealView(a->A->m,xout,PETSC_VIEWER_STDOUT_WORLD);
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_XXT"
int MatLUFactorSymbolic_MPIAIJ_XXT(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat            B;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  int            ierr,*localtoglobal,ncol,i;
  Mat_MPIAIJ_XXT *xxt;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr = MatCreateMPIAIJ(A->comm,A->m,A->n,A->M,A->N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
  B                       = *F;
  B->ops->solve           = MatSolve_MPIAIJ_XXT;
  B->ops->destroy         = MatDestroy_MPIAIJ_XXT;
  B->ops->lufactornumeric = MatLUFactorNumeric_MPIAIJ_XXT;
  B->factor               = FACTOR_LU;
  ierr                    = PetscNew(Mat_MPIAIJ_XXT,&xxt);CHKERRQ(ierr);
  B->spptr = A->spptr     = (void*)xxt;

  xxt->xxt = XXT_new();
  /* generate the local to global mapping */
  ncol = a->A->n + a->B->n;
  ierr = PetscMalloc((ncol)*sizeof(int),&localtoglobal);CHKERRQ(ierr);
  for (i=0; i<a->A->n; i++) {
    localtoglobal[i] = a->cstart + i + 1;
  }
  for (i=0; i<a->B->n; i++) {
    localtoglobal[i+a->A->n] = a->garray[i] + 1;
  }
  /*
  PetscIntView(ncol,localtoglobal,PETSC_VIEWER_STDOUT_WORLD);
  */

  /* generate the vectors needed for the local solves */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->A->m,PETSC_NULL,&xxt->b);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->A->n,PETSC_NULL,&xxt->xd);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->B->n,PETSC_NULL,&xxt->xo);CHKERRQ(ierr);
  xxt->nd = a->A->n;

  /* factor the beast */
  ierr = XXT_factor(xxt->xxt,localtoglobal,A->m,ncol,(void*)LocalMult_XXT,a);CHKERRQ(ierr);

  ierr = VecDestroy(xxt->b);CHKERRQ(ierr);
  ierr = VecDestroy(xxt->xd);CHKERRQ(ierr);
  ierr = VecDestroy(xxt->xo);CHKERRQ(ierr);
  ierr = PetscFree(localtoglobal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#include "src/contrib/libtfs/xyt.h"

typedef struct {
  xyt_ADT xyt;
  Vec     b,xd,xo;
  int     nd;
} Mat_MPIAIJ_XYT;


EXTERN int MatDestroy_MPIAIJ(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_XYT"
int MatDestroy_MPIAIJ_XYT(Mat A)
{
  Mat_MPIAIJ_XYT *xyt = (Mat_MPIAIJ_XYT*)A->spptr;
  int            ierr;

  PetscFunctionBegin;
  /* free the XYT datastructures */
  ierr = XYT_free(xyt->xyt);CHKERRQ(ierr);
  ierr = PetscFree(xyt);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJ_XYT"
int MatSolve_MPIAIJ_XYT(Mat A,Vec b,Vec x)
{
  Mat_MPIAIJ_XYT *xyt = (Mat_MPIAIJ_XYT*)A->spptr;
  PetscScalar    *bb,*xx;
  int            ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = XYT_solve(xyt->xyt,xx,bb);CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_MPIAIJ_XYT"
int MatLUFactorNumeric_MPIAIJ_XYT(Mat A,Mat *F)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "LocalMult"
static int LocalMult_XYT(Mat A,PetscScalar *xin,PetscScalar *xout)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data; 
  int            ierr;
  Mat_MPIAIJ_XYT *xyt = (Mat_MPIAIJ_XYT*)A->spptr;

  PetscFunctionBegin;
  ierr = VecPlaceArray(xyt->b,xout);CHKERRQ(ierr);
  ierr = VecPlaceArray(xyt->xd,xin);CHKERRQ(ierr);
  ierr = VecPlaceArray(xyt->xo,xin+xyt->nd);CHKERRQ(ierr);
  ierr = MatMult(a->A,xyt->xd,xyt->b);CHKERRQ(ierr);
  ierr = MatMultAdd(a->B,xyt->xo,xyt->b,xyt->b);CHKERRQ(ierr);
  /*
  PetscRealView(a->A->n+a->B->n,xin,PETSC_VIEWER_STDOUT_WORLD);
  PetscRealView(a->A->m,xout,PETSC_VIEWER_STDOUT_WORLD);
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_XYT"
int MatLUFactorSymbolic_MPIAIJ_XYT(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat            B;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  int            ierr,*localtoglobal,ncol,i;
  Mat_MPIAIJ_XYT *xyt;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr = MatCreateMPIAIJ(A->comm,A->m,A->n,A->M,A->N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
  B                       = *F;
  B->ops->solve           = MatSolve_MPIAIJ_XYT;
  B->ops->destroy         = MatDestroy_MPIAIJ_XYT;
  B->ops->lufactornumeric = MatLUFactorNumeric_MPIAIJ_XYT;
  B->factor               = FACTOR_LU;
  ierr                    = PetscNew(Mat_MPIAIJ_XYT,&xyt);CHKERRQ(ierr);
  B->spptr = A->spptr     = (void*)xyt;

  xyt->xyt = XYT_new();
  /* generate the local to global mapping */
  ncol = a->A->n + a->B->n;
  ierr = PetscMalloc((ncol)*sizeof(int),&localtoglobal);CHKERRQ(ierr);
  for (i=0; i<a->A->n; i++) {
    localtoglobal[i] = a->cstart + i + 1;
  }
  for (i=0; i<a->B->n; i++) {
    localtoglobal[i+a->A->n] = a->garray[i] + 1;
  }
  /*
  PetscIntView(ncol,localtoglobal,PETSC_VIEWER_STDOUT_WORLD);
  */

  /* generate the vectors needed for the local solves */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->A->m,PETSC_NULL,&xyt->b);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->A->n,PETSC_NULL,&xyt->xd);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,a->B->n,PETSC_NULL,&xyt->xo);CHKERRQ(ierr);
  xyt->nd = a->A->n;

  /* factor the beast */
  ierr = XYT_factor(xyt->xyt,localtoglobal,A->m,ncol,(void*)LocalMult_XYT,a);CHKERRQ(ierr);

  ierr = VecDestroy(xyt->b);CHKERRQ(ierr);
  ierr = VecDestroy(xyt->xd);CHKERRQ(ierr);
  ierr = VecDestroy(xyt->xo);CHKERRQ(ierr);
  ierr = PetscFree(localtoglobal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_TFS"
int MatLUFactorSymbolic_MPIAIJ_TFS(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  int ierr;

  PetscFunctionBegin;
  PetscLogInfo(0,"Using TFS for MPIAIJ LU factorization and solves");
  if (A->symmetric) {
    ierr = MatLUFactorSymbolic_MPIAIJ_XXT(A,r,c,info,F);CHKERRQ(ierr);
  } else {
    ierr = MatLUFactorSymbolic_MPIAIJ_XYT(A,r,c,info,F);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "dummy83"
int dummy83(int a)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif



