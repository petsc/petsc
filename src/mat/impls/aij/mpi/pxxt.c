/*$Id: xxt.c,v 1.1 2000/11/30 03:57:22 bsmith Exp bsmith $*/

/* 
        Provides an interface to the Tufo-Fischer parallel direct solver

*/
#include "src/mat/impls/aij/mpi/mpiaij.h"

#if defined(PETSC_HAVE_XXT) && !defined(__cplusplus)
#include "xxt.h"

typedef struct {
  xxt_ADT xxt;
  Vec     b,xd,xo;
  int     nd;
} Mat_MPIAIJ_XXT;


EXTERN int MatDestroy_MPIAIJ(Mat);

#undef __FUNC__  
#define __FUNC__ "MatDestroy_MPIAIJ_XXT"
int MatDestroy_MPIAIJ_XXT(Mat A)
{
  Mat_MPIAIJ     *a   = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJ_XXT *xxt = (Mat_MPIAIJ_XXT*)a->spptr;
  int            ierr;

  PetscFunctionBegin;
  /* free the XXT datastructures */
  XXT_free(xxt->xxt); 
  ierr = PetscFree(xxt);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_MPIAIJ_XXT"
int MatSolve_MPIAIJ_XXT(Mat A,Vec b,Vec x)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJ_XXT *xxt = (Mat_MPIAIJ_XXT*)a->spptr;
  Scalar         *bb,*xx;
  int            ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  XXT_solve(xxt->xxt,xx,bb);
  ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_MPIAIJ_XXT"
int MatLUFactorNumeric_MPIAIJ_XXT(Mat A,Mat *F)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "LocalMult"
static int LocalMult(Mat_MPIAIJ *a,Scalar *xin,Scalar *xout)
{
  int            ierr;
  Mat_MPIAIJ_XXT *xxt = (Mat_MPIAIJ_XXT*)a->spptr;

  PetscFunctionBegin;
  ierr = VecPlaceArray(xxt->b,xout);CHKERRQ(ierr);
  ierr = VecPlaceArray(xxt->xd,xin);CHKERRQ(ierr);
  ierr = VecPlaceArray(xxt->xo,xin+xxt->nd);CHKERRQ(ierr);
  ierr = MatMult(a->A,xxt->xd,xxt->b);CHKERRQ(ierr);
  ierr = MatMultAdd(a->B,xxt->xo,xxt->b,xxt->b);CHKERRQ(ierr);
  /*
  PetscDoubleView(a->A->n+a->B->n,xin,PETSC_VIEWER_STDOUT_WORLD);
  PetscDoubleView(a->A->m,xout,PETSC_VIEWER_STDOUT_WORLD);
  */
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorSymbolic_MPIAIJ_XXT"
int MatLUFactorSymbolic_MPIAIJ_XXT(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat            B;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data,*b;
  int            ierr,err,*localtoglobal,ncol,i;
  Mat_MPIAIJ_XXT *xxt;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square"); 
  ierr = MatCreateMPIAIJ(A->comm,A->m,A->n,A->M,A->N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);
  B                       = *F;
  B->ops->solve           = MatSolve_MPIAIJ_XXT;
  B->ops->destroy         = MatDestroy_MPIAIJ_XXT;
  B->ops->lufactornumeric = MatLUFactorNumeric_MPIAIJ_XXT;
  B->factor               = FACTOR_LU;
  b                       = (Mat_MPIAIJ*)B->data;
  ierr                    = PetscNew(Mat_MPIAIJ_XXT,&xxt);CHKERRQ(ierr);
  b->spptr = a->spptr     = (void*)xxt;

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
  err = XXT_factor(xxt->xxt,localtoglobal,A->m,ncol,(void*)LocalMult,a);
  if (!err) SETERRQ(1,"Error in XXT_factor()");

  ierr = VecDestroy(xxt->b);CHKERRQ(ierr);
  ierr = VecDestroy(xxt->xd);CHKERRQ(ierr);
  ierr = VecDestroy(xxt->xo);CHKERRQ(ierr);
  ierr = PetscFree(localtoglobal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatUseXXT_MPIAIJ"
int MatUseXXT_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_MPIAIJ_XXT;
  PetscLogInfo(0,"Using XXT for MPIAIJ LU factorization and solves");
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ "MatUseXXT_MPIAIJ"
int MatUseXXT_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif



