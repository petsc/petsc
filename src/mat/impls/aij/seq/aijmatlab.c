/*$Id: aijmatlab.c,v 1.1 2000/05/09 03:30:27 bsmith Exp bsmith $*/

/* 
        Provides an interface for the Matlab engine sparse solver

*/
#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_MATLAB)
#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */

#undef __FUNC__  
#define __FUNC__ /*<a name="MatSolve_SeqAIJ_Matlab"></a>*/"MatSolve_SeqAIJ_Matlab"
int MatSolve_SeqAIJ_Matlab(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  int             ierr;
  char            *_A,*_b,*_x;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)A,&_A);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)b,&_b);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)x,&_x);CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(MATLAB_ENGINE_(A->comm),(PetscObject)b);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(MATLAB_ENGINE_(A->comm),"u_%s\\(l_%s\\(p_%s*%s'))",_A,_A,_A,_A);CHKERRQ(ierr);
  ierr = PetscMatlabEnginePrintOutput(MATLAB_ENGINE_(A->comm),stdout);CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(MATLAB_ENGINE_(A->comm),(PetscObject)x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatLUFactorNumeric_SeqAIJ_Matlab"></a>*/"MatLUFactorNumeric_SeqAIJ_Matlab"
int MatLUFactorNumeric_SeqAIJ_Matlab(Mat A,Mat *F)
{
  Mat_SeqAIJ      *f = (Mat_SeqAIJ*)(*F)->data;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)(A)->data;
  int             ierr,len;
  char            *_A,*name;

  PetscFunctionBegin;
  ierr = PetscMatlabEnginePut(MATLAB_ENGINE_(A->comm),(PetscObject)A);CHKERRQ(ierr);
  _A   = A->name;
  ierr = PetscMatlabEngineEvaluate(MATLAB_ENGINE_(A->comm),"[l_%s,u_%s,p_%s] = lu(%s');",_A,_A,_A,_A);CHKERRQ(ierr);

  ierr = PetscStrlen(_A,&len);CHKERRQ(ierr);
  name = (char*)PetscMalloc((len+2)*sizeof(char));CHKPTRQ(name);
  sprintf(name,"_%s",_A);
  ierr = PetscObjectSetName((PetscObject)*F,name);CHKERRQ(ierr);
  ierr = PetscFree(name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatLUFactorSymbolic_SeqAIJ_Matlab"></a>*/"MatLUFactorSymbolic_SeqAIJ_Matlab"
int MatLUFactorSymbolic_SeqAIJ_Matlab(Mat A,IS r,IS c,PetscReal f,Mat *F)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  int             ierr;

  PetscFunctionBegin;
  if (A->N != A->M) SETERRQ(PETSC_ERR_ARG_SIZ,0,"matrix must be square"); 
  ierr                       = MatCreateSeqAIJ(A->comm,a->m,a->n,0,PETSC_NULL,F);CHKERRQ(ierr);
  (*F)->ops->solve           = MatSolve_SeqAIJ_Matlab;
  (*F)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_Matlab;
  (*F)->factor               = FACTOR_LU;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="MatUseMatlab_SeqAIJ"></a>*/"MatUseMatlab_SeqAIJ"
int MatUseMatlab_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_Matlab;
  PetscFunctionReturn(0);
}

#else

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"MatUseMatlab_SeqAIJ"
int MatUseMatlab_SeqAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


